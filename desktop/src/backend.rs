use serde_json::{json, Value};
use std::{
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{mpsc, Arc, Mutex},
    thread,
};

pub struct Backend {
    executable: PathBuf,
    root: PathBuf,
    development: bool,
}

pub struct Job {
    pub receiver: mpsc::Receiver<Value>,
    child: Arc<Mutex<Child>>,
    pub cancelled: bool,
}

impl Backend {
    pub fn discover() -> Result<Self, String> {
        let exe = std::env::current_exe().map_err(|e| e.to_string())?;
        let package = exe.parent().ok_or("无法定位安装目录")?;
        let worker = package.join("runtime").join(if cfg!(windows) {
            "rst19-worker.exe"
        } else {
            "rst19-worker"
        });
        if worker.is_file() {
            return Ok(Self {
                executable: worker,
                root: package.to_owned(),
                development: false,
            });
        }
        let root = std::env::var_os("RST19_PROJECT_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .unwrap()
                    .to_owned()
            });
        if !root.join("src/rst19/desktop_worker.py").is_file() {
            return Err("缺少内置算法环境 runtime/rst19-worker，请使用完整发布包。".into());
        }
        let executable = std::env::var_os("RST19_PYTHON")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                let venv = root.join(if cfg!(windows) {
                    ".venv/Scripts/python.exe"
                } else {
                    ".venv/bin/python"
                });
                if venv.is_file() {
                    venv
                } else {
                    PathBuf::from(if cfg!(windows) { "python" } else { "python3" })
                }
            });
        Ok(Self {
            executable,
            root,
            development: true,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn start(&self, request: Option<Value>, working_dir: &Path) -> Result<Job, String> {
        let mut command = Command::new(&self.executable);
        if self.development {
            command.args(["-u", "-m", "rst19.desktop_worker"]);
            command.env("PYTHONPATH", self.root.join("src"));
        }
        command
            .env("PYTHONUTF8", "1")
            .env("PYTHONIOENCODING", "utf-8");
        command
            .env("OMP_NUM_THREADS", "2")
            .env("OPENBLAS_NUM_THREADS", "2");
        if request.is_none() {
            command.arg("--catalog");
        }
        command
            .current_dir(working_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000);
        }
        let mut child = command
            .spawn()
            .map_err(|e| format!("算法环境启动失败：{e}"))?;
        if let Some(request) = request {
            let input = format!("{}\n", request);
            if let Err(error) = child.stdin.take().unwrap().write_all(input.as_bytes()) {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error.to_string());
            }
        } else {
            drop(child.stdin.take());
        }
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();
        let child = Arc::new(Mutex::new(child));
        let (sender, receiver) = mpsc::channel();
        let errors = sender.clone();
        thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                let _ = errors.send(json!({"event": "log", "text": line}));
            }
        });
        let waiter = child.clone();
        thread::spawn(move || {
            let mut terminal = false;
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                match serde_json::from_str::<Value>(&line) {
                    Ok(event) => {
                        terminal |= event["event"] == "done";
                        let _ = sender.send(event);
                    }
                    Err(_) => {
                        let _ = sender.send(json!({"event":"log", "text":line}));
                    }
                }
            }
            let status = waiter.lock().unwrap().wait();
            if !terminal {
                let _ = sender.send(json!({"event":"done", "ok":false,
                    "error":format!("算法进程退出且没有返回结果：{status:?}")}));
            }
        });
        Ok(Job {
            receiver,
            child,
            cancelled: false,
        })
    }
}

impl Job {
    pub fn cancel(&mut self) {
        self.cancelled = true;
        let _ = self.child.lock().unwrap().kill();
    }
}
impl Drop for Job {
    fn drop(&mut self) {
        let _ = self.child.lock().unwrap().kill();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cancellation_reaps_the_worker_and_produces_a_terminal_event() {
        let backend = Backend::discover().unwrap();
        let mut job = backend.start(None, backend.root()).unwrap();
        job.cancel();
        let mut terminal = None;
        for _ in 0..100 {
            match job
                .receiver
                .recv_timeout(std::time::Duration::from_secs(10))
            {
                Ok(event) if event["event"] == "done" => {
                    terminal = Some(event);
                    break;
                }
                Ok(_) => continue,
                Err(error) => panic!("Cancelled worker did not terminate: {error}"),
            }
        }
        assert!(job.cancelled);
        assert!(terminal.is_some());
        assert!(job.child.lock().unwrap().try_wait().unwrap().is_some());
    }
}
