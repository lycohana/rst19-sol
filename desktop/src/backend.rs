use serde_json::{json, Value};
use std::{
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{mpsc, Arc, Mutex},
    thread,
};

/// Every desktop operation runs in the same native Rust worker.
pub struct Backend {
    executable: PathBuf,
    root: PathBuf,
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
        let name = if cfg!(windows) {
            "rst19-native-worker.exe"
        } else {
            "rst19-native-worker"
        };
        let project = std::env::var_os("RST19_PROJECT_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .expect("desktop project parent")
                    .to_owned()
            });
        if let Some(executable) = std::env::var_os("RST19_NATIVE_WORKER").map(PathBuf::from) {
            if !executable.is_file() {
                return Err(format!(
                    "指定的 Rust worker 不存在：{}",
                    executable.display()
                ));
            }
            return Ok(Self {
                executable,
                root: project,
            });
        }
        let bundled = package.join("runtime").join(name);
        if bundled.is_file() {
            return Ok(Self {
                executable: bundled,
                root: package.to_owned(),
            });
        }
        let executable = std::iter::once(package.join(name))
            .chain(
                ["release", "debug"]
                    .into_iter()
                    .map(|profile| project.join("desktop/target").join(profile).join(name)),
            )
            .find(|p| p.is_file())
            .ok_or("找不到 Rust worker。请运行桌面构建命令或安装完整发布包。")?;
        Ok(Self {
            executable,
            root: project,
        })
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn start(&self, request: Option<Value>, working_dir: &Path) -> Result<Job, String> {
        let mut command = Command::new(&self.executable);
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
            .map_err(|e| format!("Rust worker 启动失败：{e}"))?;
        if let Some(request) = request {
            let input = format!("{request}\n");
            if let Err(error) = child
                .stdin
                .take()
                .ok_or("worker 输入流不可用")?
                .write_all(input.as_bytes())
            {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error.to_string());
            }
        } else {
            drop(child.stdin.take());
        }
        let stdout = child.stdout.take().ok_or("worker 输出流不可用")?;
        let stderr = child.stderr.take().ok_or("worker 错误流不可用")?;
        let child = Arc::new(Mutex::new(child));
        let (sender, receiver) = mpsc::channel();
        let errors = sender.clone();
        thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                let _ = errors.send(json!({"event":"log","text":line}));
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
                        let _ = sender.send(json!({"event":"log","text":line}));
                    }
                }
            }
            let status = waiter.lock().expect("worker lock").wait();
            if !terminal {
                let _=sender.send(json!({"event":"done","ok":false,"error":format!("算法进程退出且没有返回结果：{status:?}")}));
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
        let _ = self.child.lock().expect("worker lock").kill();
    }
}
impl Drop for Job {
    fn drop(&mut self) {
        let _ = self.child.lock().expect("worker lock").kill();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_catalog_tools_are_native_and_names_are_unique() {
        let catalog: Value = serde_json::from_str(include_str!("tool_catalog.json")).unwrap();
        let tools = catalog.as_array().unwrap();
        assert_eq!(tools.len(), 71);
        assert!(tools.iter().all(|tool| tool["native"] == true));
        assert_eq!(
            tools
                .iter()
                .map(|tool| tool["name"].as_str().unwrap())
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            tools.len()
        );
    }
    #[test]
    fn cancellation_reaps_native_worker_and_returns_terminal_event() {
        let backend = Backend::discover().unwrap();
        let mut job = backend.start(None, backend.root()).unwrap();
        job.cancel();
        let mut terminal = false;
        for _ in 0..100 {
            match job
                .receiver
                .recv_timeout(std::time::Duration::from_millis(100))
            {
                Ok(event) => {
                    if event["event"] == "done" {
                        terminal = true;
                        break;
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }
        }
        assert!(terminal, "cancelled worker must produce terminal status");
        assert!(job.cancelled);
    }
}
