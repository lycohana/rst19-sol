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
    native_worker: Option<PathBuf>,
    root: PathBuf,
    development: bool,
}

pub struct Job {
    pub receiver: mpsc::Receiver<Value>,
    child: Arc<Mutex<Child>>,
    pub cancelled: bool,
}

fn uses_native_worker(request: &Value) -> bool {
    request["operation"] == "preview"
        || request["operation"] == "psf.empirical.estimate"
        || matches!(
            request["operation"].as_str(),
            Some(
                "feedback.append"
                    | "feedback.load"
                    | "photometry.calibration.sensitivity"
                    | "photometry.relative.fit"
                    | "photometry.report"
                    | "sequence.brief"
                    | "sequence.analyze_core"
                    | "mosaic.build"
                    | "sequence.stack_forced_measure_batch"
                    | "wcs.affine.refined"
            )
        )
        || (request["operation"] == "tool"
            && matches!(
                request["tool"].as_str(),
                Some(
                    "rst19"
                        | "rst19-trails"
                        | "rst19-sources"
                        | "rst19-empirical-pair-audit"
                        | "rst19-empirical-pair-fit"
                        | "rst19-empirical-pair-free"
                        | "rst19-multipsf-audit"
                        | "rst19-free-multipsf-audit"
                        | "rst19-forced-stability"
                        | "rst19-feature-sequence"
                        | "rst19-feature-audit-replicates"
                        | "rst19-format-audit"
                        | "rst19-feature-sequence-compare"
                        | "rst19-feature-raw-evidence-audit"
                        | "rst19-signed-null-overlap"
                        | "rst19-signed-null-audit"
                        | "rst19-signed-null-sequence"
                        | "rst19-aux-audit"
                        | "rst19-feature-flag-profile"
                        | "rst19-feature-flag-interaction"
                        | "rst19-feature-gate-margin"
                        | "rst19-feature-effect-size"
                        | "rst19-feature-class-context"
                        | "rst19-pair-mechanism-context"
                        | "rst19-pair-mechanism-sensitivity"
                        | "rst19-pair-pixel-topology"
                        | "rst19-pair-support-audit"
                        | "rst19-aperture-peak-ratio"
                        | "rst19-aperture-sensitivity"
                        | "rst19-wcs-validation"
                        | "rst19-feature-correlation"
                        | "rst19-feature-transition-flow"
                        | "rst19-feature-gate-route"
                        | "rst19-feature-parameter-sensitivity"
                        | "rst19-feature-cross"
                        | "rst19-feature-hard-negative"
                        | "rst19-gaia-merge"
                        | "rst19-feature-evidence-axes"
                        | "rst19-direct-neighbor-audit"
                        | "rst19-feature-cross-axis-audit"
                        | "rst19-source-group-audit"
                        | "rst19-source-group-sensitivity"
                        | "rst19-source-separation-audit"
                        | "rst19-source-peak-consistency"
                        | "rst19-source-proposal-peak-audit"
                        | "rst19-source-proposal-temporal-audit"
                        | "rst19-source-geometry-audit"
                        | "rst19-innovation-package"
                        | "rst19-temporal-codes"
                        | "rst19-sweep"
                        | "rst19-feature-psf-leaveout"
                        | "rst19-relative-photometry"
                        | "rst19-photometric-report"
                        | "rst19-mosaic"
                        | "rst19-motion-crossing-ablation"
                        | "rst19-contaminated-pair-injection"
                        | "rst19-contaminated-pair-class"
                )
            ))
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
            let native_worker = package.join("runtime").join(if cfg!(windows) {
                "rst19-native-worker.exe"
            } else {
                "rst19-native-worker"
            });
            return Ok(Self {
                executable: worker,
                native_worker: native_worker.is_file().then_some(native_worker),
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
        let native_name = if cfg!(windows) {
            "rst19-native-worker.exe"
        } else {
            "rst19-native-worker"
        };
        let native_worker = std::env::var_os("RST19_NATIVE_WORKER")
            .map(PathBuf::from)
            .filter(|path| path.is_file())
            .or_else(|| {
                ["release", "debug"]
                    .into_iter()
                    .map(|profile| root.join("desktop/target").join(profile).join(native_name))
                    .find(|path| path.is_file())
            });
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
            native_worker,
            root,
            development: true,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn executable_for(&self, request: Option<&Value>) -> Result<&Path, String> {
        if request.is_none() || request.is_some_and(uses_native_worker) {
            return self
                .native_worker
                .as_deref()
                .ok_or_else(|| "找不到 Rust native worker。请重新构建桌面应用。".into());
        }
        Ok(&self.executable)
    }

    pub fn start(&self, request: Option<Value>, working_dir: &Path) -> Result<Job, String> {
        let use_native = request.is_none() || request.as_ref().is_some_and(uses_native_worker);
        if self.development
            && !use_native
            && !self.root.join("src/rst19/desktop_worker.py").is_file()
        {
            return Err("此工具尚未迁移到 Rust，且当前工作区没有 Python 回退模块。".into());
        }
        let executable = self.executable_for(request.as_ref())?;
        let mut command = Command::new(executable);
        if self.development && !use_native {
            command.args(["-u", "-m", "rst19.desktop_worker"]);
            command.env("PYTHONPATH", self.root.join("src"));
        }
        if !use_native {
            command
                .env("PYTHONUTF8", "1")
                .env("PYTHONIOENCODING", "utf-8")
                .env("OMP_NUM_THREADS", "2")
                .env("OPENBLAS_NUM_THREADS", "2");
        }
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
        let mut child = command.spawn().map_err(|e| {
            if use_native {
                format!("Rust native worker 启动失败：{e}")
            } else {
                format!("Python 回退 worker 启动失败：{e}")
            }
        })?;
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
    fn preview_single_frame_and_sequence_core_use_the_native_worker_route() {
        assert!(uses_native_worker(&json!({"operation":"preview"})));
        assert!(uses_native_worker(
            &json!({"operation":"psf.empirical.estimate"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"feedback.load","path":"feedback.jsonl"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"feedback.append","path":"feedback.jsonl"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"photometry.calibration.sensitivity"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"photometry.relative.fit"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"photometry.report"})
        ));
        assert!(uses_native_worker(&json!({"operation":"sequence.brief"})));
        assert!(uses_native_worker(
            &json!({"operation":"sequence.analyze_core"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"sequence.stack_forced_measure_batch"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"wcs.affine.refined"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-format-audit"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-trails"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-sources"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-empirical-pair-audit"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-multipsf-audit"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-free-multipsf-audit"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-forced-stability"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-feature-sequence-compare"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-feature-raw-evidence-audit"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-signed-null-overlap"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-signed-null-audit"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-signed-null-sequence"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-aux-audit"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-feature-flag-profile"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-feature-flag-interaction"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-feature-gate-margin"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-feature-effect-size"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-feature-class-context"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-pair-mechanism-context"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-pair-mechanism-sensitivity"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-pair-pixel-topology"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-pair-support-audit"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-aperture-peak-ratio"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-aperture-sensitivity"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-wcs-validation"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-feature-correlation"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-feature-transition-flow"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-feature-gate-route"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-feature-parameter-sensitivity"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-feature-cross"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-feature-hard-negative"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-gaia-merge"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-feature-evidence-axes"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-direct-neighbor-audit"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-feature-cross-axis-audit"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-source-group-audit"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-source-group-sensitivity"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-source-separation-audit"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-source-peak-consistency"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-source-proposal-peak-audit"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-source-proposal-temporal-audit"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-relative-photometry"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-photometric-report"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-innovation-package"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-temporal-codes"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-sweep"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-feature-psf-leaveout"})
        ));
        assert!(!uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-sequence"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-mosaic"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-motion-crossing-ablation"})
        ));
        assert!(uses_native_worker(
            &json!({"operation":"tool","tool":"rst19-contaminated-pair-injection"})
        ));
    }

    #[test]
    fn migrated_operations_never_fall_back_to_python_when_native_worker_is_missing() {
        let backend = Backend {
            executable: PathBuf::from("python"),
            native_worker: None,
            root: PathBuf::new(),
            development: true,
        };
        let request = json!({"operation":"tool","tool":"rst19"});
        let error = backend.executable_for(Some(&request)).unwrap_err();
        assert!(error.contains("Rust native worker"));
        let psf_request = json!({"operation":"psf.empirical.estimate"});
        assert!(backend
            .executable_for(Some(&psf_request))
            .unwrap_err()
            .contains("Rust native worker"));
        assert!(backend
            .executable_for(None)
            .unwrap_err()
            .contains("Rust native worker"));
        let backend = Backend {
            executable: PathBuf::from("python"),
            native_worker: Some(PathBuf::from("rst19-native-worker")),
            root: PathBuf::new(),
            development: true,
        };
        assert_eq!(
            backend.executable_for(None).unwrap(),
            Path::new("rst19-native-worker")
        );
        let pending = json!({"operation":"tool","tool":"rst19-sequence"});
        assert_eq!(
            backend.executable_for(Some(&pending)).unwrap(),
            Path::new("python")
        );
    }

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
