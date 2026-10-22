use std::future::Future;
use std::time::Duration;

use mirror_sandbox::STOP_GRACE;
use nix::errno::Errno;
use nix::sys::signal::{killpg, Signal};
use nix::sys::wait::{waitpid, WaitPidFlag, WaitStatus};
use nix::unistd::Pid;
use tokio::sync::watch;
use tokio::time::Instant;

pub enum ProcessSupervision<T> {
    Completed(T),
    Interrupted {
        reason: ProcessInterrupt,
        finished: Option<T>,
        confirmed: bool,
    },
}

#[derive(Clone, Copy)]
pub enum ProcessInterrupt {
    Cancel,
    Deadline,
}

const POLL_INTERVAL: Duration = Duration::from_millis(20);

pub async fn supervise_process<T, F: Future<Output = T>>(
    group: Pid,
    timeout: Duration,
    cancelled: impl Future<Output = ()>,
    execute: impl FnOnce(watch::Receiver<bool>) -> F,
) -> ProcessSupervision<T> {
    let (stopping, stopped) = watch::channel(false);
    let execution = execute(stopped);
    tokio::pin!(execution);
    let reason = tokio::select! {
        result = &mut execution => return ProcessSupervision::Completed(result),
        () = cancelled => ProcessInterrupt::Cancel,
        () = tokio::time::sleep(timeout) => ProcessInterrupt::Deadline,
    };
    stopping.send_replace(true);
    let _ = killpg(group, Signal::SIGTERM);
    let mut finished = tokio::time::timeout(STOP_GRACE, &mut execution).await.ok();
    if finished.is_none() {
        let _ = killpg(group, Signal::SIGKILL);
        finished = tokio::time::timeout(STOP_GRACE, &mut execution).await.ok();
    }
    let confirmed = if finished.is_some() {
        let deadline = Instant::now() + STOP_GRACE;
        loop {
            while matches!(
                waitpid(Pid::from_raw(-group.as_raw()), Some(WaitPidFlag::WNOHANG)),
                Ok(WaitStatus::Exited(..) | WaitStatus::Signaled(..))
            ) {}
            match killpg(group, None::<Signal>) {
                Err(Errno::ESRCH) => break true,
                _ if Instant::now() >= deadline => break false,
                _ => {
                    let _ = killpg(group, Signal::SIGKILL);
                    tokio::time::sleep(POLL_INTERVAL).await;
                }
            }
        }
    } else {
        false
    };
    ProcessSupervision::Interrupted {
        reason,
        finished,
        confirmed,
    }
}
