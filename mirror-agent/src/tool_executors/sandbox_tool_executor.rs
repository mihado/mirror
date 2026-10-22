use std::time::Duration;

use async_trait::async_trait;
use mirror_protocol::ToolCall;

use crate::tools::SandboxCallOutcome;

/// Runs workspace tool calls in a sandbox for [`crate::tools::SandboxRoutedTool`].
#[async_trait]
pub trait SandboxToolExecutor: Send + Sync {
    /// `timeout` bounds the call inside the sandbox.
    async fn execute(&self, call: &ToolCall, timeout: Duration) -> SandboxCallOutcome;
}
