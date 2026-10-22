use std::sync::Arc;
use std::time::Duration;

use mirror_protocol::FunctionCall;
use pretty_assertions::assert_eq;
use serde_json::Map;
use tokio::sync::mpsc;

use super::*;
use crate::tools::ToolDefinition;

/// Sleeps for `delay`, then reports its name on `finished`.
struct LoggedTool {
    name: &'static str,
    concurrency_safe: bool,
    delay: Duration,
    finished: mpsc::UnboundedSender<&'static str>,
}

impl AnyTool for LoggedTool {
    fn name(&self) -> &str {
        self.name
    }

    fn needs_approval(&self) -> bool {
        false
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name.to_string(),
            description: String::new(),
            parameters: Map::new(),
        }
    }

    fn is_concurrency_safe(&self) -> bool {
        self.concurrency_safe
    }

    fn execute<'a>(
        &'a self,
        _arguments: &'a str,
        _context: &'a ExecutionContext,
    ) -> Pin<Box<dyn Future<Output = ToolOutput> + Send + 'a>> {
        Box::pin(async move {
            tokio::time::sleep(self.delay).await;
            self.finished
                .send(self.name)
                .expect("finish order receiver");
            ToolOutput::success(self.name)
        })
    }
}

struct PanickingTool;

impl AnyTool for PanickingTool {
    fn name(&self) -> &str {
        "explode"
    }

    fn needs_approval(&self) -> bool {
        false
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "explode".to_string(),
            description: String::new(),
            parameters: Map::new(),
        }
    }

    fn is_concurrency_safe(&self) -> bool {
        true
    }

    fn execute<'a>(
        &'a self,
        _arguments: &'a str,
        _context: &'a ExecutionContext,
    ) -> Pin<Box<dyn Future<Output = ToolOutput> + Send + 'a>> {
        Box::pin(async { panic!("tool exploded") })
    }
}

fn logged_tool(
    finished: &mpsc::UnboundedSender<&'static str>,
    name: &'static str,
    concurrency_safe: bool,
    delay: Duration,
) -> Arc<dyn AnyTool> {
    Arc::new(LoggedTool {
        name,
        concurrency_safe,
        delay,
        finished: finished.clone(),
    })
}

fn tool_call(name: &str) -> ToolCall {
    ToolCall {
        id: format!("call-{name}"),
        r#type: "function".to_string(),
        function: Some(FunctionCall {
            name: name.to_string(),
            arguments: "{}".to_string(),
        }),
        ..ToolCall::default()
    }
}

/// The content and success of an output, without its measured duration.
fn outcome(output: &ToolOutput) -> (&str, bool) {
    (&output.content, output.success)
}

#[tokio::test(start_paused = true)]
async fn execute_overlaps_safe_calls_and_runs_unsafe_calls_alone() {
    let (finished, mut finish_order) = mpsc::unbounded_channel();
    let executor = DirectToolExecutor::new(vec![
        logged_tool(&finished, "slow_read", true, Duration::from_secs(5)),
        logged_tool(&finished, "fast_read", true, Duration::ZERO),
        logged_tool(&finished, "write", false, Duration::from_secs(1)),
        logged_tool(&finished, "later_read", true, Duration::ZERO),
    ]);
    let calls = ["slow_read", "fast_read", "write", "later_read"].map(tool_call);

    let outputs = executor
        .execute(
            Uuid::new_v4(),
            &calls,
            &[true; 4],
            &ExecutionContext::default(),
        )
        .await
        .expect("batch");

    drop(executor);
    drop(finished);
    let mut order = Vec::new();
    finish_order.recv_many(&mut order, calls.len()).await;
    assert_eq!(order, ["fast_read", "slow_read", "write", "later_read"]);
    assert_eq!(
        outputs.iter().map(outcome).collect::<Vec<_>>(),
        [
            ("slow_read", true),
            ("fast_read", true),
            ("write", true),
            ("later_read", true),
        ]
    );
}

#[tokio::test(start_paused = true)]
async fn execute_starts_a_queued_call_when_any_running_call_finishes() {
    let (finished, _finish_order) = mpsc::unbounded_channel();
    let executor = DirectToolExecutor::new(vec![
        logged_tool(&finished, "slow_read", true, Duration::from_secs(30)),
        logged_tool(&finished, "fast_read", true, Duration::ZERO),
    ]);
    let calls = [
        "slow_read",
        "fast_read",
        "fast_read",
        "fast_read",
        "fast_read",
        "fast_read",
        "fast_read",
        "fast_read",
        "fast_read",
        "fast_read",
        "slow_read",
    ]
    .map(tool_call);
    let started = tokio::time::Instant::now();

    executor
        .execute(
            Uuid::new_v4(),
            &calls,
            &[true; 11],
            &ExecutionContext::default(),
        )
        .await
        .expect("batch");

    assert_eq!(started.elapsed(), Duration::from_secs(30));
}

#[tokio::test]
async fn execute_turns_a_panic_into_that_call_s_error() {
    let (finished, _finish_order) = mpsc::unbounded_channel();
    let executor = DirectToolExecutor::new(vec![
        Arc::new(PanickingTool),
        logged_tool(&finished, "read", true, Duration::ZERO),
    ]);
    let calls = ["explode", "read"].map(tool_call);

    let outputs = executor
        .execute(
            Uuid::new_v4(),
            &calls,
            &[true; 2],
            &ExecutionContext::default(),
        )
        .await
        .expect("batch");

    assert_eq!(
        outputs.iter().map(outcome).collect::<Vec<_>>(),
        [("Tool error: tool exploded", false), ("read", true)]
    );
}
