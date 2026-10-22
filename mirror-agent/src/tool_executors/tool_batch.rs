//! Execution order for the calls of one tool batch.

use std::panic::AssertUnwindSafe;

use futures_util::future::BoxFuture;
use futures_util::stream;
use futures_util::FutureExt;
use futures_util::StreamExt;

use crate::tools::ToolOutput;

/// The most calls of one batch that run at the same time.
const MAX_CONCURRENT_CALLS: usize = 10;

/// One call of a batch, not yet started.
pub(super) struct BatchCall {
    /// Whether the call may run alongside other concurrency-safe calls.
    pub(super) concurrency_safe: bool,
    pub(super) execution: BoxFuture<'static, ToolOutput>,
}

/// Run a batch's calls and return their outputs in call order.
///
/// Consecutive concurrency-safe calls run together. Any other call starts
/// after every earlier call has finished and finishes before any later call
/// starts, so it sees the effects of the calls before it and the calls after
/// it see its own.
pub(super) async fn run_batch(calls: Vec<BatchCall>) -> Vec<ToolOutput> {
    let mut outputs = Vec::with_capacity(calls.len());
    let mut calls = calls.into_iter().peekable();
    while let Some(first) = calls.next() {
        let mut group = vec![first.execution];
        if first.concurrency_safe {
            while let Some(next) = calls.next_if(|call| call.concurrency_safe) {
                group.push(next.execution);
            }
        }
        outputs.extend(run_group(group).await);
    }
    outputs
}

/// Run calls together and return their outputs in their original order.
///
/// A call starts as soon as any running call finishes, so one slow call does
/// not hold back the rest of its group.
async fn run_group(executions: Vec<BoxFuture<'static, ToolOutput>>) -> Vec<ToolOutput> {
    let calls: Vec<_> = executions
        .into_iter()
        .enumerate()
        .map(|(position, execution)| run_call(position, execution))
        .collect();
    let mut finished: Vec<(usize, ToolOutput)> = stream::iter(calls)
        .buffer_unordered(MAX_CONCURRENT_CALLS)
        .collect()
        .await;
    finished.sort_by_key(|(position, _)| *position);
    finished.into_iter().map(|(_, output)| output).collect()
}

/// Run one call, tagged with its position in its group.
async fn run_call(
    position: usize,
    execution: BoxFuture<'static, ToolOutput>,
) -> (usize, ToolOutput) {
    (position, isolate(execution).await)
}

/// A panicking call yields its own error output instead of ending the batch.
async fn isolate(execution: BoxFuture<'static, ToolOutput>) -> ToolOutput {
    AssertUnwindSafe(execution)
        .catch_unwind()
        .await
        .unwrap_or_else(|payload| ToolOutput::error(panic_message(payload)))
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|message| (*message).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "tool panicked".to_string())
}
