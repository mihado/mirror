use pretty_assertions::assert_eq;

use super::*;

async fn trace(server: &Server, path: &str) -> Vec<Value> {
    let response = server
        .client
        .get(format!("{}{path}", server.url))
        .send()
        .await
        .expect("read trace without authentication");
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "application/x-ndjson");
    assert_eq!(response.headers()["cache-control"], "no-store");
    let jsonl = response.text().await.expect("JSONL");
    assert!(jsonl.is_empty() || jsonl.ends_with('\n'));
    jsonl
        .lines()
        .map(|line| serde_json::from_str(line).expect("record"))
        .collect()
}

#[tokio::test]
async fn memory_traces_preserve_native_and_model_records_without_files() {
    let mut fixture = Fixture::start().await;
    fixture.in_memory_output = true;
    let server = fixture
        .serve("", &["--system-prompt", "Actual system prompt."])
        .await;
    assert!(trace(&server, "/trace/model-calls").await.is_empty());
    assert_eq!(trace(&server, "/trace").await.len(), 1);
    assert_eq!(server.prompt().await.status(), StatusCode::NO_CONTENT);
    let first = fixture.request().await;
    let second = fixture.request().await;
    let native = trace(&server, "/trace").await;
    let calls = trace(&server, "/trace/model-calls").await;
    assert_eq!(native.len(), 7);
    assert_eq!(
        native[1]["chat"]["user"]["content"],
        "Write result.txt and finish."
    );
    assert_eq!(native[6]["chat"]["assistant"]["content"], "Task complete");
    assert_eq!(trace(&server, "/trace").await, native);
    assert_eq!(calls.len(), 5);
    assert_eq!(calls[0]["content"], first["messages"][0]["content"]);
    assert_eq!(calls[0]["tools"][0], first["tools"][0]["function"]);
    assert_eq!(calls[1]["content"], first["messages"][1]["content"]);
    assert_eq!(calls[2]["tool_calls"][0]["function"]["name"], "bash");
    assert_eq!(calls[3]["content"], second["messages"][3]["content"]);
    assert_eq!(calls[4]["content"], "Task complete");
    assert_eq!(calls[4]["usage"]["completion_tokens"], 10);
    assert!(calls[4]["timestamp_ms"].is_u64());
    assert!(!serde_json::to_string(&calls)
        .expect("JSON")
        .contains("secret-model-key"));
    assert!(!fixture.workspace.path().join("mirror.jsonl").exists());
    assert!(!fixture
        .workspace
        .path()
        .join("mirror.model-calls.jsonl")
        .exists());
    assert!(!fixture.workspace.path().join("logs").exists());
    assert_eq!(server.prompt().await.status(), StatusCode::NO_CONTENT);
    assert!(trace(&server, "/trace").await.starts_with(&native));
    assert_eq!(trace(&server, "/trace/model-calls").await.len(), 7);
    server.stop().await;
}

#[tokio::test]
async fn memory_traces_are_readable_during_an_active_turn_and_reject_browser_origins() {
    let mut fixture = Fixture::start().await;
    fixture.in_memory_output = true;
    let server = fixture.serve("/pending", &[]).await;
    let client = server.client.clone();
    let url = format!("{}/message", server.url);
    let request = tokio::spawn(async move {
        client
            .post(url)
            .header("content-type", "text/plain")
            .body("pending prompt")
            .send()
            .await
    });
    fixture.request().await;
    assert_eq!(
        trace(&server, "/trace").await[1]["chat"]["user"]["content"],
        "pending prompt"
    );
    assert_eq!(
        trace(&server, "/trace/model-calls").await[0]["content"],
        "pending prompt"
    );
    let browser = server
        .client
        .get(format!("{}/trace", server.url))
        .header("origin", "https://example.invalid")
        .send()
        .await
        .expect("browser request");
    assert_eq!(browser.status(), StatusCode::FORBIDDEN);
    request.abort();
    server.stop().await;
    assert!(!fixture.workspace.path().join("mirror.jsonl").exists());
}

#[tokio::test]
async fn memory_traces_remain_available_after_a_turn_timeout() {
    let mut fixture = Fixture::start().await;
    fixture.in_memory_output = true;
    let server = fixture
        .serve("/pending-after-tool", &["--turn-timeout-seconds", "0.5"])
        .await;
    assert_eq!(server.prompt().await.status(), StatusCode::GATEWAY_TIMEOUT);
    assert!(trace(&server, "/trace").await.len() >= 4);
    assert_eq!(
        trace(&server, "/trace/model-calls")
            .await
            .last()
            .expect("last record")["role"],
        "tool"
    );
    server.stop().await;
    assert!(!fixture
        .workspace
        .path()
        .join("mirror.model-calls.jsonl")
        .exists());
}

#[tokio::test]
async fn memory_and_file_outputs_contain_identical_records() {
    let mut fixture = Fixture::start().await;
    fixture.in_memory_output = true;
    let server = fixture.serve("", &["--output", "logs/trace.jsonl"]).await;
    assert_eq!(server.prompt().await.status(), StatusCode::NO_CONTENT);
    let native = trace(&server, "/trace").await;
    let calls = trace(&server, "/trace/model-calls").await;
    assert_eq!(native, fixture.trace());
    assert_eq!(calls, fixture.model_calls());
    assert_eq!(
        native.last().expect("final response")["chat"]["assistant"]["content"],
        "Task complete"
    );
    server.stop().await;
    assert_eq!(native, fixture.trace());
    assert_eq!(calls, fixture.model_calls());
}

#[tokio::test]
async fn memory_traces_work_with_personal_memory() {
    let mut fixture = Fixture::start().await;
    fixture.in_memory_output = true;
    let home = fixture.workspace.path().join("home/.mirror");
    std::fs::create_dir_all(&home).expect("memory home");
    std::fs::write(
        home.join("PINNED_MEMORIES.md"),
        "- Use concise responses.\n",
    )
    .expect("pinned memory");
    let server = fixture
        .serve("", &["--memory-dir", "personal-memory"])
        .await;

    assert_eq!(server.prompt().await.status(), StatusCode::NO_CONTENT);
    let request = fixture.request().await;
    assert!(request["messages"][0]["content"]
        .as_str()
        .expect("system prompt")
        .contains("Use concise responses."));
    assert_eq!(
        trace(&server, "/trace")
            .await
            .last()
            .expect("final response")["chat"]["assistant"]["content"],
        "Task complete"
    );
    assert!(!fixture.workspace.path().join("mirror.jsonl").exists());
    server.stop().await;
}

#[tokio::test]
async fn memory_model_calls_continue_after_file_write_failure() {
    let mut fixture = Fixture::start().await;
    fixture.in_memory_output = true;
    let logs = fixture.workspace.path().join("logs");
    std::fs::create_dir_all(&logs).expect("logs");
    std::os::unix::fs::symlink("/dev/full", logs.join("trace.model-calls.jsonl"))
        .expect("failing trace file");
    let server = fixture.serve("", &["--output", "logs/trace.jsonl"]).await;

    assert_eq!(server.prompt().await.status(), StatusCode::NO_CONTENT);
    let first = trace(&server, "/trace/model-calls").await;
    assert_eq!(
        first.last().expect("final response")["content"],
        "Task complete"
    );
    assert_eq!(server.prompt().await.status(), StatusCode::NO_CONTENT);
    let second = trace(&server, "/trace/model-calls").await;
    assert!(second.starts_with(&first));
    assert!(second.len() > first.len());
    assert_eq!(
        second.last().expect("final response")["content"],
        "Task complete"
    );
    server.stop().await;
}

#[tokio::test]
async fn file_mode_does_not_expose_trace_endpoints() {
    let fixture = Fixture::start().await;
    let server = fixture.serve("", &[]).await;
    assert_eq!(
        server
            .client
            .get(format!("{}/trace", server.url))
            .send()
            .await
            .expect("native route absent")
            .status(),
        StatusCode::NOT_FOUND
    );
    assert_eq!(
        server
            .client
            .get(format!("{}/trace/model-calls", server.url))
            .send()
            .await
            .expect("diagnostic route absent")
            .status(),
        StatusCode::NOT_FOUND
    );
    server.stop().await;
}
