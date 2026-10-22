use pretty_assertions::assert_eq;

use super::*;
use crate::tools::web_test_server::TestResponse;
use crate::tools::web_test_server::TestServer;
use crate::tools::AnyTool;
use crate::web::YouClient;

#[tokio::test]
async fn returns_normalized_safe_unicode_results() {
    let mut server = TestServer::start(vec![TestResponse::json(
        200,
        serde_json::json!({
            "results": {"web": [
                {
                    "title": " Python\n版本 ",
                    "url": "https://python.org/downloads/",
                    "description": "Version   details"
                },
                {
                    "title": "Unsafe",
                    "url": "javascript:alert(1)",
                    "description": "not a source"
                },
                {
                    "title": "Credentials",
                    "url": "https://user@example.com/private",
                    "description": "not a source"
                }
            ]}
        }),
    )])
    .await;
    let client = YouClient::for_test(Some("secret"), &server.base_url);
    let result = WebSearchTool::new(Box::new(move |_| Ok(Box::new(client.clone()))))
        .run(
            WebSearchArgs {
                query: "latest Python release".to_string(),
            },
            &ExecutionContext::default(),
        )
        .await
        .expect("search");

    assert!(result.success);
    assert_eq!(result.display.as_deref(), Some("you: 1 result(s)"));
    assert_eq!(
        result.content,
        r#"{
  "results": [
    {
      "title": " Python\n版本 ",
      "link": "https://python.org/downloads/",
      "snippet": "Version   details"
    }
  ]
}"#
    );
    let request = server.request().await;
    assert!(request.path.starts_with("/v1/search?"));
}

#[tokio::test]
async fn remote_workspace_search_uses_the_host_provider_not_the_shell_endpoint() {
    use crate::tools::{DirectToolExecutor, RemoteToolExecutor, ToolExecutor};
    use std::sync::Arc;

    let response = serde_json::json!({
        "results": {"web": [{"url": "https://example.com", "title": "Result", "snippets": ["Summary"]}]},
    });
    let mut server = TestServer::start(vec![
        TestResponse::json(200, response.clone()),
        TestResponse::json(200, response),
    ])
    .await;
    let client = YouClient::for_test(Some("synthetic-provider-key"), &server.base_url);
    let tool = Arc::new(WebSearchTool::new(Box::new(move |_| {
        Ok(Box::new(client.clone()))
    })));
    let local = DirectToolExecutor::new(vec![tool.clone()]);
    let remote = RemoteToolExecutor::new(
        vec![tool],
        reqwest::Url::parse("http://127.0.0.1:1/exec").expect("URL"),
    )
    .expect("host-safe tool accepted without contacting shell endpoint");
    let args = r#"{"query":"example"}"#.to_string();
    let local = local
        .call_tool(
            "web_search".into(),
            args.clone(),
            ExecutionContext::default(),
        )
        .await
        .expect("local search");
    let remote = remote
        .call_tool("web_search".into(), args, ExecutionContext::default())
        .await
        .expect("remote-compatible search");
    assert!(remote.success);
    // Separate invocations have different timings but must return the same output.
    assert_eq!(
        ToolOutput {
            duration: None,
            ..local
        },
        ToolOutput {
            duration: None,
            ..remote
        },
    );
    for _ in 0..2 {
        let request = server.request().await;
        assert!(request.path.starts_with("/v1/search?"));
        assert_eq!(request.api_key.as_deref(), Some("synthetic-provider-key"));
    }
}

#[test]
fn caps_result_fields_and_total_output() {
    let large = SearchResult {
        title: Some("t".repeat(MAX_TITLE_CHARS + 100)),
        link: format!("https://example.com/{}", "u".repeat(1_900)),
        snippet: Some("s".repeat(MAX_SNIPPET_CHARS + 100)),
    };

    let (content, count) = serialize_results(vec![large; 10]).expect("serialize");
    let payload: Value = serde_json::from_str(&content).expect("JSON");

    assert!(content.chars().count() <= MAX_OUTPUT_CHARS);
    assert!(count > 0 && count < 10);
    assert_eq!(payload["results"].as_array().expect("results").len(), count);
    assert!(payload["results"][0]["title"]
        .as_str()
        .expect("title")
        .ends_with('…'));
}

#[tokio::test]
async fn reports_provider_errors() {
    let client = YouClient::for_test(None, "http://127.0.0.1:1");
    let tool = WebSearchTool::new(Box::new(move |_| Ok(Box::new(client.clone()))));

    let result = tool
        .run(
            WebSearchArgs {
                query: "query".to_string(),
            },
            &ExecutionContext::default(),
        )
        .await
        .expect("tool result");

    assert!(!result.success);
    assert!(result
        .content
        .starts_with("Tool error: YOU_API_KEY is not set"));
    assert_eq!(
        AnyTool::definition(&tool).parameters["required"],
        serde_json::json!(["query"])
    );
}
