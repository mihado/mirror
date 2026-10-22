use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use assert_matches::assert_matches;
use futures_util::future::BoxFuture;
use pretty_assertions::assert_eq;
use serde_json::json;

use super::*;
use crate::tools::web_test_server::{TestResponse, TestServer};

#[tokio::test]
async fn contents_requests_full_markdown_and_preserves_page_fields() {
    let mut server = TestServer::start(vec![
        TestResponse::json(
            200,
            json!([{
                "url": "https://example.com/final", "title": " 版本 ",
                "markdown": " first line\n\n完整 text "
            }]),
        ),
        TestResponse::json(200, json!([{ "url": "", "markdown": "fallback" }])),
    ])
    .await;
    let client = YouClient::for_test(Some("synthetic-you-key"), &server.base_url);
    assert_eq!(
        client.contents("https://example.com/page").await,
        Ok(Page {
            url: "https://example.com/final".into(),
            title: Some(" 版本 ".into()),
            text: " first line\n\n完整 text ".into(),
        })
    );
    let request = server.request().await;
    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/v1/contents");
    assert_eq!(request.api_key.as_deref(), Some("synthetic-you-key"));
    assert_eq!(request.authorization, None);
    assert_eq!(
        request.body,
        json!({
            "urls": ["https://example.com/page"], "formats": ["markdown"], "crawl_timeout": 60,
        })
    );
    assert_eq!(
        client.contents("https://example.com/page").await,
        Ok(Page {
            url: "https://example.com/page".into(),
            title: None,
            text: "fallback".into(),
        })
    );
}

#[tokio::test]
async fn contents_rejects_unavailable_and_malformed_pages() {
    let server = TestServer::start(vec![
        TestResponse::json(200, json!([])),
        TestResponse::json(200, json!([{ "markdown": null }])),
        TestResponse::json(200, json!([{ "markdown": " \n\u{1c}\u{a0}" }])),
        TestResponse::json(200, json!([{}, {}])),
        TestResponse::json(200, json!([{ "markdown": 42 }])),
        TestResponse::text(200, "synthetic-you-key"),
    ])
    .await;
    let client = YouClient::for_test(Some("synthetic-you-key"), &server.base_url);
    let unavailable = Err(YouError::Response(
        "This website could not be accessed. Try a different source.".into(),
    ));
    let malformed = Err(YouError::Response(
        "Malformed You.com Contents response".into(),
    ));
    assert_eq!(client.contents("https://example.com").await, unavailable);
    assert_eq!(client.contents("https://example.com").await, unavailable);
    assert_eq!(client.contents("https://example.com").await, unavailable);
    assert_eq!(client.contents("https://example.com").await, malformed);
    assert_eq!(client.contents("https://example.com").await, malformed);
    assert_eq!(client.contents("https://example.com").await, malformed);
}

#[tokio::test]
async fn contents_reports_missing_keys_and_nonretryable_errors_without_bodies() {
    let missing = YouClient::for_test(None, "http://127.0.0.1:1");
    assert_eq!(
        missing.contents("https://example.com").await,
        Err(YouError::MissingApiKey)
    );
    let server = TestServer::start(vec![
        TestResponse::text(403, "synthetic-you-key"),
        TestResponse::text(400, "synthetic-you-key"),
        TestResponse::text(422, "synthetic-you-key"),
    ])
    .await;
    let client = YouClient::for_test(Some("synthetic-you-key"), &server.base_url);
    assert_eq!(
        client.contents("https://example.com").await,
        Err(YouError::Response("You.com rejected YOU_API_KEY.".into()))
    );
    let invalid = Err(YouError::Response(
        "Invalid URL for You.com Contents. Try a different URL.".into(),
    ));
    assert_eq!(client.contents("https://example.com").await, invalid);
    assert_eq!(client.contents("https://example.com").await, invalid);
}

#[tokio::test]
async fn contents_recovers_with_one_budget_for_transport_and_http_failures() {
    let mut server = TestServer::start(vec![
        TestResponse::disconnected(),
        TestResponse::text(503, "synthetic-you-key"),
        TestResponse::json(200, json!([{ "markdown": "recovered" }])),
    ])
    .await;
    let client = YouClient::for_test(Some("synthetic-you-key"), &server.base_url);
    let started = tokio::time::Instant::now();
    assert_eq!(
        client.contents("https://example.com").await,
        Ok(Page {
            url: "https://example.com".into(),
            title: None,
            text: "recovered".into(),
        })
    );
    assert!(started.elapsed() >= Duration::from_secs(4));
    assert_eq!(server.request().await.path, "/v1/contents");
    assert_eq!(server.request().await.path, "/v1/contents");
    assert_eq!(server.request().await.path, "/v1/contents");
}

#[tokio::test]
async fn contents_stops_after_three_mixed_failures() {
    let mut server = TestServer::start(vec![
        TestResponse::text(429, "synthetic-you-key"),
        TestResponse::truncated(),
        TestResponse::text(504, "synthetic-you-key"),
        TestResponse::json(200, json!([{ "markdown": "too late" }])),
    ])
    .await;
    let client = YouClient::for_test(Some("synthetic-you-key"), &server.base_url);
    assert_eq!(
        client.contents("https://example.com").await,
        Err(YouError::Response(
            "You.com Contents API returned HTTP 504".into()
        ))
    );
    let _ = server.request().await;
    let _ = server.request().await;
    let _ = server.request().await;
    assert!(
        tokio::time::timeout(Duration::from_millis(100), server.request())
            .await
            .is_err()
    );
}

#[tokio::test]
async fn contents_revoked_authority_stops_retry_dispatch() {
    struct FirstRequestOnly(AtomicBool);
    impl YouRequestGuard for FirstRequestOnly {
        fn verify(&self) -> BoxFuture<'_, Result<(), YouError>> {
            Box::pin(async move {
                if self.0.swap(false, Ordering::SeqCst) {
                    Ok(())
                } else {
                    Err(YouError::Response("lease revoked".into()))
                }
            })
        }
    }
    let mut server = TestServer::start(vec![
        TestResponse::text(503, "unavailable"),
        TestResponse::json(200, json!([{ "markdown": "too late" }])),
    ])
    .await;
    let client = YouClient::for_test(Some("synthetic-you-key"), &server.base_url)
        .with_request_guard(Arc::new(FirstRequestOnly(AtomicBool::new(true))));
    assert_eq!(
        client.contents("https://example.com").await,
        Err(YouError::Response("lease revoked".into()))
    );
    let _ = server.request().await;
    assert!(
        tokio::time::timeout(Duration::from_millis(100), server.request())
            .await
            .is_err()
    );
}

#[tokio::test]
async fn search_uses_you_parameters_and_first_real_snippets() {
    let mut server = TestServer::start(vec![TestResponse::json(200, json!({
        "results": {"web": [
            {"url": "https://example.com/first", "title": " First ",
             "description": "fallback", "snippets": ["", "\u{1c}\u{a0}", " NULL ", "  版本\npassage  ", "ignored"]},
            {"url": "https://example.com/fallback", "title": "", "description": " fallback ", "snippets": ["null"]},
            {"url": "https://example.com/empty", "title": "Empty", "description": "", "snippets": null},
            {"url": "https://example.com/missing", "title": "Missing"}
        ]}, "ignored": true
    }))]).await;
    let client = YouClient::for_test(Some("synthetic-you-key"), &server.base_url);
    assert_eq!(
        client.search("Python & 版本?", 10).await.expect("search"),
        vec![
            SearchResult {
                title: Some(" First ".into()),
                link: "https://example.com/first".into(),
                snippet: Some("  版本\npassage  ".into())
            },
            SearchResult {
                title: Some("".into()),
                link: "https://example.com/fallback".into(),
                snippet: Some(" fallback ".into())
            },
            SearchResult {
                title: Some("Empty".into()),
                link: "https://example.com/empty".into(),
                snippet: None
            },
            SearchResult {
                title: Some("Missing".into()),
                link: "https://example.com/missing".into(),
                snippet: None
            },
        ]
    );
    let request = server.request().await;
    assert_eq!(request.method, "GET");
    let url =
        reqwest::Url::parse(&format!("{}{}", server.base_url, request.path)).expect("request URL");
    assert_eq!(url.path(), "/v1/search");
    assert_eq!(
        url.query_pairs().collect::<BTreeMap<_, _>>(),
        BTreeMap::from([
            ("query".into(), "Python & 版本?".into()),
            ("count".into(), "10".into()),
            ("offset".into(), "0".into()),
            ("snippet_engine".into(), "rerank_v5".into()),
            ("snippet_engine_target_chars".into(), "200".into()),
        ])
    );
    assert_eq!(request.api_key.as_deref(), Some("synthetic-you-key"));
    assert_eq!(request.authorization, None);
    assert_eq!(request.body, serde_json::Value::Null);
    assert!(!format!("{client:?}").contains("synthetic-you-key"));
}

#[tokio::test]
async fn search_limits_results_and_accepts_empty_responses() {
    let mut server = TestServer::start(vec![
        TestResponse::json(
            200,
            json!({"results": {"web": [
                {"url": "https://example.com/1", "title": "one"},
                {"url": "https://example.com/2", "title": "two"}
            ]}}),
        ),
        TestResponse::json(200, json!({"results": null})),
        TestResponse::json(200, json!({"results": {}})),
    ])
    .await;
    let client = YouClient::for_test(Some("synthetic-you-key"), &server.base_url);
    assert_eq!(
        client
            .search("query", 1)
            .await
            .expect("limited search")
            .len(),
        1
    );
    assert_eq!(client.search("query", 10).await.expect("null results"), []);
    assert_eq!(client.search("query", 10).await.expect("missing web"), []);
    let _ = server.request().await;
}

#[tokio::test]
async fn search_retries_transient_statuses_with_anti_serp_backoff() {
    let mut server = TestServer::start(vec![
        TestResponse::text(503, "blocked"),
        TestResponse::text(504, "timeout"),
        TestResponse::text(429, "limited"),
        TestResponse::json(200, json!({"results": {"web": []}})),
    ])
    .await;
    let client = YouClient::for_test(Some("synthetic-you-key"), &server.base_url);
    let started = tokio::time::Instant::now();
    assert_eq!(client.search("query", 10).await.expect("recovered"), []);
    assert!(started.elapsed() >= Duration::from_secs(15 + 30 + 8));
    let _ = server.request().await;
}

#[tokio::test]
async fn search_reports_missing_keys_and_sanitized_provider_errors() {
    let missing = YouClient::for_test(Some(" \t "), "http://127.0.0.1:1");
    assert!(!missing.is_configured());
    assert_eq!(
        missing.search("query", 10).await,
        Err(YouError::MissingApiKey)
    );
    let mut server = TestServer::start(vec![
        TestResponse::text(401, "synthetic-you-key"),
        TestResponse::text(400, "synthetic-you-key"),
        TestResponse::text(200, "synthetic-you-key"),
        TestResponse::json(
            200,
            json!({"results": {"web": [{"url": "https://example.com"}]}}),
        ),
    ])
    .await;
    let client = YouClient::for_test(Some("synthetic-you-key"), &server.base_url);
    assert!(client.is_configured());
    assert_eq!(
        client.search("query", 10).await,
        Err(YouError::Response("You.com rejected YOU_API_KEY.".into()))
    );
    assert_eq!(
        client.search("query", 10).await,
        Err(YouError::Response(
            "You.com search returned HTTP 400.".into()
        ))
    );
    assert_eq!(
        client.search("query", 10).await,
        Err(YouError::Response(
            "You.com search returned an invalid response.".into()
        ))
    );
    assert_matches!(client.search("query", 10).await, Err(YouError::Response(message)) if message == "You.com search returned an invalid response.");
    let _ = server.request().await;
}

#[tokio::test]
async fn requests_do_not_forward_credentials_on_redirects() {
    let mut destination = TestServer::start(vec![TestResponse::json(200, json!({}))]).await;
    let location = destination.base_url.clone();
    let redirect = move || {
        let location = location.clone();
        async move {
            (
                axum::http::StatusCode::FOUND,
                [(axum::http::header::LOCATION, location)],
            )
        }
    };
    let app = axum::Router::new()
        .route("/v1/search", axum::routing::get(redirect.clone()))
        .route("/v1/contents", axum::routing::post(redirect));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("redirect listener");
    let base_url = format!("http://{}", listener.local_addr().expect("address"));
    let task =
        tokio::spawn(async move { axum::serve(listener, app).await.expect("redirect server") });
    let client = YouClient::for_test(Some("synthetic-you-key"), &base_url);
    assert_eq!(
        client.search("query", 10).await,
        Err(YouError::Response(
            "You.com search returned HTTP 302.".into()
        ))
    );
    assert_eq!(
        client.contents("https://example.com").await,
        Err(YouError::Response(
            "You.com Contents API returned HTTP 302".into()
        ))
    );
    assert!(
        tokio::time::timeout(Duration::from_millis(100), destination.request())
            .await
            .is_err()
    );
    task.abort();
}

#[tokio::test]
async fn search_revoked_authority_stops_retry_dispatch() {
    struct FirstRequestOnly(AtomicBool);
    impl YouRequestGuard for FirstRequestOnly {
        fn verify(&self) -> BoxFuture<'_, Result<(), YouError>> {
            Box::pin(async move {
                if self.0.swap(false, Ordering::SeqCst) {
                    Ok(())
                } else {
                    Err(YouError::Response("lease revoked".into()))
                }
            })
        }
    }
    let mut server = TestServer::start(vec![
        TestResponse::text(503, "unavailable"),
        TestResponse::json(200, json!({"results": {"web": []}})),
    ])
    .await;
    let client = YouClient::for_test(Some("synthetic-you-key"), &server.base_url)
        .with_request_guard(Arc::new(FirstRequestOnly(AtomicBool::new(true))));
    assert_eq!(
        client.search("query", 10).await,
        Err(YouError::Response("lease revoked".into()))
    );
    let _ = server.request().await;
    assert!(
        tokio::time::timeout(Duration::from_millis(100), server.request())
            .await
            .is_err()
    );
}
