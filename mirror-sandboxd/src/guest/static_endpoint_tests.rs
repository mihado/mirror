use pretty_assertions::assert_eq;

use super::*;
use mirror_sandbox::StaticRoute;

fn plan(listen: &str) -> Result<StaticEndpointPlan, String> {
    StaticEndpointPlan::new(StaticEndpoint {
        listen: listen.into(),
        routes: Vec::new(),
        response_headers: HashMap::new(),
    })
}

async fn get(port: u16, path: &str, headers: &[(&str, &str)]) -> (u16, String, String, String) {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let request = headers.iter().fold(
        reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("HTTP client")
            .get(format!("http://127.0.0.1:{port}{path}")),
        |request, (name, value)| request.header(*name, *value),
    );
    let response = request.send().await.expect("response");
    let status = response.status().as_u16();
    let header = |name: &str| {
        response
            .headers()
            .get(name)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_owned()
    };
    let (endpoint, content_type) = (header("x-endpoint"), header("content-type"));
    (
        status,
        endpoint,
        content_type,
        response.text().await.expect("body"),
    )
}

#[test]
fn loopback_addresses_other_than_127_0_0_1_are_refused() {
    assert_eq!(
        plan("[::1]:8080").err(),
        Some("listen address [::1]:8080 must be 127.0.0.1 with a fixed port".into())
    );
    assert_eq!(
        plan("127.0.0.2:8080").err(),
        Some("listen address 127.0.0.2:8080 must be 127.0.0.1 with a fixed port".into())
    );
    assert!(plan("127.0.0.1:8080").is_ok());
}

#[tokio::test]
async fn routes_match_the_path_and_require_their_headers() {
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .expect("free port")
        .port();
    let endpoint = StaticEndpointPlan::new(StaticEndpoint {
        listen: format!("127.0.0.1:{port}"),
        routes: vec![StaticRoute {
            path: "/token".into(),
            required_headers: HashMap::from([("X-Client".into(), "tool".into())]),
            status: 200,
            response_headers: HashMap::from([("Content-Type".into(), "text/plain".into())]),
            body: b"placeholder".to_vec(),
        }],
        response_headers: HashMap::from([("X-Endpoint".into(), "static".into())]),
    })
    .expect("valid endpoint");
    let (_closing, closed) = watch::channel(false);
    let _server = endpoint.start(closed).await.expect("endpoint listens");
    let client = [("X-Client", "tool")];
    assert_eq!(
        vec![
            get(port, "/token?scopes=a,b&recursive=true", &client).await,
            get(port, "/token", &[]).await,
            get(port, "/token", &[("X-Client", "other")]).await,
            get(port, "/token/extra", &client).await,
        ],
        vec![
            (
                200,
                "static".into(),
                "text/plain".into(),
                "placeholder".into()
            ),
            (403, "static".into(), String::new(), String::new()),
            (403, "static".into(), String::new(), String::new()),
            (404, "static".into(), String::new(), String::new()),
        ]
    );
}
