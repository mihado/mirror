use std::collections::HashMap;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::Router;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tower::limit::ConcurrencyLimitLayer;

use crate::loopback_listener::loopback_listener;
use mirror_sandbox::StaticEndpoint;

pub(crate) struct StaticEndpointPlan {
    address: SocketAddr,
    table: Arc<RouteTable>,
}

struct RouteTable {
    headers: HeaderMap,
    routes: HashMap<String, Route>,
}

struct Route {
    required: HeaderMap,
    status: StatusCode,
    headers: HeaderMap,
    body: Bytes,
}

pub(crate) struct StaticEndpointServer {
    task: JoinHandle<()>,
}

const MAX_CONCURRENT_REQUESTS: usize = 32;

impl StaticEndpointPlan {
    pub(crate) fn new(endpoint: StaticEndpoint) -> Result<Self, String> {
        let address: SocketAddr = endpoint
            .listen
            .parse()
            .map_err(|_| format!("listen address {:?} is not an IP and port", endpoint.listen))?;
        if address.ip() != Ipv4Addr::LOCALHOST || address.port() == 0 {
            return Err(format!(
                "listen address {address} must be 127.0.0.1 with a fixed port"
            ));
        }
        let routes = endpoint
            .routes
            .into_iter()
            .map(|route| {
                let status = u16::try_from(route.status)
                    .ok()
                    .and_then(|status| StatusCode::from_u16(status).ok())
                    .filter(|status| !status.is_informational())
                    .ok_or_else(|| {
                        format!("route {:?} has invalid status {}", route.path, route.status)
                    })?;
                let parsed = Route {
                    required: headers(&route.required_headers)?,
                    status,
                    headers: headers(&route.response_headers)?,
                    body: Bytes::from(route.body),
                };
                Ok((route.path, parsed))
            })
            .collect::<Result<_, String>>()?;
        Ok(Self {
            address,
            table: Arc::new(RouteTable {
                headers: headers(&endpoint.response_headers)?,
                routes,
            }),
        })
    }

    pub(crate) async fn start(
        &self,
        mut closing: watch::Receiver<bool>,
    ) -> Result<StaticEndpointServer, String> {
        let listener = loopback_listener(self.address)
            .await
            .map_err(|error| format!("endpoint {} could not listen: {error}", self.address))?;
        let router = Router::new()
            .fallback(respond)
            .with_state(self.table.clone())
            .layer(ConcurrencyLimitLayer::new(MAX_CONCURRENT_REQUESTS));
        let task = tokio::spawn(async move {
            tokio::select! {
                _ = axum::serve(listener, router) => {}
                _ = closing.wait_for(|closed| *closed) => {}
            }
        });
        Ok(StaticEndpointServer { task })
    }
}

impl Drop for StaticEndpointServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn respond(
    State(table): State<Arc<RouteTable>>,
    method: Method,
    request_headers: HeaderMap,
    uri: Uri,
) -> Response {
    let route = table.routes.get(uri.path());
    let (mut response, served) = match route {
        _ if method != Method::GET => (StatusCode::METHOD_NOT_ALLOWED.into_response(), None),
        None => (StatusCode::NOT_FOUND.into_response(), None),
        Some(route) if !admits(&route.required, &request_headers) => {
            (StatusCode::FORBIDDEN.into_response(), None)
        }
        Some(route) => (
            (route.status, Body::from(route.body.clone())).into_response(),
            Some(route),
        ),
    };
    let headers = response.headers_mut();
    for (name, value) in table
        .headers
        .iter()
        .chain(served.into_iter().flat_map(|route| route.headers.iter()))
    {
        headers.insert(name.clone(), value.clone());
    }
    response
}

fn admits(required: &HeaderMap, request: &HeaderMap) -> bool {
    required
        .iter()
        .all(|(name, value)| request.get_all(name).iter().any(|sent| sent == value))
}

fn headers(values: &HashMap<String, String>) -> Result<HeaderMap, String> {
    values
        .iter()
        .map(|(name, value)| {
            Ok((
                HeaderName::try_from(name.as_str())
                    .map_err(|_| format!("header name {name:?} is invalid"))?,
                HeaderValue::try_from(value.as_str())
                    .map_err(|_| format!("header {name:?} has an invalid value"))?,
            ))
        })
        .collect()
}

#[cfg(test)]
#[path = "static_endpoint_tests.rs"]
mod tests;
