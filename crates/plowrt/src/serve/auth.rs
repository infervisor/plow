//! Optional API-key authentication (`--api-key` / `PLOW_API_KEYS`) for every route but the
//! health probes, on whichever listener the router serves.

use std::sync::Arc;

use axum::extract::Request;
use axum::http::{header, HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::Router;

use crate::config::ApiKey;

/// Routes a probe reaches without a key.
const OPEN: [&str; 2] = ["/health", "/healthz"];

/// `router` behind the keys; unchanged when there are none.
pub fn require(router: Router, keys: &[ApiKey]) -> Router {
    if keys.is_empty() {
        return router;
    }
    let keys: Arc<[Vec<u8>]> = keys.iter().map(|k| k.0.as_bytes().to_vec()).collect();
    router.layer(axum::middleware::from_fn(move |request: Request, next: Next| {
        let keys = Arc::clone(&keys);
        async move {
            let presented = presented(&request);
            if OPEN.contains(&request.uri().path())
                || presented.is_some_and(|p| keys.iter().fold(false, |hit, key| hit | same(key, p)))
            {
                return next.run(request).await;
            }
            let mut response = crate::serve::api_error(
                StatusCode::UNAUTHORIZED,
                "missing or invalid API key",
                "invalid_request_error",
                Some("invalid_api_key"),
                None,
            );
            response.headers_mut().insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
            response
        }
    }))
}

/// `Authorization: Bearer <key>`, else `x-api-key: <key>`.
fn presented(request: &Request) -> Option<&[u8]> {
    let headers = request.headers();
    if let Some(value) = headers.get(header::AUTHORIZATION) {
        let value = value.as_bytes();
        return (value.len() > 7 && value[..7].eq_ignore_ascii_case(b"bearer ")).then(|| value[7..].trim_ascii());
    }
    headers.get("x-api-key").map(|v| v.as_bytes().trim_ascii())
}

/// Equality whose time does not depend on where the bytes differ.
fn same(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Every non-loopback TCP bind without keys is reachable unauthenticated: say so at startup.
pub fn warn_if_open(bind: std::net::IpAddr, keys: &[ApiKey]) {
    if keys.is_empty() && !bind.is_loopback() {
        tracing::warn!(%bind, "serving without API keys on a non-loopback address; set --api-key or PLOW_API_KEYS");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::routing::get;
    use tower::ServiceExt;

    fn app() -> Router {
        let keys = ["k1".parse().unwrap(), "secret-2".parse().unwrap()];
        require(Router::new().route("/v1/models", get(|| async { "ok" })).route("/health", get(|| async { "ok" })), &keys)
    }

    async fn status(header: Option<(&str, &str)>, path: &str) -> StatusCode {
        let mut request = Request::get(path);
        if let Some((name, value)) = header {
            request = request.header(name, value);
        }
        app().oneshot(request.body(Body::empty()).unwrap()).await.unwrap().status()
    }

    #[tokio::test]
    async fn keys_gate_every_route_but_health() {
        assert_eq!(status(None, "/v1/models").await, StatusCode::UNAUTHORIZED);
        assert_eq!(status(Some(("authorization", "Bearer nope")), "/v1/models").await, StatusCode::UNAUTHORIZED);
        assert_eq!(status(Some(("authorization", "Basic k1")), "/v1/models").await, StatusCode::UNAUTHORIZED);
        assert_eq!(status(Some(("authorization", "Bearer k1")), "/v1/models").await, StatusCode::OK);
        assert_eq!(status(Some(("authorization", "bearer secret-2")), "/v1/models").await, StatusCode::OK);
        assert_eq!(status(Some(("x-api-key", "k1")), "/v1/models").await, StatusCode::OK);
        assert_eq!(status(None, "/health").await, StatusCode::OK);
        let open = require(Router::new().route("/v1/models", get(|| async { "ok" })), &[]);
        let response = open.oneshot(Request::get("/v1/models").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert!("".parse::<ApiKey>().is_err());
        assert_eq!(format!("{:?}", "k1".parse::<ApiKey>().unwrap()), "ApiKey(<redacted>)");
    }
}
