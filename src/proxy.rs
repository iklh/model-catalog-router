use crate::config::Config;
use anyhow::{bail, Context, Result};
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::header::{AUTHORIZATION, CONNECTION, CONTENT_LENGTH, HOST, TRANSFER_ENCODING};
use axum::http::{HeaderMap, HeaderName, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get};
use axum::{Json, Router};
use futures_util::TryStreamExt;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::watch;
use tower_http::trace::TraceLayer;
use tracing::info;

#[derive(Clone)]
struct AppState {
    config: Arc<Config>,
    client: reqwest::Client,
    mode: ServiceMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ServiceMode {
    Base,
    OpenAiCompact,
}

impl ServiceMode {
    fn name(self) -> &'static str {
        match self {
            Self::Base => "base",
            Self::OpenAiCompact => "openai-compact",
        }
    }
}

pub async fn serve(config: Config) -> Result<()> {
    let listen = config.server.listen;
    let listener = tokio::net::TcpListener::bind(listen)
        .await
        .with_context(|| format!("failed to listen on {listen}"))?;
    info!(%listen, mode = ServiceMode::Base.name(), "model catalog router started");
    axum::serve(listener, build_router(config, ServiceMode::Base)?)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("router server failed")
}

pub async fn serve_openai_compact(config: Config) -> Result<()> {
    ensure_openai_compact_models(&config)?;
    let listen = config.server.openai_compact_listen;
    let listener = tokio::net::TcpListener::bind(listen)
        .await
        .with_context(|| format!("failed to listen on {listen}"))?;
    info!(
        %listen,
        mode = ServiceMode::OpenAiCompact.name(),
        "model catalog router started"
    );
    axum::serve(listener, build_router(config, ServiceMode::OpenAiCompact)?)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("OpenAI compact router server failed")
}

pub async fn serve_all(config: Config) -> Result<()> {
    ensure_openai_compact_models(&config)?;
    let base_listen = config.server.listen;
    let compact_listen = config.server.openai_compact_listen;
    let base_listener = tokio::net::TcpListener::bind(base_listen)
        .await
        .with_context(|| format!("failed to listen on {base_listen}"))?;
    let compact_listener = tokio::net::TcpListener::bind(compact_listen)
        .await
        .with_context(|| format!("failed to listen on {compact_listen}"))?;
    let base_app = build_router(config.clone(), ServiceMode::Base)?;
    let compact_app = build_router(config, ServiceMode::OpenAiCompact)?;

    info!(
        listen = %base_listen,
        mode = ServiceMode::Base.name(),
        "model catalog router started"
    );
    info!(
        listen = %compact_listen,
        mode = ServiceMode::OpenAiCompact.name(),
        "model catalog router started"
    );

    let (shutdown_sender, shutdown_receiver) = watch::channel(false);
    tokio::spawn(async move {
        shutdown_signal().await;
        let _ = shutdown_sender.send(true);
    });
    let base_shutdown = wait_for_shutdown(shutdown_receiver.clone());
    let compact_shutdown = wait_for_shutdown(shutdown_receiver);
    let base_server = async {
        axum::serve(base_listener, base_app)
            .with_graceful_shutdown(base_shutdown)
            .await
            .context("base router server failed")
    };
    let compact_server = async {
        axum::serve(compact_listener, compact_app)
            .with_graceful_shutdown(compact_shutdown)
            .await
            .context("OpenAI compact router server failed")
    };
    tokio::try_join!(base_server, compact_server)?;
    Ok(())
}

fn build_router(config: Config, mode: ServiceMode) -> Result<Router> {
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .build()
        .context("failed to build HTTP client")?;
    let state = AppState {
        config: Arc::new(config),
        client,
        mode,
    };
    Ok(Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/*path", any(forward))
        .layer(TraceLayer::new_for_http())
        .with_state(state))
}

fn ensure_openai_compact_models(config: &Config) -> Result<()> {
    if crate::catalog::configured_openai_compact_models(config).is_empty() {
        bail!(
            "no remote compaction models are configured; run `model-catalog-router config` and select models with matching `{}` aliases",
            crate::catalog::OPENAI_COMPACT_SUFFIX
        );
    }
    Ok(())
}

async fn wait_for_shutdown(mut receiver: watch::Receiver<bool>) {
    while !*receiver.borrow() {
        if receiver.changed().await.is_err() {
            break;
        }
    }
}

async fn health(State(state): State<AppState>) -> Json<Value> {
    Json(json!({ "status": "ok", "mode": state.mode.name() }))
}

async fn list_models(State(state): State<AppState>) -> Result<Json<Value>, ProxyError> {
    let models = match state.mode {
        ServiceMode::Base => crate::catalog::configured_models(&state.config),
        ServiceMode::OpenAiCompact => {
            crate::catalog::configured_openai_compact_models(&state.config)
        }
    };
    let data = models
        .into_iter()
        .map(
            |model| json!({ "id": model.routed_id, "object": "model", "owned_by": model.provider }),
        )
        .collect::<Vec<_>>();
    Ok(Json(json!({ "object": "list", "data": data })))
}

async fn forward(
    State(state): State<AppState>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ProxyError> {
    let mut payload: Value = serde_json::from_slice(&body)
        .map_err(|error| ProxyError::bad_request(format!("request body must be JSON: {error}")))?;
    let routed_model = payload
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| ProxyError::bad_request("request body must contain a string `model`"))?
        .to_owned();
    let (provider_name, upstream_model) =
        split_model(&routed_model, &state.config.catalog.separator)?;
    let provider_name = provider_name.to_owned();
    let upstream_model = upstream_model.to_owned();
    let provider = state
        .config
        .providers
        .get(&provider_name)
        .filter(|provider| provider.enabled)
        .ok_or_else(|| {
            ProxyError::bad_request(format!("unknown or disabled provider `{provider_name}`"))
        })?;
    let allowed_models = match state.mode {
        ServiceMode::Base => &provider.models,
        ServiceMode::OpenAiCompact => &provider.remote_compaction_models,
    };
    if !allowed_models.iter().any(|model| model == &upstream_model) {
        return Err(ProxyError::bad_request(format!(
            "model `{upstream_model}` is not enabled for provider `{provider_name}` in {} mode",
            state.mode.name()
        )));
    }

    let path = uri.path().strip_prefix("/v1/").unwrap_or_default();
    let forwarded_model = if state.mode == ServiceMode::OpenAiCompact && path == "responses/compact"
    {
        format!("{upstream_model}{}", crate::catalog::OPENAI_COMPACT_SUFFIX)
    } else {
        upstream_model.clone()
    };
    payload["model"] = Value::String(forwarded_model.clone());

    let mut endpoint = provider.endpoint(path).map_err(ProxyError::internal)?;
    endpoint.set_query(uri.query());
    let api_key = provider
        .resolved_api_key(&provider_name)
        .map_err(ProxyError::internal)?;
    let mut request = state
        .client
        .request(method, endpoint)
        .header(AUTHORIZATION, format!("Bearer {api_key}"))
        .json(&payload);
    for (name, value) in &headers {
        if should_forward_request_header(name) {
            request = request.header(name, value);
        }
    }

    info!(
        provider = provider_name,
        model = forwarded_model,
        path,
        mode = state.mode.name(),
        "forwarding request"
    );
    let upstream = request.send().await.map_err(ProxyError::bad_gateway)?;
    response_from_upstream(upstream)
}

fn response_from_upstream(upstream: reqwest::Response) -> Result<Response, ProxyError> {
    let status = upstream.status();
    let headers = upstream.headers().clone();
    let stream = upstream.bytes_stream().map_err(std::io::Error::other);
    let mut response = Response::new(Body::from_stream(stream));
    *response.status_mut() = status;
    for (name, value) in &headers {
        if should_forward_response_header(name) {
            response.headers_mut().append(name, value.clone());
        }
    }
    Ok(response)
}

fn split_model<'a>(model: &'a str, separator: &str) -> Result<(&'a str, &'a str), ProxyError> {
    let (provider, upstream) = model.split_once(separator).ok_or_else(|| {
        ProxyError::bad_request(format!(
            "model `{model}` must use `<provider>{separator}<model>`"
        ))
    })?;
    if provider.is_empty() || upstream.is_empty() {
        return Err(ProxyError::bad_request(
            "provider and upstream model must not be empty",
        ));
    }
    Ok((provider, upstream))
}

fn should_forward_request_header(name: &HeaderName) -> bool {
    name != AUTHORIZATION
        && name != HOST
        && name != CONTENT_LENGTH
        && name != CONNECTION
        && name != TRANSFER_ENCODING
}

fn should_forward_response_header(name: &HeaderName) -> bool {
    name != CONTENT_LENGTH && name != CONNECTION && name != TRANSFER_ENCODING
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    info!("shutdown signal received");
}

#[derive(Debug)]
struct ProxyError {
    status: StatusCode,
    message: String,
}

impl ProxyError {
    fn bad_request(error: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: error.to_string(),
        }
    }
    fn bad_gateway(error: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::BAD_GATEWAY,
            message: error.to_string(),
        }
    }
    fn internal(error: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: error.to_string(),
        }
    }
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let body = Json(json!({ "error": { "message": self.message, "type": "router_error" } }));
        (self.status, body).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CatalogConfig, ProviderConfig, ServerConfig};
    use axum::extract::State;
    use axum::http::HeaderMap;
    use axum::routing::post;
    use axum::{Json, Router};
    use std::collections::BTreeMap;
    use std::net::SocketAddr;
    use std::sync::{Arc, Mutex};
    use url::Url;

    #[test]
    fn splits_only_the_provider_prefix() {
        assert_eq!(
            split_model("alpha/org/model", "/").unwrap(),
            ("alpha", "org/model")
        );
    }

    #[test]
    fn rejects_unrouted_model() {
        assert!(split_model("model-only", "/").is_err());
    }

    #[tokio::test]
    async fn rewrites_model_and_upstream_authorization() {
        #[derive(Clone, Default)]
        struct Capture(Arc<Mutex<Option<(String, String)>>>);

        async fn upstream(
            State(capture): State<Capture>,
            headers: HeaderMap,
            Json(payload): Json<Value>,
        ) -> Response {
            let authorization = headers
                .get(AUTHORIZATION)
                .and_then(|value| value.to_str().ok())
                .unwrap_or_default()
                .to_owned();
            let model = payload["model"].as_str().unwrap_or_default().to_owned();
            *capture.0.lock().unwrap() = Some((authorization, model));
            Response::builder()
                .header("content-type", "text/event-stream")
                .body(Body::from("data: routed\n\n"))
                .unwrap()
        }

        let capture = Capture::default();
        let upstream_app = Router::new()
            .route("/v1/responses", post(upstream))
            .with_state(capture.clone());
        let (upstream_addr, upstream_task) = spawn(upstream_app).await;

        let mut providers = BTreeMap::new();
        providers.insert(
            "alpha".to_owned(),
            ProviderConfig {
                base_url: Url::parse(&format!("http://{upstream_addr}/v1")).unwrap(),
                api_key: Some("upstream-secret".to_owned()),
                api_key_env: None,
                enabled: true,
                models: vec!["org/model".to_owned()],
                remote_compaction_models: vec!["org/model".to_owned()],
            },
        );
        let config = Config {
            server: ServerConfig {
                listen: "127.0.0.1:0".parse().unwrap(),
                openai_compact_listen: "127.0.0.1:1".parse().unwrap(),
            },
            catalog: CatalogConfig {
                separator: "/".to_owned(),
                context_window: 128_000,
            },
            providers,
        };
        let router_state = AppState {
            config: Arc::new(config),
            client: reqwest::Client::new(),
            mode: ServiceMode::Base,
        };
        let router_app = Router::new()
            .route("/v1/*path", any(forward))
            .with_state(router_state);
        let (router_addr, router_task) = spawn(router_app).await;

        let response = reqwest::Client::new()
            .post(format!("http://{router_addr}/v1/responses"))
            .json(&json!({ "model": "alpha/org/model", "stream": true }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.text().await.unwrap(), "data: routed\n\n");
        assert_eq!(
            capture.0.lock().unwrap().clone(),
            Some(("Bearer upstream-secret".to_owned(), "org/model".to_owned()))
        );

        let rejected = reqwest::Client::new()
            .post(format!("http://{router_addr}/v1/responses"))
            .json(&json!({ "model": "alpha/not-selected" }))
            .send()
            .await
            .unwrap();
        assert_eq!(rejected.status(), StatusCode::BAD_REQUEST);

        router_task.abort();
        upstream_task.abort();
    }

    #[tokio::test]
    async fn compact_mode_rewrites_only_remote_compaction_requests() {
        #[derive(Clone, Default)]
        struct Capture(Arc<Mutex<Vec<String>>>);

        async fn upstream(State(capture): State<Capture>, Json(payload): Json<Value>) -> Response {
            capture
                .0
                .lock()
                .unwrap()
                .push(payload["model"].as_str().unwrap_or_default().to_owned());
            Json(json!({ "ok": true })).into_response()
        }

        let capture = Capture::default();
        let upstream_app = Router::new()
            .route("/v1/responses", post(upstream))
            .route("/v1/responses/compact", post(upstream))
            .with_state(capture.clone());
        let (upstream_addr, upstream_task) = spawn(upstream_app).await;

        let provider = ProviderConfig {
            base_url: Url::parse(&format!("http://{upstream_addr}/v1")).unwrap(),
            api_key: Some("upstream-secret".to_owned()),
            api_key_env: None,
            enabled: true,
            models: vec!["sol".to_owned(), "terra".to_owned()],
            remote_compaction_models: vec!["sol".to_owned()],
        };
        let config = Config {
            server: ServerConfig::default(),
            catalog: CatalogConfig::default(),
            providers: BTreeMap::from([("alpha".to_owned(), provider)]),
        };
        let router_state = AppState {
            config: Arc::new(config),
            client: reqwest::Client::new(),
            mode: ServiceMode::OpenAiCompact,
        };
        let router_app = Router::new()
            .route("/v1/*path", any(forward))
            .with_state(router_state);
        let (router_addr, router_task) = spawn(router_app).await;
        let client = reqwest::Client::new();

        let normal = client
            .post(format!("http://{router_addr}/v1/responses"))
            .json(&json!({ "model": "alpha/sol" }))
            .send()
            .await
            .unwrap();
        assert_eq!(normal.status(), StatusCode::OK);

        let compact = client
            .post(format!("http://{router_addr}/v1/responses/compact"))
            .json(&json!({ "model": "alpha/sol" }))
            .send()
            .await
            .unwrap();
        assert_eq!(compact.status(), StatusCode::OK);

        let unsupported = client
            .post(format!("http://{router_addr}/v1/responses"))
            .json(&json!({ "model": "alpha/terra" }))
            .send()
            .await
            .unwrap();
        assert_eq!(unsupported.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            capture.0.lock().unwrap().as_slice(),
            ["sol", "sol-openai-compact"]
        );

        router_task.abort();
        upstream_task.abort();
    }

    async fn spawn(app: Router) -> (SocketAddr, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (address, task)
    }
}
