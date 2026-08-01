use crate::chat::{
    ChatAssistantMessage, ChatChunkChoice, ChatClient, ChatCompletionChunk, ChatCompletionResponse,
    ChatDelta, ChatFunctionCallDelta, ChatMessage, ChatResponse, ChatRole, ChatStream,
    ChatStreamItem, ChatToolCallDelta, ChatUsage,
};
use crate::config::{Config, ProviderConfig};
use crate::responses::{
    chat_completion_to_responses, responses_request_to_chat_with_web_search, responses_sse_event,
    ResponsesStreamConverter, ToolRegistry,
};
use crate::web_search::{
    WebSearchBackend, WebSearchRequest, INTERNAL_WEB_SEARCH_TOOL_NAME, MAX_WEB_SEARCH_ROUNDS,
};
use anyhow::{bail, Context, Result};
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::header::{
    AUTHORIZATION, CONNECTION, CONTENT_LENGTH, CONTENT_TYPE, HOST, TRANSFER_ENCODING,
};
use axum::http::{HeaderMap, HeaderName, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get};
use axum::{Json, Router};
use futures_util::{future::join_all, stream, TryStreamExt};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::watch;
use tower_http::trace::TraceLayer;
use tracing::info;

#[derive(Clone)]
struct AppState {
    config: Arc<Config>,
    client: reqwest::Client,
    mode: ServiceMode,
    web_search: Option<Arc<dyn WebSearchBackend>>,
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
        web_search: None,
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

    if state.mode == ServiceMode::Base
        && method == Method::POST
        && path == "responses"
        && provider
            .chat_models
            .iter()
            .any(|model| model == &upstream_model)
    {
        info!(
            provider = provider_name,
            model = forwarded_model,
            path,
            mode = state.mode.name(),
            transport = "chat-completions",
            "converting Responses request"
        );
        return forward_responses_via_chat(
            &state.client,
            &provider_name,
            provider,
            &payload,
            state.web_search.clone(),
        )
        .await;
    }

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

async fn forward_responses_via_chat(
    client: &reqwest::Client,
    provider_name: &str,
    provider: &ProviderConfig,
    payload: &Value,
    web_search: Option<Arc<dyn WebSearchBackend>>,
) -> Result<Response, ProxyError> {
    let converted = responses_request_to_chat_with_web_search(payload, web_search.is_some())
        .map_err(ProxyError::bad_request)?;
    let requested_model = converted.request.model.clone();
    let bridge_active = converted
        .request
        .tools
        .iter()
        .any(|tool| tool.function.name == INTERNAL_WEB_SEARCH_TOOL_NAME);
    let Some(web_search) = web_search.filter(|_| bridge_active) else {
        let response = ChatClient::from_client(client.clone())
            .send(provider_name, provider, &converted.request)
            .await
            .map_err(ProxyError::bad_gateway)?;

        return match response {
            ChatResponse::Completion(completion) => {
                let response =
                    chat_completion_to_responses(&completion, &requested_model, &converted.tools)
                        .map_err(ProxyError::bad_gateway)?;
                Ok(Json(response).into_response())
            }
            ChatResponse::Stream(stream) => Ok(chat_stream_to_responses(
                stream,
                requested_model,
                converted.tools,
            )),
        };
    };

    let original_stream = converted.request.stream;
    let mut request = converted.request;
    request.stream = false;
    request.stream_options = None;
    let chat_client = ChatClient::from_client(client.clone());
    let mut search_rounds = 0;
    let mut usage = UsageAccumulator::default();

    loop {
        let mut completion = chat_client
            .send(provider_name, provider, &request)
            .await
            .map_err(ProxyError::bad_gateway)?
            .into_completion(&requested_model)
            .await
            .map_err(ProxyError::bad_gateway)?;
        usage.add(completion.usage.as_ref());

        let choice = completion
            .choices
            .iter()
            .find(|choice| choice.index == 0)
            .or_else(|| completion.choices.first())
            .ok_or_else(|| ProxyError::bad_gateway("Chat completion contained no choices"))?;
        let internal_calls = choice
            .message
            .tool_calls
            .iter()
            .filter(|call| call.function.name == INTERNAL_WEB_SEARCH_TOOL_NAME)
            .collect::<Vec<_>>();
        let has_external_calls = choice
            .message
            .tool_calls
            .iter()
            .any(|call| call.function.name != INTERNAL_WEB_SEARCH_TOOL_NAME);

        if internal_calls.is_empty() {
            completion.usage = usage.finish();
            return if original_stream {
                completion_to_responses_stream(completion, requested_model, converted.tools)
                    .map_err(ProxyError::bad_gateway)
            } else {
                let response =
                    chat_completion_to_responses(&completion, &requested_model, &converted.tools)
                        .map_err(ProxyError::bad_gateway)?;
                Ok(Json(response).into_response())
            };
        }

        if has_external_calls {
            return Err(ProxyError::bad_gateway(
                "Chat model returned Router Web Search and external tool calls in the same round",
            ));
        }
        if search_rounds == MAX_WEB_SEARCH_ROUNDS {
            return Err(ProxyError::bad_gateway(format!(
                "Chat model exceeded the maximum of {MAX_WEB_SEARCH_ROUNDS} Web Search rounds"
            )));
        }
        search_rounds += 1;

        let searches = internal_calls
            .iter()
            .map(|call| {
                WebSearchRequest::from_arguments(&call.function.arguments)
                    .map(|search| (call.id.clone(), search))
            })
            .collect::<Result<Vec<_>>>()
            .map_err(ProxyError::bad_gateway)?;
        let results = join_all(
            searches
                .iter()
                .map(|(_, search)| web_search.search(search.clone())),
        )
        .await;

        request
            .messages
            .push(assistant_message_for_follow_up(&choice.message));
        for ((call_id, _), result) in searches.into_iter().zip(results) {
            let content = result
                .map_err(ProxyError::bad_gateway)?
                .into_tool_content()
                .map_err(ProxyError::bad_gateway)?;
            request.messages.push(ChatMessage {
                role: ChatRole::Tool,
                content: Some(content),
                name: None,
                tool_call_id: Some(call_id),
                tool_calls: Vec::new(),
                reasoning_content: None,
            });
        }
        request.tool_choice = None;
    }
}

fn assistant_message_for_follow_up(message: &ChatAssistantMessage) -> ChatMessage {
    ChatMessage {
        role: ChatRole::Assistant,
        content: message.content.clone().or_else(|| message.refusal.clone()),
        name: None,
        tool_call_id: None,
        tool_calls: message.tool_calls.clone(),
        reasoning_content: message.reasoning_content.clone(),
    }
}

#[derive(Debug, Default)]
struct UsageAccumulator {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    total_tokens: Option<u64>,
    cached_tokens: u64,
    reasoning_tokens: u64,
    saw_usage: bool,
}

impl UsageAccumulator {
    fn add(&mut self, usage: Option<&ChatUsage>) {
        let Some(usage) = usage else {
            return;
        };
        self.saw_usage = true;
        add_optional(&mut self.prompt_tokens, usage.prompt_tokens);
        add_optional(&mut self.completion_tokens, usage.completion_tokens);
        add_optional(&mut self.total_tokens, usage.total_tokens);
        self.cached_tokens += usage
            .details
            .get("prompt_tokens_details")
            .and_then(|value| value.get("cached_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        self.reasoning_tokens += usage
            .details
            .get("completion_tokens_details")
            .and_then(|value| value.get("reasoning_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
    }

    fn finish(&self) -> Option<ChatUsage> {
        self.saw_usage.then(|| ChatUsage {
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            total_tokens: self.total_tokens,
            details: std::collections::BTreeMap::from([
                (
                    "prompt_tokens_details".to_owned(),
                    json!({ "cached_tokens": self.cached_tokens }),
                ),
                (
                    "completion_tokens_details".to_owned(),
                    json!({ "reasoning_tokens": self.reasoning_tokens }),
                ),
            ]),
        })
    }
}

fn add_optional(total: &mut Option<u64>, value: Option<u64>) {
    if let Some(value) = value {
        *total = Some(total.unwrap_or(0) + value);
    }
}

fn completion_to_responses_stream(
    completion: ChatCompletionResponse,
    requested_model: String,
    tools: ToolRegistry,
) -> Result<Response> {
    let id = completion.id.clone();
    let model = completion.model.clone();
    let usage = completion.usage.clone();
    let items = completion
        .choices
        .into_iter()
        .map(|choice| ChatChunkChoice {
            index: choice.index,
            delta: ChatDelta {
                role: choice.message.role,
                content: choice.message.content,
                reasoning_content: choice.message.reasoning_content,
                reasoning: None,
                refusal: choice.message.refusal,
                tool_calls: choice
                    .message
                    .tool_calls
                    .into_iter()
                    .enumerate()
                    .map(|(index, call)| ChatToolCallDelta {
                        index,
                        id: Some(call.id),
                        kind: Some(call.kind),
                        function: Some(ChatFunctionCallDelta {
                            name: Some(call.function.name),
                            arguments: Some(call.function.arguments),
                        }),
                    })
                    .collect(),
            },
            finish_reason: choice.finish_reason,
        })
        .collect();
    let chunk = ChatCompletionChunk {
        id,
        model,
        choices: items,
        usage,
    };
    let mut converter = ResponsesStreamConverter::new(requested_model, tools);
    let mut body = String::new();
    for item in [ChatStreamItem::Chunk(chunk), ChatStreamItem::Done] {
        for event in converter.push(item)? {
            body.push_str(&responses_sse_event(&event)?);
        }
    }
    Ok(Response::builder()
        .header(CONTENT_TYPE, "text/event-stream")
        .body(Body::from(body))
        .expect("valid buffered Responses stream response"))
}

fn chat_stream_to_responses(
    stream: ChatStream,
    requested_model: String,
    tools: ToolRegistry,
) -> Response {
    struct StreamState {
        upstream: ChatStream,
        converter: ResponsesStreamConverter,
        pending: VecDeque<Bytes>,
        finished: bool,
    }

    let state = StreamState {
        upstream: stream,
        converter: ResponsesStreamConverter::new(requested_model, tools),
        pending: VecDeque::new(),
        finished: false,
    };
    let body_stream = stream::try_unfold(state, |mut state| async move {
        loop {
            if let Some(bytes) = state.pending.pop_front() {
                return Ok::<_, std::io::Error>(Some((bytes, state)));
            }
            if state.finished {
                return Ok(None);
            }

            let events = match state
                .upstream
                .next_event()
                .await
                .map_err(std::io::Error::other)?
            {
                Some(item) => {
                    let done = matches!(&item, crate::chat::ChatStreamItem::Done);
                    let events = state.converter.push(item).map_err(std::io::Error::other)?;
                    if done {
                        state.finished = true;
                    }
                    events
                }
                None => {
                    state.finished = true;
                    state.converter.finish().map_err(std::io::Error::other)?
                }
            };
            for event in events {
                let serialized = responses_sse_event(&event).map_err(std::io::Error::other)?;
                state.pending.push_back(Bytes::from(serialized));
            }
        }
    });

    Response::builder()
        .header(CONTENT_TYPE, "text/event-stream")
        .body(Body::from_stream(body_stream))
        .expect("valid Responses stream response")
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
    use futures_util::future::BoxFuture;
    use std::collections::BTreeMap;
    use std::net::SocketAddr;
    use std::sync::{Arc, Mutex};
    use url::Url;

    #[derive(Clone, Default)]
    struct FakeWebSearch {
        queries: Arc<Mutex<Vec<String>>>,
    }

    impl WebSearchBackend for FakeWebSearch {
        fn search(
            &self,
            request: WebSearchRequest,
        ) -> BoxFuture<'static, Result<crate::web_search::WebSearchResult>> {
            self.queries.lock().unwrap().push(request.query.clone());
            Box::pin(async move {
                Ok(crate::web_search::WebSearchResult {
                    output: Value::String(format!("results for {}", request.query)),
                })
            })
        }
    }

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
                chat_models: Vec::new(),
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
            web_search: None,
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
    async fn converts_non_streaming_responses_for_chat_models() {
        #[derive(Clone, Default)]
        struct Capture(Arc<Mutex<Option<Value>>>);

        async fn upstream(
            State(capture): State<Capture>,
            headers: HeaderMap,
            Json(payload): Json<Value>,
        ) -> Response {
            assert_eq!(
                headers
                    .get(AUTHORIZATION)
                    .and_then(|value| value.to_str().ok()),
                Some("Bearer upstream-secret")
            );
            *capture.0.lock().unwrap() = Some(payload);
            Json(json!({
                "id": "chat-1",
                "model": "glm-test",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "converted reply"
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 3,
                    "total_tokens": 11
                }
            }))
            .into_response()
        }

        let capture = Capture::default();
        let upstream_app = Router::new()
            .route("/v1/chat/completions", post(upstream))
            .with_state(capture.clone());
        let (upstream_addr, upstream_task) = spawn(upstream_app).await;
        let config = Config {
            providers: BTreeMap::from([(
                "alpha".to_owned(),
                ProviderConfig {
                    base_url: Url::parse(&format!("http://{upstream_addr}/v1")).unwrap(),
                    api_key: Some("upstream-secret".to_owned()),
                    api_key_env: None,
                    enabled: true,
                    models: vec!["glm-test".to_owned()],
                    chat_models: vec!["glm-test".to_owned()],
                    remote_compaction_models: Vec::new(),
                },
            )]),
            ..Config::default()
        };
        let router_state = AppState {
            config: Arc::new(config),
            client: reqwest::Client::new(),
            mode: ServiceMode::Base,
            web_search: None,
        };
        let router_app = Router::new()
            .route("/v1/*path", any(forward))
            .with_state(router_state);
        let (router_addr, router_task) = spawn(router_app).await;

        let response = reqwest::Client::new()
            .post(format!("http://{router_addr}/v1/responses"))
            .json(&json!({
                "model": "alpha/glm-test",
                "instructions": "Be concise.",
                "input": "hello",
                "reasoning": { "effort": "none" }
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let response: Value = response.json().await.unwrap();
        assert_eq!(response["object"], "response");
        assert_eq!(response["model"], "glm-test");
        assert_eq!(
            response["output"][0]["content"][0]["text"],
            "converted reply"
        );
        assert_eq!(response["usage"]["total_tokens"], 11);

        let captured = capture.0.lock().unwrap().clone().unwrap();
        assert_eq!(captured["model"], "glm-test");
        assert_eq!(captured["stream"], false);
        assert_eq!(captured["messages"][0]["role"], "system");
        assert_eq!(captured["messages"][0]["content"], "Be concise.");
        assert_eq!(captured["messages"][1]["role"], "user");
        assert_eq!(captured["messages"][1]["content"], "hello");
        assert!(captured.get("reasoning_effort").is_none());

        router_task.abort();
        upstream_task.abort();
    }

    #[tokio::test]
    async fn converts_namespace_tools_through_chat_models() {
        #[derive(Clone, Default)]
        struct Capture(Arc<Mutex<Option<Value>>>);

        async fn upstream(State(capture): State<Capture>, Json(payload): Json<Value>) -> Response {
            *capture.0.lock().unwrap() = Some(payload);
            Json(json!({
                "id": "chat-namespace",
                "model": "glm-test",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "collaboration__spawn_agent",
                                "arguments": "{\"task\":\"inspect\"}"
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }]
            }))
            .into_response()
        }

        let capture = Capture::default();
        let upstream_app = Router::new()
            .route("/v1/chat/completions", post(upstream))
            .with_state(capture.clone());
        let (upstream_addr, upstream_task) = spawn(upstream_app).await;
        let config = Config {
            providers: BTreeMap::from([(
                "alpha".to_owned(),
                ProviderConfig {
                    base_url: Url::parse(&format!("http://{upstream_addr}/v1")).unwrap(),
                    api_key: Some("upstream-secret".to_owned()),
                    api_key_env: None,
                    enabled: true,
                    models: vec!["glm-test".to_owned()],
                    chat_models: vec!["glm-test".to_owned()],
                    remote_compaction_models: Vec::new(),
                },
            )]),
            ..Config::default()
        };
        let router_state = AppState {
            config: Arc::new(config),
            client: reqwest::Client::new(),
            mode: ServiceMode::Base,
            web_search: None,
        };
        let router_app = Router::new()
            .route("/v1/*path", any(forward))
            .with_state(router_state);
        let (router_addr, router_task) = spawn(router_app).await;

        let response = reqwest::Client::new()
            .post(format!("http://{router_addr}/v1/responses"))
            .json(&json!({
                "model": "alpha/glm-test",
                "input": "inspect",
                "tools": [{
                    "type": "namespace",
                    "name": "collaboration",
                    "tools": [{
                        "type": "function",
                        "name": "spawn_agent",
                        "parameters": {
                            "type": "object",
                            "properties": { "task": { "type": "string" } }
                        }
                    }]
                }]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let response: Value = response.json().await.unwrap();
        assert_eq!(response["output"][0]["type"], "function_call");
        assert_eq!(response["output"][0]["namespace"], "collaboration");
        assert_eq!(response["output"][0]["name"], "spawn_agent");

        let captured = capture.0.lock().unwrap().clone().unwrap();
        assert_eq!(
            captured["tools"][0]["function"]["name"],
            "collaboration__spawn_agent"
        );

        router_task.abort();
        upstream_task.abort();
    }

    #[tokio::test]
    async fn converts_tool_search_and_omits_web_search_for_chat_models() {
        #[derive(Clone, Default)]
        struct Capture(Arc<Mutex<Option<Value>>>);

        async fn upstream(State(capture): State<Capture>, Json(payload): Json<Value>) -> Response {
            *capture.0.lock().unwrap() = Some(payload);
            Json(json!({
                "id": "chat-tool-search",
                "model": "glm-test",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "search-1",
                            "type": "function",
                            "function": {
                                "name": "tool_search",
                                "arguments": "{\"query\":\"calendar\",\"limit\":1}"
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }]
            }))
            .into_response()
        }

        let capture = Capture::default();
        let upstream_app = Router::new()
            .route("/v1/chat/completions", post(upstream))
            .with_state(capture.clone());
        let (upstream_addr, upstream_task) = spawn(upstream_app).await;
        let config = Config {
            providers: BTreeMap::from([(
                "alpha".to_owned(),
                ProviderConfig {
                    base_url: Url::parse(&format!("http://{upstream_addr}/v1")).unwrap(),
                    api_key: Some("upstream-secret".to_owned()),
                    api_key_env: None,
                    enabled: true,
                    models: vec!["glm-test".to_owned()],
                    chat_models: vec!["glm-test".to_owned()],
                    remote_compaction_models: Vec::new(),
                },
            )]),
            ..Config::default()
        };
        let router_state = AppState {
            config: Arc::new(config),
            client: reqwest::Client::new(),
            mode: ServiceMode::Base,
            web_search: None,
        };
        let router_app = Router::new()
            .route("/v1/*path", any(forward))
            .with_state(router_state);
        let (router_addr, router_task) = spawn(router_app).await;

        let response = reqwest::Client::new()
            .post(format!("http://{router_addr}/v1/responses"))
            .json(&json!({
                "model": "alpha/glm-test",
                "input": "find a calendar tool",
                "tools": [
                    {
                        "type": "tool_search",
                        "execution": "client",
                        "description": "Search available tools",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": { "type": "string" },
                                "limit": { "type": "integer" }
                            },
                            "required": ["query"]
                        }
                    },
                    {
                        "type": "web_search",
                        "external_web_access": true
                    }
                ]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let response: Value = response.json().await.unwrap();
        assert_eq!(response["output"][0]["type"], "tool_search_call");
        assert_eq!(response["output"][0]["call_id"], "search-1");
        assert_eq!(response["output"][0]["execution"], "client");
        assert_eq!(
            response["output"][0]["arguments"],
            json!({ "query": "calendar", "limit": 1 })
        );

        let captured = capture.0.lock().unwrap().clone().unwrap();
        assert_eq!(captured["tools"].as_array().unwrap().len(), 1);
        assert_eq!(captured["tools"][0]["function"]["name"], "tool_search");
        assert!(captured["tools"]
            .as_array()
            .unwrap()
            .iter()
            .all(|tool| tool["function"]["name"] != "web_search"));

        router_task.abort();
        upstream_task.abort();
    }

    #[tokio::test]
    async fn converts_streaming_chat_events_to_responses_sse() {
        async fn upstream(Json(payload): Json<Value>) -> Response {
            assert_eq!(payload["model"], "glm-test");
            assert_eq!(payload["stream"], true);
            assert_eq!(payload["stream_options"]["include_usage"], true);
            Response::builder()
                .header(CONTENT_TYPE, "text/event-stream")
                .body(Body::from(concat!(
                    "data: {\"id\":\"chat-1\",\"model\":\"glm-test\",\"choices\":[",
                    "{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hello\"}}]}\n\n",
                    "data: {\"id\":\"chat-1\",\"choices\":[],\"usage\":",
                    "{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n\n",
                    "data: [DONE]\n\n"
                )))
                .unwrap()
        }

        let upstream_app = Router::new().route("/v1/chat/completions", post(upstream));
        let (upstream_addr, upstream_task) = spawn(upstream_app).await;
        let config = Config {
            providers: BTreeMap::from([(
                "alpha".to_owned(),
                ProviderConfig {
                    base_url: Url::parse(&format!("http://{upstream_addr}/v1")).unwrap(),
                    api_key: Some("upstream-secret".to_owned()),
                    api_key_env: None,
                    enabled: true,
                    models: vec!["glm-test".to_owned()],
                    chat_models: vec!["glm-test".to_owned()],
                    remote_compaction_models: Vec::new(),
                },
            )]),
            ..Config::default()
        };
        let router_state = AppState {
            config: Arc::new(config),
            client: reqwest::Client::new(),
            mode: ServiceMode::Base,
            web_search: Some(Arc::new(FakeWebSearch::default())),
        };
        let router_app = Router::new()
            .route("/v1/*path", any(forward))
            .with_state(router_state);
        let (router_addr, router_task) = spawn(router_app).await;

        let response = reqwest::Client::new()
            .post(format!("http://{router_addr}/v1/responses"))
            .json(&json!({
                "model": "alpha/glm-test",
                "input": "hello",
                "stream": true
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get(CONTENT_TYPE)
                .and_then(|value| value.to_str().ok()),
            Some("text/event-stream")
        );
        let body = response.text().await.unwrap();
        assert!(body.contains("event: response.created"));
        assert!(body.contains("event: response.output_text.delta"));
        assert!(body.contains("\"delta\":\"hello\""));
        assert!(body.contains("event: response.completed"));
        assert!(body.contains("\"total_tokens\":7"));

        router_task.abort();
        upstream_task.abort();
    }

    #[tokio::test]
    async fn bridges_web_search_and_returns_buffered_responses_stream() {
        #[derive(Clone, Default)]
        struct Capture(Arc<Mutex<Vec<Value>>>);

        async fn upstream(State(capture): State<Capture>, Json(payload): Json<Value>) -> Response {
            let mut requests = capture.0.lock().unwrap();
            requests.push(payload);
            let round = requests.len();
            drop(requests);
            if round == 1 {
                Json(json!({
                    "id": "chat-search-1",
                    "model": "glm-test",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "tool_calls": [{
                                "id": "search-call-1",
                                "type": "function",
                                "function": {
                                    "name": INTERNAL_WEB_SEARCH_TOOL_NAME,
                                    "arguments": "{\"query\":\"current Rust release\"}"
                                }
                            }, {
                                "id": "search-call-2",
                                "type": "function",
                                "function": {
                                    "name": INTERNAL_WEB_SEARCH_TOOL_NAME,
                                    "arguments": "{\"query\":\"Rust release date\"}"
                                }
                            }]
                        },
                        "finish_reason": "tool_calls"
                    }],
                    "usage": {
                        "prompt_tokens": 4,
                        "completion_tokens": 2,
                        "total_tokens": 6
                    }
                }))
                .into_response()
            } else {
                Json(json!({
                    "id": "chat-search-2",
                    "model": "glm-test",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Rust search completed."
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 8,
                        "completion_tokens": 3,
                        "total_tokens": 11
                    }
                }))
                .into_response()
            }
        }

        let capture = Capture::default();
        let upstream_app = Router::new()
            .route("/v1/chat/completions", post(upstream))
            .with_state(capture.clone());
        let (upstream_addr, upstream_task) = spawn(upstream_app).await;
        let config = chat_test_config(upstream_addr);
        let backend = FakeWebSearch::default();
        let router_state = AppState {
            config: Arc::new(config),
            client: reqwest::Client::new(),
            mode: ServiceMode::Base,
            web_search: Some(Arc::new(backend.clone())),
        };
        let router_app = Router::new()
            .route("/v1/*path", any(forward))
            .with_state(router_state);
        let (router_addr, router_task) = spawn(router_app).await;

        let response = reqwest::Client::new()
            .post(format!("http://{router_addr}/v1/responses"))
            .json(&json!({
                "model": "alpha/glm-test",
                "input": "What is the current Rust release?",
                "stream": true,
                "tools": [{ "type": "web_search" }],
                "tool_choice": { "type": "web_search" }
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.text().await.unwrap();
        assert!(body.contains("event: response.output_text.delta"));
        assert!(body.contains("Rust search completed."));
        assert!(body.contains("\"total_tokens\":17"));
        assert_eq!(
            backend.queries.lock().unwrap().as_slice(),
            ["current Rust release", "Rust release date"]
        );

        let requests = capture.0.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0]["stream"], false);
        assert_eq!(
            requests[0]["tools"][0]["function"]["name"],
            INTERNAL_WEB_SEARCH_TOOL_NAME
        );
        assert_eq!(
            requests[1]["messages"][1]["tool_calls"][0]["function"]["name"],
            INTERNAL_WEB_SEARCH_TOOL_NAME
        );
        assert_eq!(requests[1]["messages"][2]["role"], "tool");
        assert_eq!(requests[1]["messages"][2]["tool_call_id"], "search-call-1");
        assert_eq!(
            requests[1]["messages"][2]["content"],
            "results for current Rust release"
        );
        assert_eq!(requests[1]["messages"][3]["tool_call_id"], "search-call-2");
        assert_eq!(
            requests[1]["messages"][3]["content"],
            "results for Rust release date"
        );
        assert!(requests[1].get("tool_choice").is_none());

        router_task.abort();
        upstream_task.abort();
    }

    #[tokio::test]
    async fn rejects_mixed_internal_and_external_tool_calls() {
        async fn upstream() -> Response {
            Json(json!({
                "id": "chat-mixed",
                "model": "glm-test",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "search-call",
                                "type": "function",
                                "function": {
                                    "name": INTERNAL_WEB_SEARCH_TOOL_NAME,
                                    "arguments": "{\"query\":\"news\"}"
                                }
                            },
                            {
                                "id": "shell-call",
                                "type": "function",
                                "function": {
                                    "name": "shell",
                                    "arguments": "{\"cmd\":\"pwd\"}"
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }]
            }))
            .into_response()
        }

        let upstream_app = Router::new().route("/v1/chat/completions", post(upstream));
        let (upstream_addr, upstream_task) = spawn(upstream_app).await;
        let backend = FakeWebSearch::default();
        let router_state = AppState {
            config: Arc::new(chat_test_config(upstream_addr)),
            client: reqwest::Client::new(),
            mode: ServiceMode::Base,
            web_search: Some(Arc::new(backend.clone())),
        };
        let router_app = Router::new()
            .route("/v1/*path", any(forward))
            .with_state(router_state);
        let (router_addr, router_task) = spawn(router_app).await;

        let response = reqwest::Client::new()
            .post(format!("http://{router_addr}/v1/responses"))
            .json(&json!({
                "model": "alpha/glm-test",
                "input": "search and run",
                "tools": [
                    { "type": "web_search" },
                    {
                        "type": "function",
                        "name": "shell",
                        "parameters": { "type": "object", "properties": {} }
                    }
                ]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        let error: Value = response.json().await.unwrap();
        assert!(error["error"]["message"]
            .as_str()
            .unwrap()
            .contains("same round"));
        assert!(backend.queries.lock().unwrap().is_empty());

        router_task.abort();
        upstream_task.abort();
    }

    #[tokio::test]
    async fn limits_web_search_to_eight_rounds() {
        async fn upstream() -> Response {
            Json(json!({
                "id": "chat-loop",
                "model": "glm-test",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "search-call",
                            "type": "function",
                            "function": {
                                "name": INTERNAL_WEB_SEARCH_TOOL_NAME,
                                "arguments": "{\"query\":\"again\"}"
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }]
            }))
            .into_response()
        }

        let upstream_app = Router::new().route("/v1/chat/completions", post(upstream));
        let (upstream_addr, upstream_task) = spawn(upstream_app).await;
        let backend = FakeWebSearch::default();
        let router_state = AppState {
            config: Arc::new(chat_test_config(upstream_addr)),
            client: reqwest::Client::new(),
            mode: ServiceMode::Base,
            web_search: Some(Arc::new(backend.clone())),
        };
        let router_app = Router::new()
            .route("/v1/*path", any(forward))
            .with_state(router_state);
        let (router_addr, router_task) = spawn(router_app).await;

        let response = reqwest::Client::new()
            .post(format!("http://{router_addr}/v1/responses"))
            .json(&json!({
                "model": "alpha/glm-test",
                "input": "keep searching",
                "tools": [{ "type": "web_search" }]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        let error: Value = response.json().await.unwrap();
        assert!(error["error"]["message"]
            .as_str()
            .unwrap()
            .contains("maximum of 8"));
        assert_eq!(backend.queries.lock().unwrap().len(), MAX_WEB_SEARCH_ROUNDS);

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
            chat_models: Vec::new(),
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
            web_search: None,
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

    fn chat_test_config(upstream_addr: SocketAddr) -> Config {
        Config {
            providers: BTreeMap::from([(
                "alpha".to_owned(),
                ProviderConfig {
                    base_url: Url::parse(&format!("http://{upstream_addr}/v1")).unwrap(),
                    api_key: Some("upstream-secret".to_owned()),
                    api_key_env: None,
                    enabled: true,
                    models: vec!["glm-test".to_owned()],
                    chat_models: vec!["glm-test".to_owned()],
                    remote_compaction_models: Vec::new(),
                },
            )]),
            ..Config::default()
        }
    }
}
