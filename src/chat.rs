use crate::config::ProviderConfig;
use anyhow::{bail, Context, Result};
use bytes::Bytes;
use futures_util::stream::BoxStream;
use futures_util::StreamExt;
use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, VecDeque};

#[derive(Debug, Clone)]
pub struct ChatClient {
    client: reqwest::Client,
}

impl ChatClient {
    pub fn new() -> Result<Self> {
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()
            .context("failed to build Chat HTTP client")?;
        Ok(Self { client })
    }

    pub fn from_client(client: reqwest::Client) -> Self {
        Self { client }
    }

    pub async fn send(
        &self,
        provider_name: &str,
        provider: &ProviderConfig,
        request: &ChatCompletionRequest,
    ) -> Result<ChatResponse> {
        let endpoint = provider.endpoint("chat/completions")?;
        let api_key = provider.resolved_api_key(provider_name)?;
        let response = self
            .client
            .post(endpoint)
            .header(AUTHORIZATION, format!("Bearer {api_key}"))
            .header(
                ACCEPT,
                if request.stream {
                    "text/event-stream"
                } else {
                    "application/json"
                },
            )
            .json(request)
            .send()
            .await
            .with_context(|| {
                format!("provider `{provider_name}` Chat Completions request failed")
            })?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            let message = upstream_error_message(&body);
            bail!("provider `{provider_name}` returned {status} from /chat/completions: {message}");
        }

        let is_event_stream = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| {
                value
                    .split(';')
                    .next()
                    .is_some_and(|mime| mime.trim().eq_ignore_ascii_case("text/event-stream"))
            });
        if is_event_stream {
            return Ok(ChatResponse::Stream(ChatStream::new(
                response.bytes_stream().boxed(),
            )));
        }

        let completion = response
            .json::<ChatCompletionResponse>()
            .await
            .with_context(|| {
                format!("provider `{provider_name}` returned an invalid Chat Completions response")
            })?;
        Ok(ChatResponse::Completion(completion))
    }
}

#[derive(Debug)]
pub enum ChatResponse {
    Completion(ChatCompletionResponse),
    Stream(ChatStream),
}

impl ChatResponse {
    pub async fn into_completion(self, requested_model: &str) -> Result<ChatCompletionResponse> {
        match self {
            Self::Completion(completion) => Ok(completion),
            Self::Stream(mut stream) => {
                let mut id = None;
                let mut model = None;
                let mut choices = BTreeMap::<usize, CollectedChoice>::new();
                let mut usage = None;
                while let Some(item) = stream.next_event().await? {
                    let ChatStreamItem::Chunk(chunk) = item else {
                        break;
                    };
                    if chunk.id.is_some() {
                        id = chunk.id;
                    }
                    if chunk.model.is_some() {
                        model = chunk.model;
                    }
                    if chunk.usage.is_some() {
                        usage = chunk.usage;
                    }
                    for choice in chunk.choices {
                        choices.entry(choice.index).or_default().push(choice);
                    }
                }
                if choices.is_empty() {
                    bail!("Chat completion stream contained no choices");
                }
                Ok(ChatCompletionResponse {
                    id,
                    model: model.or_else(|| Some(requested_model.to_owned())),
                    choices: choices
                        .into_iter()
                        .map(|(index, choice)| choice.finish(index))
                        .collect(),
                    usage,
                })
            }
        }
    }
}

#[derive(Debug, Default)]
struct CollectedChoice {
    role: Option<ChatRole>,
    content: Option<Value>,
    reasoning_content: Option<Value>,
    refusal: Option<Value>,
    tool_calls: BTreeMap<usize, CollectedToolCall>,
    finish_reason: Option<String>,
}

impl CollectedChoice {
    fn push(&mut self, choice: ChatChunkChoice) {
        if choice.delta.role.is_some() {
            self.role = choice.delta.role;
        }
        append_stream_value(&mut self.content, choice.delta.content);
        append_stream_value(
            &mut self.reasoning_content,
            choice.delta.reasoning_content.or(choice.delta.reasoning),
        );
        append_stream_value(&mut self.refusal, choice.delta.refusal);
        for delta in choice.delta.tool_calls {
            self.tool_calls.entry(delta.index).or_default().push(delta);
        }
        if choice.finish_reason.is_some() {
            self.finish_reason = choice.finish_reason;
        }
    }

    fn finish(self, index: usize) -> ChatChoice {
        ChatChoice {
            index,
            message: ChatAssistantMessage {
                role: self.role.or(Some(ChatRole::Assistant)),
                content: self.content,
                reasoning_content: self.reasoning_content,
                refusal: self.refusal,
                tool_calls: self
                    .tool_calls
                    .into_values()
                    .map(CollectedToolCall::finish)
                    .collect(),
            },
            finish_reason: self.finish_reason,
        }
    }
}

#[derive(Debug, Default)]
struct CollectedToolCall {
    id: String,
    kind: String,
    name: String,
    arguments: String,
}

impl CollectedToolCall {
    fn push(&mut self, delta: ChatToolCallDelta) {
        if let Some(id) = delta.id {
            self.id.push_str(&id);
        }
        if let Some(kind) = delta.kind {
            self.kind.push_str(&kind);
        }
        if let Some(function) = delta.function {
            if let Some(name) = function.name {
                self.name.push_str(&name);
            }
            if let Some(arguments) = function.arguments {
                self.arguments.push_str(&arguments);
            }
        }
    }

    fn finish(self) -> ChatToolCall {
        ChatToolCall {
            id: self.id,
            kind: if self.kind.is_empty() {
                "function".to_owned()
            } else {
                self.kind
            },
            function: ChatFunctionCall {
                name: self.name,
                arguments: self.arguments,
            },
        }
    }
}

fn append_stream_value(target: &mut Option<Value>, delta: Option<Value>) {
    let Some(delta) = delta else {
        return;
    };
    match (target.as_mut(), delta) {
        (None, delta) => *target = Some(delta),
        (Some(Value::String(current)), Value::String(delta)) => current.push_str(&delta),
        (Some(Value::Array(current)), Value::Array(mut delta)) => current.append(&mut delta),
        (Some(current), delta) => {
            let previous = std::mem::replace(current, Value::Null);
            *current = Value::Array(vec![previous, delta]);
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ChatTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<ChatStreamOptions>,
}

impl ChatCompletionRequest {
    pub fn new(model: impl Into<String>, messages: Vec<ChatMessage>, stream: bool) -> Self {
        Self {
            model: model.into(),
            messages,
            stream,
            tools: Vec::new(),
            tool_choice: None,
            parallel_tool_calls: None,
            reasoning_effort: None,
            max_completion_tokens: None,
            temperature: None,
            stream_options: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatStreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatMessage {
    pub role: ChatRole,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ChatToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<Value>,
}

impl ChatMessage {
    pub fn text(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: Some(Value::String(content.into())),
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
            reasoning_content: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatTool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ChatFunctionDefinition,
}

impl ChatTool {
    pub fn function(function: ChatFunctionDefinition) -> Self {
        Self {
            kind: "function".to_owned(),
            function,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatFunctionDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ChatFunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ChatCompletionResponse {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub choices: Vec<ChatChoice>,
    #[serde(default)]
    pub usage: Option<ChatUsage>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ChatChoice {
    #[serde(default)]
    pub index: usize,
    pub message: ChatAssistantMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ChatAssistantMessage {
    #[serde(default)]
    pub role: Option<ChatRole>,
    #[serde(default)]
    pub content: Option<Value>,
    #[serde(default)]
    pub reasoning_content: Option<Value>,
    #[serde(default)]
    pub refusal: Option<Value>,
    #[serde(default)]
    pub tool_calls: Vec<ChatToolCall>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ChatCompletionChunk {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub choices: Vec<ChatChunkChoice>,
    #[serde(default)]
    pub usage: Option<ChatUsage>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ChatChunkChoice {
    #[serde(default)]
    pub index: usize,
    #[serde(default)]
    pub delta: ChatDelta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct ChatDelta {
    #[serde(default)]
    pub role: Option<ChatRole>,
    #[serde(default)]
    pub content: Option<Value>,
    #[serde(default)]
    pub reasoning_content: Option<Value>,
    #[serde(default)]
    pub reasoning: Option<Value>,
    #[serde(default)]
    pub refusal: Option<Value>,
    #[serde(default)]
    pub tool_calls: Vec<ChatToolCallDelta>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ChatToolCallDelta {
    #[serde(default)]
    pub index: usize,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub function: Option<ChatFunctionCallDelta>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ChatFunctionCallDelta {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
pub struct ChatUsage {
    #[serde(default)]
    pub prompt_tokens: Option<u64>,
    #[serde(default)]
    pub completion_tokens: Option<u64>,
    #[serde(default)]
    pub total_tokens: Option<u64>,
    #[serde(flatten)]
    pub details: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChatStreamItem {
    Chunk(ChatCompletionChunk),
    Done,
}

pub struct ChatStream {
    upstream: BoxStream<'static, reqwest::Result<Bytes>>,
    decoder: ChatSseDecoder,
    pending: VecDeque<ChatStreamItem>,
    closed: bool,
}

impl std::fmt::Debug for ChatStream {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ChatStream")
            .field("decoder", &self.decoder)
            .field("pending", &self.pending)
            .field("closed", &self.closed)
            .finish_non_exhaustive()
    }
}

impl ChatStream {
    fn new(upstream: BoxStream<'static, reqwest::Result<Bytes>>) -> Self {
        Self {
            upstream,
            decoder: ChatSseDecoder::default(),
            pending: VecDeque::new(),
            closed: false,
        }
    }

    pub async fn next_event(&mut self) -> Result<Option<ChatStreamItem>> {
        loop {
            if let Some(item) = self.pending.pop_front() {
                if item == ChatStreamItem::Done {
                    self.closed = true;
                }
                return Ok(Some(item));
            }
            if self.closed {
                return Ok(None);
            }

            match self.upstream.next().await {
                Some(Ok(bytes)) => {
                    self.pending.extend(self.decoder.push(&bytes)?);
                }
                Some(Err(error)) => return Err(error).context("Chat response stream failed"),
                None => {
                    self.pending.extend(self.decoder.finish()?);
                    if self.pending.is_empty() {
                        self.closed = true;
                        return Ok(None);
                    }
                }
            }
        }
    }
}

#[derive(Debug, Default)]
pub struct ChatSseDecoder {
    buffer: Vec<u8>,
    done: bool,
}

impl ChatSseDecoder {
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<ChatStreamItem>> {
        if self.done {
            return Ok(Vec::new());
        }
        self.buffer.extend_from_slice(bytes);
        self.drain_complete_events()
    }

    pub fn finish(&mut self) -> Result<Vec<ChatStreamItem>> {
        if self.done {
            self.buffer.clear();
            return Ok(Vec::new());
        }

        let mut items = self.drain_complete_events()?;
        if !self.buffer.iter().all(u8::is_ascii_whitespace) {
            let block = std::mem::take(&mut self.buffer);
            if let Some(item) = parse_sse_block(&block)? {
                self.done = item == ChatStreamItem::Done;
                items.push(item);
            }
        } else {
            self.buffer.clear();
        }
        if !self.done {
            self.done = true;
            items.push(ChatStreamItem::Done);
        }
        Ok(items)
    }

    fn drain_complete_events(&mut self) -> Result<Vec<ChatStreamItem>> {
        let mut items = Vec::new();
        while let Some((end, delimiter_len)) = find_sse_delimiter(&self.buffer) {
            let block = self.buffer[..end].to_vec();
            self.buffer.drain(..end + delimiter_len);
            let Some(item) = parse_sse_block(&block)? else {
                continue;
            };
            self.done = item == ChatStreamItem::Done;
            items.push(item);
            if self.done {
                self.buffer.clear();
                break;
            }
        }
        Ok(items)
    }
}

fn find_sse_delimiter(bytes: &[u8]) -> Option<(usize, usize)> {
    let line_feed = bytes
        .windows(2)
        .position(|window| window == b"\n\n")
        .map(|position| (position, 2));
    let carriage_return = bytes
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|position| (position, 4));
    match (line_feed, carriage_return) {
        (Some(left), Some(right)) => Some(if left.0 <= right.0 { left } else { right }),
        (Some(delimiter), None) | (None, Some(delimiter)) => Some(delimiter),
        (None, None) => None,
    }
}

fn parse_sse_block(block: &[u8]) -> Result<Option<ChatStreamItem>> {
    let mut data = Vec::new();
    for line in block.split(|byte| *byte == b'\n') {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if line.starts_with(b":") {
            continue;
        }
        let Some(value) = line.strip_prefix(b"data:") else {
            continue;
        };
        let value = value.strip_prefix(b" ").unwrap_or(value);
        if !data.is_empty() {
            data.push(b'\n');
        }
        data.extend_from_slice(value);
    }
    let data = trim_ascii_whitespace(&data);
    if data.is_empty() {
        return Ok(None);
    }
    if data == b"[DONE]" {
        return Ok(Some(ChatStreamItem::Done));
    }

    let value: Value =
        serde_json::from_slice(data).context("invalid JSON in Chat Completions SSE event")?;
    if value.get("error").is_some() {
        bail!(
            "Chat Completions stream returned an error: {}",
            error_message(&value)
        );
    }
    let chunk =
        serde_json::from_value(value).context("invalid Chat Completions chunk in SSE event")?;
    Ok(Some(ChatStreamItem::Chunk(chunk)))
}

fn trim_ascii_whitespace(mut value: &[u8]) -> &[u8] {
    while value.first().is_some_and(u8::is_ascii_whitespace) {
        value = &value[1..];
    }
    while value.last().is_some_and(u8::is_ascii_whitespace) {
        value = &value[..value.len() - 1];
    }
    value
}

fn upstream_error_message(body: &str) -> String {
    serde_json::from_str::<Value>(body)
        .ok()
        .map(|value| error_message(&value))
        .filter(|message| !message.is_empty())
        .unwrap_or_else(|| truncate(body, 500))
}

fn error_message(value: &Value) -> String {
    value
        .pointer("/error/message")
        .and_then(Value::as_str)
        .or_else(|| value.get("message").and_then(Value::as_str))
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| value.to_string())
}

fn truncate(value: &str, maximum: usize) -> String {
    let mut chars = value.chars();
    let prefix = chars.by_ref().take(maximum).collect::<String>();
    if chars.next().is_some() {
        format!("{prefix}...")
    } else {
        prefix
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ProviderConfig;
    use axum::extract::State;
    use axum::http::HeaderMap;
    use axum::routing::post;
    use axum::{Json, Router};
    use serde_json::json;
    use std::sync::{Arc, Mutex};
    use url::Url;

    #[test]
    fn conservative_request_omits_optional_compatibility_fields() {
        let request = ChatCompletionRequest::new(
            "glm-test",
            vec![ChatMessage::text(ChatRole::User, "hello")],
            true,
        );
        assert_eq!(
            serde_json::to_value(request).unwrap(),
            json!({
                "model": "glm-test",
                "messages": [{ "role": "user", "content": "hello" }],
                "stream": true
            })
        );
    }

    #[test]
    fn decoder_handles_fragmented_text_reasoning_tools_and_usage() {
        let mut decoder = ChatSseDecoder::default();
        let first = br#"data: {"id":"chat-1","model":"model-a","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"think","tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"shell","arguments":"{\"cmd\":"}}]}}]}

data: {"id":"chat-1","choices":[{"index":0,"delta":{"content":"hello","tool_calls":[{"index":0,"function":{"arguments":"\"pwd\"}"}}]},"finish_reason":"tool_calls"}],"#;
        let second = br#""usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}

data: [DONE]

"#;

        assert!(decoder.push(&first[..73]).unwrap().is_empty());
        let mut items = decoder.push(&first[73..]).unwrap();
        items.extend(decoder.push(second).unwrap());
        assert_eq!(items.len(), 3);

        let ChatStreamItem::Chunk(first) = &items[0] else {
            panic!("expected first chunk");
        };
        assert_eq!(
            first.choices[0].delta.reasoning_content,
            Some(json!("think"))
        );
        assert_eq!(
            first.choices[0].delta.tool_calls[0]
                .function
                .as_ref()
                .unwrap()
                .name
                .as_deref(),
            Some("shell")
        );

        let ChatStreamItem::Chunk(second) = &items[1] else {
            panic!("expected second chunk");
        };
        assert_eq!(second.choices[0].delta.content, Some(json!("hello")));
        assert_eq!(
            second.choices[0].delta.tool_calls[0]
                .function
                .as_ref()
                .unwrap()
                .arguments
                .as_deref(),
            Some("\"pwd\"}")
        );
        assert_eq!(second.usage.as_ref().unwrap().total_tokens, Some(14));
        assert_eq!(items[2], ChatStreamItem::Done);
    }

    #[tokio::test]
    async fn response_collects_streamed_text_tool_call_and_usage() {
        let upstream = futures_util::stream::iter(vec![Ok(Bytes::from_static(
            br#"data: {"id":"chat-1","model":"model-a","choices":[{"index":0,"delta":{"role":"assistant","content":"hello ","tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"web_","arguments":"{\"query\":"}}]}}]}

data: {"choices":[{"index":0,"delta":{"content":"world","tool_calls":[{"index":0,"function":{"name":"search","arguments":"\"Rust\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}

data: [DONE]

"#,
        ))])
        .boxed();

        let completion = ChatResponse::Stream(ChatStream::new(upstream))
            .into_completion("fallback-model")
            .await
            .unwrap();

        assert_eq!(completion.id.as_deref(), Some("chat-1"));
        assert_eq!(completion.model.as_deref(), Some("model-a"));
        assert_eq!(completion.choices.len(), 1);
        assert_eq!(
            completion.choices[0].message.content,
            Some(json!("hello world"))
        );
        assert_eq!(
            completion.choices[0].message.tool_calls,
            vec![ChatToolCall {
                id: "call-1".to_owned(),
                kind: "function".to_owned(),
                function: ChatFunctionCall {
                    name: "web_search".to_owned(),
                    arguments: "{\"query\":\"Rust\"}".to_owned(),
                },
            }]
        );
        assert_eq!(
            completion.choices[0].finish_reason.as_deref(),
            Some("tool_calls")
        );
        assert_eq!(
            completion.usage,
            Some(ChatUsage {
                prompt_tokens: Some(10),
                completion_tokens: Some(4),
                total_tokens: Some(14),
                details: BTreeMap::new(),
            })
        );
    }

    #[test]
    fn decoder_accepts_crlf_and_emits_done_at_clean_eof() {
        let mut decoder = ChatSseDecoder::default();
        let items = decoder
            .push(
                b"event: message\r\ndata: {\"id\":\"first\",\"choices\":[]}\r\n\r\n\
                  data: {\"id\":\"second\",\"choices\":[]}\n\n",
            )
            .unwrap();
        assert_eq!(items.len(), 2);
        let ChatStreamItem::Chunk(first) = &items[0] else {
            panic!("expected first chunk");
        };
        let ChatStreamItem::Chunk(second) = &items[1] else {
            panic!("expected second chunk");
        };
        assert_eq!(first.id.as_deref(), Some("first"));
        assert_eq!(second.id.as_deref(), Some("second"));
        assert_eq!(decoder.finish().unwrap(), vec![ChatStreamItem::Done]);
    }

    #[test]
    fn decoder_surfaces_stream_errors() {
        let mut decoder = ChatSseDecoder::default();
        let error = decoder
            .push(b"data: {\"error\":{\"message\":\"unsupported tool\"}}\n\n")
            .unwrap_err();
        assert!(error.to_string().contains("unsupported tool"));
    }

    #[tokio::test]
    async fn client_posts_chat_request_with_upstream_authorization() {
        #[derive(Clone, Default)]
        struct Capture(Arc<Mutex<Option<(String, Value)>>>);

        async fn upstream(
            State(capture): State<Capture>,
            headers: HeaderMap,
            Json(payload): Json<Value>,
        ) -> Json<Value> {
            let authorization = headers
                .get(AUTHORIZATION)
                .and_then(|value| value.to_str().ok())
                .unwrap_or_default()
                .to_owned();
            *capture.0.lock().unwrap() = Some((authorization, payload));
            Json(json!({
                "id": "chat-1",
                "model": "model-a",
                "choices": [{
                    "index": 0,
                    "message": { "role": "assistant", "content": "hello" },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "total_tokens": 4
                }
            }))
        }

        let capture = Capture::default();
        let app = Router::new()
            .route("/v1/chat/completions", post(upstream))
            .with_state(capture.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let provider = ProviderConfig {
            base_url: Url::parse(&format!("http://{address}/v1")).unwrap(),
            api_key: Some("upstream-secret".to_owned()),
            api_key_env: None,
            enabled: true,
            models: vec!["model-a".to_owned()],
            chat_models: Vec::new(),
            messages_models: Vec::new(),
            remote_compaction_models: Vec::new(),
        };
        let request = ChatCompletionRequest::new(
            "model-a",
            vec![ChatMessage::text(ChatRole::User, "hello")],
            false,
        );
        let response = ChatClient::new()
            .unwrap()
            .send("alpha", &provider, &request)
            .await
            .unwrap();

        let ChatResponse::Completion(completion) = response else {
            panic!("expected non-stream completion");
        };
        assert_eq!(completion.choices[0].message.content, Some(json!("hello")));
        assert_eq!(
            capture.0.lock().unwrap().clone(),
            Some((
                "Bearer upstream-secret".to_owned(),
                json!({
                    "model": "model-a",
                    "messages": [{ "role": "user", "content": "hello" }],
                    "stream": false
                })
            ))
        );

        task.abort();
    }

    #[tokio::test]
    async fn client_parses_streaming_chat_response() {
        async fn upstream() -> axum::response::Response {
            axum::response::Response::builder()
                .header(CONTENT_TYPE, "text/event-stream; charset=utf-8")
                .body(axum::body::Body::from(
                    "data: {\"id\":\"chat-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"}}]}\n\n\
                     data: [DONE] \n\n",
                ))
                .unwrap()
        }

        let app = Router::new().route("/v1/chat/completions", post(upstream));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let provider = ProviderConfig {
            base_url: Url::parse(&format!("http://{address}/v1")).unwrap(),
            api_key: Some("upstream-secret".to_owned()),
            api_key_env: None,
            enabled: true,
            models: vec!["model-a".to_owned()],
            chat_models: Vec::new(),
            messages_models: Vec::new(),
            remote_compaction_models: Vec::new(),
        };
        let request = ChatCompletionRequest::new(
            "model-a",
            vec![ChatMessage::text(ChatRole::User, "hello")],
            true,
        );
        let response = ChatClient::new()
            .unwrap()
            .send("alpha", &provider, &request)
            .await
            .unwrap();
        let ChatResponse::Stream(mut stream) = response else {
            panic!("expected streaming completion");
        };

        let Some(ChatStreamItem::Chunk(chunk)) = stream.next_event().await.unwrap() else {
            panic!("expected chunk");
        };
        assert_eq!(chunk.choices[0].delta.content, Some(json!("hello")));
        assert_eq!(
            stream.next_event().await.unwrap(),
            Some(ChatStreamItem::Done)
        );
        assert_eq!(stream.next_event().await.unwrap(), None);

        task.abort();
    }
}
