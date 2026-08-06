use crate::chat::{
    ChatAssistantMessage, ChatChunkChoice, ChatCompletionChunk, ChatCompletionResponse, ChatDelta,
    ChatFunctionCall, ChatFunctionCallDelta, ChatRole, ChatToolCall, ChatToolCallDelta, ChatUsage,
};
use crate::config::ProviderConfig;
use crate::responses::{responses_request_to_chat_with_web_search, ToolRegistry};
use anyhow::{bail, Context, Result};
use bytes::Bytes;
use futures_util::{stream::BoxStream, StreamExt};
use reqwest::header::{ACCEPT, CONTENT_TYPE};
use serde_json::{json, Value};

pub struct AnthropicResponse {
    pub response: Option<ChatCompletionResponse>,
    pub stream: Option<AnthropicStream>,
}

pub struct AnthropicStream {
    upstream: BoxStream<'static, reqwest::Result<Bytes>>,
    buffer: String,
    done: bool,
    index: usize,
    id: Option<String>,
    model: Option<String>,
}

pub async fn send(
    client: &reqwest::Client,
    provider_name: &str,
    provider: &ProviderConfig,
    payload: &Value,
    web_search_enabled: bool,
) -> Result<AnthropicResponse> {
    let converted = responses_request_to_chat_with_web_search(payload, web_search_enabled)?;
    let requested_model = converted.request.model.clone();
    if converted.request.max_completion_tokens.is_none() {
        tracing::warn!(
            model = %requested_model,
            "Responses request omitted max_output_tokens; forwarding Anthropic Messages request without max_tokens"
        );
    }
    let body = request_body(&converted.request, &converted.tools)?;
    let endpoint = provider.endpoint("messages")?;
    let key = provider.resolved_api_key(provider_name)?;
    let response = client
        .post(endpoint)
        .header("x-api-key", key)
        .header("anthropic-version", "2023-06-01")
        .header(
            ACCEPT,
            if converted.request.stream {
                "text/event-stream"
            } else {
                "application/json"
            },
        )
        .json(&body)
        .send()
        .await
        .context("Anthropic Messages request failed")?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        bail!("provider `{provider_name}` returned {status} from /messages: {text}");
    }
    let is_stream = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| {
            v.split(';')
                .next()
                .is_some_and(|m| m.trim().eq_ignore_ascii_case("text/event-stream"))
        });
    if is_stream {
        Ok(AnthropicResponse {
            response: None,
            stream: Some(AnthropicStream {
                upstream: response.bytes_stream().boxed(),
                buffer: String::new(),
                done: false,
                index: 0,
                id: None,
                model: Some(requested_model),
            }),
        })
    } else {
        let body = response
            .json::<Value>()
            .await
            .context("invalid Anthropic response")?;
        let stop_reason = body.get("stop_reason").and_then(|value| value.as_str());
        tracing::info!(
            model = %requested_model,
            stop_reason = ?stop_reason,
            usage = ?body.get("usage"),
            "Anthropic Messages response completed"
        );
        if stop_reason == Some("max_tokens") {
            tracing::warn!(
                model = %requested_model,
                usage = ?body.get("usage"),
                "Anthropic Messages response stopped because max_tokens was reached"
            );
        }
        Ok(AnthropicResponse {
            response: Some(response_to_chat(body)?),
            stream: None,
        })
    }
}

fn request_body(
    request: &crate::chat::ChatCompletionRequest,
    _tools: &ToolRegistry,
) -> Result<Value> {
    let mut system = Vec::new();
    let mut messages: Vec<Value> = Vec::new();
    for message in &request.messages {
        if message.role == ChatRole::System || message.role == ChatRole::Developer {
            if let Some(content) = &message.content {
                system.push(content_to_text(content));
            }
            continue;
        }
        let role = match message.role {
            ChatRole::Assistant => "assistant",
            ChatRole::Tool => "user",
            _ => "user",
        };
        let mut content = Vec::new();
        if let Some(text) = &message.content {
            content.push(json!({"type":"text","text":content_to_text(text)}));
        }
        for call in &message.tool_calls {
            content.push(json!({
                "type":"tool_use", "id":call.id, "name":call.function.name,
                "input": serde_json::from_str::<Value>(&call.function.arguments)
                    .unwrap_or_else(|_| Value::String(call.function.arguments.clone()))
            }));
        }
        if message.role == ChatRole::Tool {
            content = vec![json!({
                "type":"tool_result",
                "tool_use_id": message.tool_call_id.clone().unwrap_or_default(),
                "content": content_to_text(message.content.as_ref().unwrap_or(&Value::Null))
            })];
        }
        if let Some(previous) = messages.last_mut() {
            if previous.get("role").and_then(Value::as_str) == Some(role) && role == "user" {
                if let Some(existing) = previous.get_mut("content").and_then(Value::as_array_mut) {
                    existing.extend(content);
                    continue;
                }
            }
        }
        messages.push(json!({"role": role, "content": content}));
    }
    let mut body = json!({
        "model": request.model,
        "messages": messages,
        "stream": request.stream
    });
    if let Some(max_tokens) = request.max_completion_tokens {
        body["max_tokens"] = json!(max_tokens);
    }
    if !system.is_empty() {
        body["system"] = Value::String(system.join("\n\n"));
    }
    if !request.tools.is_empty() {
        body["tools"] = Value::Array(request.tools.iter().map(|tool| {
            json!({"name":tool.function.name, "description":tool.function.description, "input_schema":tool.function.parameters})
        }).collect());
    }
    Ok(body)
}

fn response_to_chat(body: Value) -> Result<ChatCompletionResponse> {
    let content = body
        .get("content")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut text = String::new();
    let mut calls = Vec::new();
    for item in content {
        match item.get("type").and_then(Value::as_str) {
            Some("text") => {
                text.push_str(item.get("text").and_then(Value::as_str).unwrap_or_default())
            }
            Some("tool_use") => calls.push(ChatToolCall {
                id: item
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                kind: "function".into(),
                function: ChatFunctionCall {
                    name: item
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .into(),
                    arguments: serde_json::to_string(item.get("input").unwrap_or(&Value::Null))?,
                },
            }),
            _ => {}
        }
    }
    Ok(ChatCompletionResponse {
        id: body.get("id").and_then(Value::as_str).map(str::to_owned),
        model: body.get("model").and_then(Value::as_str).map(str::to_owned),
        choices: vec![crate::chat::ChatChoice {
            index: 0,
            message: ChatAssistantMessage {
                role: Some(ChatRole::Assistant),
                content: Some(Value::String(text)),
                reasoning_content: None,
                refusal: None,
                tool_calls: calls,
            },
            finish_reason: body
                .get("stop_reason")
                .and_then(Value::as_str)
                .map(str::to_owned),
        }],
        usage: body.get("usage").map(|usage| ChatUsage {
            prompt_tokens: usage.get("input_tokens").and_then(Value::as_u64),
            completion_tokens: usage.get("output_tokens").and_then(Value::as_u64),
            total_tokens: None,
            details: Default::default(),
        }),
    })
}

fn content_to_text(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| value.to_string())
}

impl AnthropicStream {
    pub async fn next_chunk(&mut self) -> Result<Option<ChatCompletionChunk>> {
        loop {
            if let Some(pos) = self.buffer.find("\n\n") {
                let event = self.buffer.drain(..pos + 2).collect::<String>();
                let data = event
                    .lines()
                    .filter_map(|line| line.strip_prefix("data:"))
                    .map(str::trim)
                    .collect::<String>();
                if data.is_empty() || data == "[DONE]" {
                    self.done = true;
                    return Ok(None);
                }
                let value: Value = serde_json::from_str(&data)?;
                if let Some(id) = value
                    .get("message")
                    .and_then(|v| v.get("id"))
                    .and_then(Value::as_str)
                {
                    self.id = Some(id.into());
                }
                if let Some(model) = value
                    .get("message")
                    .and_then(|v| v.get("model"))
                    .and_then(Value::as_str)
                {
                    self.model = Some(model.into());
                }
                let event_type = value
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let delta = if event_type == "content_block_start" {
                    let block = value.get("content_block").unwrap_or(&Value::Null);
                    let kind = block
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    if kind == "tool_use" {
                        ChatDelta {
                            tool_calls: vec![ChatToolCallDelta {
                                index: value.get("index").and_then(Value::as_u64).unwrap_or(0)
                                    as usize,
                                id: block.get("id").and_then(Value::as_str).map(str::to_owned),
                                kind: Some("function".to_owned()),
                                function: Some(ChatFunctionCallDelta {
                                    name: block
                                        .get("name")
                                        .and_then(Value::as_str)
                                        .map(str::to_owned),
                                    arguments: None,
                                }),
                            }],
                            ..Default::default()
                        }
                    } else {
                        ChatDelta::default()
                    }
                } else if event_type == "content_block_delta" {
                    let d = value.get("delta").unwrap_or(&Value::Null);
                    let kind = d.get("type").and_then(Value::as_str).unwrap_or_default();
                    if kind == "input_json_delta" {
                        ChatDelta {
                            tool_calls: vec![ChatToolCallDelta {
                                index: value.get("index").and_then(Value::as_u64).unwrap_or(0)
                                    as usize,
                                id: None,
                                kind: None,
                                function: Some(ChatFunctionCallDelta {
                                    name: None,
                                    arguments: d
                                        .get("partial_json")
                                        .and_then(Value::as_str)
                                        .map(str::to_owned),
                                }),
                            }],
                            ..Default::default()
                        }
                    } else {
                        ChatDelta {
                            content: d.get("text").cloned(),
                            ..Default::default()
                        }
                    }
                } else if event_type == "message_delta" {
                    ChatDelta::default()
                } else {
                    ChatDelta::default()
                };
                if event_type == "message_delta" {
                    let stop_reason = value
                        .get("delta")
                        .and_then(|v| v.get("stop_reason"))
                        .and_then(Value::as_str);
                    tracing::info!(
                        model = self.model.as_deref().unwrap_or_default(),
                        stop_reason = ?stop_reason,
                        usage = ?value.get("usage"),
                        "Anthropic Messages stream completed"
                    );
                    if stop_reason == Some("max_tokens") {
                        tracing::warn!(
                            model = self.model.as_deref().unwrap_or_default(),
                            usage = ?value.get("usage"),
                            "Anthropic Messages response stopped because max_tokens was reached"
                        );
                    }
                    let reason = value
                        .get("delta")
                        .and_then(|v| v.get("stop_reason"))
                        .and_then(Value::as_str)
                        .map(|reason| match reason {
                            "max_tokens" => "length".to_owned(),
                            "tool_use" => "tool_calls".to_owned(),
                            other => other.to_owned(),
                        });
                    return Ok(Some(ChatCompletionChunk {
                        id: self.id.clone(),
                        model: self.model.clone(),
                        choices: vec![ChatChunkChoice {
                            index: self.index,
                            delta,
                            finish_reason: reason,
                        }],
                        usage: None,
                    }));
                }
                if event_type == "content_block_start" || event_type == "content_block_delta" {
                    return Ok(Some(ChatCompletionChunk {
                        id: self.id.clone(),
                        model: self.model.clone(),
                        choices: vec![ChatChunkChoice {
                            index: self.index,
                            delta,
                            finish_reason: None,
                        }],
                        usage: None,
                    }));
                }
                continue;
            }
            match self.upstream.next().await {
                Some(Ok(bytes)) => self.buffer.push_str(std::str::from_utf8(&bytes)?),
                Some(Err(error)) => return Err(error.into()),
                None => {
                    self.done = true;
                    return Ok(None);
                }
            }
        }
    }
}
