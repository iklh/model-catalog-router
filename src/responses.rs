use crate::chat::{
    ChatAssistantMessage, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse,
    ChatFunctionCall, ChatFunctionDefinition, ChatMessage, ChatRole, ChatStreamItem, ChatTool,
    ChatToolCall, ChatUsage,
};
use anyhow::{bail, Context, Result};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponsesToolKind {
    Function,
    Custom,
    ToolSearch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RegisteredTool {
    kind: ResponsesToolKind,
    original_name: String,
    namespace: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolRegistry {
    tools: HashMap<String, RegisteredTool>,
    namespaced_tools: HashMap<(String, String), String>,
}

impl ToolRegistry {
    pub fn kind(&self, name: &str) -> Option<ResponsesToolKind> {
        self.tools.get(name).map(|tool| tool.kind)
    }

    fn insert(&mut self, name: String, kind: ResponsesToolKind) -> Result<()> {
        self.insert_registered(
            name.clone(),
            RegisteredTool {
                kind,
                original_name: name,
                namespace: None,
            },
        )
    }

    fn insert_namespaced(&mut self, namespace: String, name: String) -> Result<String> {
        let flattened = flatten_namespaced_tool_name(&namespace, &name);
        self.insert_registered(
            flattened.clone(),
            RegisteredTool {
                kind: ResponsesToolKind::Function,
                original_name: name.clone(),
                namespace: Some(namespace.clone()),
            },
        )?;
        self.namespaced_tools
            .insert((namespace, name), flattened.clone());
        Ok(flattened)
    }

    fn insert_registered(&mut self, name: String, tool: RegisteredTool) -> Result<()> {
        if let Some(previous) = self.tools.get(&name) {
            if previous != &tool {
                bail!(
                    "tool name collision after namespace flattening: `{name}` refers to multiple tools"
                );
            }
            return Ok(());
        }
        self.tools.insert(name, tool);
        Ok(())
    }

    fn chat_name(&self, namespace: Option<&str>, name: &str) -> String {
        if let Some(namespace) = namespace {
            return self
                .namespaced_tools
                .get(&(namespace.to_owned(), name.to_owned()))
                .cloned()
                .unwrap_or_else(|| flatten_namespaced_tool_name(namespace, name));
        }
        name.to_owned()
    }

    fn response_tool(&self, chat_name: &str) -> Option<&RegisteredTool> {
        self.tools.get(chat_name)
    }
}

fn flatten_namespaced_tool_name(namespace: &str, name: &str) -> String {
    if namespace.ends_with("__") {
        format!("{namespace}{name}")
    } else {
        format!("{namespace}__{name}")
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConvertedChatRequest {
    pub request: ChatCompletionRequest,
    pub tools: ToolRegistry,
}

pub fn responses_request_to_chat(payload: &Value) -> Result<ConvertedChatRequest> {
    let object = payload
        .as_object()
        .context("Responses request must be a JSON object")?;
    let model = required_string(object, "model", "Responses request")?;
    let stream = object
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mut messages = Vec::new();

    if let Some(instructions) = object.get("instructions") {
        let text = value_as_text(instructions, "request `instructions`")?;
        if !text.is_empty() {
            messages.push(ChatMessage::text(ChatRole::System, text));
        }
    }

    let mut registry = ToolRegistry::default();
    let mut tools = convert_tools(object.get("tools"), &mut registry)?;
    tools.extend(convert_input_tools(object.get("input"), &mut registry)?);
    if let Some(input) = object.get("input") {
        convert_input(input, &mut messages, &registry)?;
    } else {
        bail!("Responses request must contain `input`");
    }
    let mut request = ChatCompletionRequest::new(model, messages, stream);
    request.tools = tools;
    request.tool_choice = convert_tool_choice(object.get("tool_choice"), &registry)?;
    request.parallel_tool_calls = optional_bool(object, "parallel_tool_calls")?;
    request.reasoning_effort = object
        .get("reasoning")
        .and_then(Value::as_object)
        .and_then(|reasoning| reasoning.get("effort"))
        .map(|value| value_as_string(value, "`reasoning.effort`"))
        .transpose()?
        .filter(|effort| effort != "none");
    request.max_completion_tokens = optional_u64(object, "max_output_tokens")?;
    request.temperature = optional_f64(object, "temperature")?;
    if stream {
        request.stream_options = Some(crate::chat::ChatStreamOptions {
            include_usage: true,
        });
    }

    Ok(ConvertedChatRequest {
        request,
        tools: registry,
    })
}

fn convert_input_tools(
    input: Option<&Value>,
    registry: &mut ToolRegistry,
) -> Result<Vec<ChatTool>> {
    let Some(Value::Array(items)) = input else {
        return Ok(Vec::new());
    };
    let mut converted = Vec::new();
    for item in items {
        let Some(object) = item.as_object() else {
            continue;
        };
        match object.get("type").and_then(Value::as_str) {
            Some("additional_tools") | Some("tool_search_output") => {
                converted.extend(convert_tools(object.get("tools"), registry)?);
            }
            _ => {}
        }
    }
    Ok(converted)
}

fn convert_input(
    input: &Value,
    messages: &mut Vec<ChatMessage>,
    registry: &ToolRegistry,
) -> Result<()> {
    match input {
        Value::String(text) => {
            messages.push(ChatMessage::text(ChatRole::User, text));
            Ok(())
        }
        Value::Array(items) => {
            for item in items {
                convert_input_item(item, messages, registry)?;
            }
            Ok(())
        }
        _ => bail!("Responses request `input` must be a string or an array"),
    }
}

fn convert_input_item(
    item: &Value,
    messages: &mut Vec<ChatMessage>,
    registry: &ToolRegistry,
) -> Result<()> {
    let object = item
        .as_object()
        .context("each Responses input item must be an object")?;
    let kind = object
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("message");
    match kind {
        "message" => convert_message_item(object, messages),
        "agent_message" => convert_agent_message_item(object, messages),
        "function_call" => {
            let name = required_string(object, "name", "function_call item")?;
            let namespace = object
                .get("namespace")
                .map(|value| value_as_string(value, "function_call item `namespace`"))
                .transpose()?;
            let name = registry.chat_name(namespace.as_deref(), &name);
            let call_id = required_string(object, "call_id", "function_call item")?;
            let arguments = required_string(object, "arguments", "function_call item")?;
            push_assistant_tool_call(
                messages,
                ChatToolCall {
                    id: call_id,
                    kind: "function".to_owned(),
                    function: ChatFunctionCall { name, arguments },
                },
            );
            Ok(())
        }
        "custom_tool_call" => {
            let name = required_string(object, "name", "custom_tool_call item")?;
            let call_id = required_string(object, "call_id", "custom_tool_call item")?;
            let input = required_string(object, "input", "custom_tool_call item")?;
            let arguments = serde_json::to_string(&json!({ "input": input }))?;
            push_assistant_tool_call(
                messages,
                ChatToolCall {
                    id: call_id,
                    kind: "function".to_owned(),
                    function: ChatFunctionCall { name, arguments },
                },
            );
            Ok(())
        }
        "tool_search_call" => {
            let execution = required_string(object, "execution", "tool_search_call item")?;
            if execution != "client" {
                return Ok(());
            }
            let call_id = required_string(object, "call_id", "tool_search_call item")?;
            let arguments = object
                .get("arguments")
                .context("tool_search_call item must contain `arguments`")?;
            push_assistant_tool_call(
                messages,
                ChatToolCall {
                    id: call_id,
                    kind: "function".to_owned(),
                    function: ChatFunctionCall {
                        name: "tool_search".to_owned(),
                        arguments: serde_json::to_string(arguments)?,
                    },
                },
            );
            Ok(())
        }
        "function_call_output" | "custom_tool_call_output" | "mcp_tool_call_output" => {
            let call_id = required_string(object, "call_id", "tool output item")?;
            let output = object
                .get("output")
                .context("tool output item must contain `output`")?;
            messages.push(ChatMessage {
                role: ChatRole::Tool,
                content: Some(tool_output_content(output)?),
                name: None,
                tool_call_id: Some(call_id),
                tool_calls: Vec::new(),
                reasoning_content: None,
            });
            Ok(())
        }
        "tool_search_output" => {
            let execution = required_string(object, "execution", "tool_search_output item")?;
            if execution != "client" {
                return Ok(());
            }
            let call_id = required_string(object, "call_id", "tool_search_output item")?;
            let tools = object
                .get("tools")
                .context("tool_search_output item must contain `tools`")?;
            messages.push(ChatMessage {
                role: ChatRole::Tool,
                content: Some(Value::String(serde_json::to_string(tools)?)),
                name: None,
                tool_call_id: Some(call_id),
                tool_calls: Vec::new(),
                reasoning_content: None,
            });
            Ok(())
        }
        "additional_tools" | "web_search_call" => Ok(()),
        "reasoning" => {
            let reasoning = reasoning_item_text(object)?;
            if !reasoning.is_empty() {
                push_assistant_reasoning(messages, reasoning);
            }
            Ok(())
        }
        unsupported => bail!("unsupported Responses input item type `{unsupported}`"),
    }
}

fn convert_agent_message_item(
    object: &Map<String, Value>,
    messages: &mut Vec<ChatMessage>,
) -> Result<()> {
    let author = required_string(object, "author", "agent_message item")?;
    let recipient = required_string(object, "recipient", "agent_message item")?;
    let content = object
        .get("content")
        .and_then(Value::as_array)
        .context("agent_message item must contain a `content` array")?;
    let mut text_parts = Vec::new();
    for part in content {
        let part = part
            .as_object()
            .context("agent_message content parts must be objects")?;
        let kind = required_string(part, "type", "agent_message content part")?;
        match kind.as_str() {
            "input_text" => text_parts.push(required_string(
                part,
                "text",
                "agent_message text content part",
            )?),
            "encrypted_content" => {
                bail!("encrypted agent_message content cannot be converted to Chat Completions")
            }
            unsupported => {
                bail!("unsupported agent_message content type `{unsupported}`")
            }
        }
    }
    if text_parts.is_empty() {
        return Ok(());
    }
    messages.push(ChatMessage::text(
        ChatRole::Assistant,
        format!(
            "Agent message from {author} to {recipient}:\n{}",
            text_parts.join("\n")
        ),
    ));
    Ok(())
}

fn convert_message_item(
    object: &Map<String, Value>,
    messages: &mut Vec<ChatMessage>,
) -> Result<()> {
    let role = match required_string(object, "role", "message item")?.as_str() {
        "system" => ChatRole::System,
        "developer" => ChatRole::Developer,
        "user" => ChatRole::User,
        "assistant" => ChatRole::Assistant,
        role => bail!("unsupported Responses message role `{role}`"),
    };
    let content = object
        .get("content")
        .context("Responses message item must contain `content`")?;
    let content = responses_content_to_chat(content)?;
    if role == ChatRole::Assistant {
        push_or_merge_assistant_content(messages, content);
    } else {
        messages.push(ChatMessage {
            role,
            content: Some(content),
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
            reasoning_content: None,
        });
    }
    Ok(())
}

fn responses_content_to_chat(content: &Value) -> Result<Value> {
    match content {
        Value::String(_) => Ok(content.clone()),
        Value::Array(parts) => {
            let mut converted = Vec::new();
            for part in parts {
                let object = part
                    .as_object()
                    .context("Responses message content parts must be objects")?;
                let kind = required_string(object, "type", "message content part")?;
                match kind.as_str() {
                    "input_text" | "output_text" | "text" => converted.push(json!({
                        "type": "text",
                        "text": required_string(object, "text", "text content part")?
                    })),
                    "input_image" => {
                        let url = object
                            .get("image_url")
                            .or_else(|| object.get("url"))
                            .context("input_image content part must contain `image_url`")?;
                        converted.push(json!({
                            "type": "image_url",
                            "image_url": { "url": value_as_string(url, "`image_url`")? }
                        }));
                    }
                    "refusal" => converted.push(json!({
                        "type": "text",
                        "text": required_string(object, "refusal", "refusal content part")?
                    })),
                    unsupported => {
                        bail!("unsupported Responses message content type `{unsupported}`")
                    }
                }
            }
            if converted.len() == 1 && converted[0].get("type") == Some(&json!("text")) {
                Ok(converted[0]["text"].clone())
            } else {
                Ok(Value::Array(converted))
            }
        }
        _ => bail!("Responses message `content` must be a string or an array"),
    }
}

fn convert_tools(tools: Option<&Value>, registry: &mut ToolRegistry) -> Result<Vec<ChatTool>> {
    let Some(tools) = tools else {
        return Ok(Vec::new());
    };
    let tools = tools
        .as_array()
        .context("Responses request `tools` must be an array")?;
    let mut converted = Vec::new();
    for tool in tools {
        let object = tool
            .as_object()
            .context("each Responses tool must be an object")?;
        let kind = required_string(object, "type", "Responses tool")?;
        match kind.as_str() {
            "function" => {
                let name = required_string(object, "name", "Responses function tool")?;
                let description = optional_tool_description(object)?;
                registry.insert(name.clone(), ResponsesToolKind::Function)?;
                converted.push(convert_function_tool(object, name, description)?);
            }
            "custom" => {
                let name = required_string(object, "name", "Responses custom tool")?;
                let description = optional_tool_description(object)?;
                registry.insert(name.clone(), ResponsesToolKind::Custom)?;
                converted.push(ChatTool::function(ChatFunctionDefinition {
                    name,
                    description,
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "input": {
                                "type": "string",
                                "description": "The complete raw input for this custom tool."
                            }
                        },
                        "required": ["input"],
                        "additionalProperties": false
                    }),
                    strict: None,
                }));
            }
            "namespace" => {
                let name = required_string(object, "name", "Responses namespace tool")?;
                let namespace_tools = object
                    .get("tools")
                    .and_then(Value::as_array)
                    .context("Responses namespace tool must contain a `tools` array")?;
                for namespace_tool in namespace_tools {
                    let namespace_tool = namespace_tool
                        .as_object()
                        .context("each Responses namespace child tool must be an object")?;
                    let child_kind =
                        required_string(namespace_tool, "type", "Responses namespace child tool")?;
                    if child_kind != "function" {
                        bail!("unsupported Responses namespace child tool type `{child_kind}`");
                    }
                    let child_name =
                        required_string(namespace_tool, "name", "Responses namespace child tool")?;
                    let child_description = namespace_tool
                        .get("description")
                        .map(|value| value_as_string(value, "namespace tool `description`"))
                        .transpose()?;
                    let flattened_name = registry.insert_namespaced(name.clone(), child_name)?;
                    converted.push(convert_function_tool(
                        namespace_tool,
                        flattened_name,
                        child_description,
                    )?);
                }
            }
            "tool_search" => {
                let execution = required_string(object, "execution", "Responses tool_search tool")?;
                if execution != "client" {
                    continue;
                }
                let name = "tool_search".to_owned();
                registry.insert(name.clone(), ResponsesToolKind::ToolSearch)?;
                converted.push(ChatTool::function(ChatFunctionDefinition {
                    name,
                    description: optional_tool_description(object)?,
                    parameters: object
                        .get("parameters")
                        .cloned()
                        .unwrap_or_else(|| json!({ "type": "object", "properties": {} })),
                    strict: None,
                }));
            }
            "web_search" => {}
            unsupported => bail!("unsupported Responses tool type `{unsupported}`"),
        }
    }
    Ok(converted)
}

fn optional_tool_description(object: &Map<String, Value>) -> Result<Option<String>> {
    object
        .get("description")
        .map(|value| value_as_string(value, "tool `description`"))
        .transpose()
}

fn convert_function_tool(
    object: &Map<String, Value>,
    name: String,
    description: Option<String>,
) -> Result<ChatTool> {
    let parameters = object
        .get("parameters")
        .cloned()
        .unwrap_or_else(|| json!({ "type": "object", "properties": {} }));
    Ok(ChatTool::function(ChatFunctionDefinition {
        name,
        description,
        parameters,
        strict: optional_bool(object, "strict")?,
    }))
}

fn convert_tool_choice(choice: Option<&Value>, registry: &ToolRegistry) -> Result<Option<Value>> {
    let Some(choice) = choice else {
        return Ok(None);
    };
    if choice.is_string() {
        return Ok(Some(choice.clone()));
    }
    let object = choice
        .as_object()
        .context("Responses `tool_choice` must be a string or object")?;
    let kind = required_string(object, "type", "`tool_choice`")?;
    match kind.as_str() {
        "function" | "custom" => {
            let name = required_string(object, "name", "`tool_choice`")?;
            let namespace = object
                .get("namespace")
                .map(|value| value_as_string(value, "`tool_choice.namespace`"))
                .transpose()?;
            let name = registry.chat_name(namespace.as_deref(), &name);
            if registry.kind(&name).is_none() {
                bail!("`tool_choice` refers to undeclared tool `{name}`");
            }
            Ok(Some(json!({
                "type": "function",
                "function": { "name": name }
            })))
        }
        "allowed_tools" => bail!("Responses `allowed_tools` tool choice is not supported"),
        unsupported => bail!("unsupported Responses tool choice type `{unsupported}`"),
    }
}

fn push_or_merge_assistant_content(messages: &mut Vec<ChatMessage>, content: Value) {
    if let Some(message) = messages
        .last_mut()
        .filter(|message| message.role == ChatRole::Assistant)
    {
        message.content = Some(merge_chat_content(message.content.take(), content));
    } else {
        messages.push(ChatMessage {
            role: ChatRole::Assistant,
            content: Some(content),
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
            reasoning_content: None,
        });
    }
}

fn push_assistant_reasoning(messages: &mut Vec<ChatMessage>, reasoning: String) {
    if let Some(message) = messages
        .last_mut()
        .filter(|message| message.role == ChatRole::Assistant)
    {
        message.reasoning_content = Some(merge_chat_content(
            message.reasoning_content.take(),
            json!(reasoning),
        ));
    } else {
        messages.push(ChatMessage {
            role: ChatRole::Assistant,
            content: None,
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
            reasoning_content: Some(json!(reasoning)),
        });
    }
}

fn push_assistant_tool_call(messages: &mut Vec<ChatMessage>, call: ChatToolCall) {
    if let Some(message) = messages
        .last_mut()
        .filter(|message| message.role == ChatRole::Assistant)
    {
        message.tool_calls.push(call);
    } else {
        messages.push(ChatMessage {
            role: ChatRole::Assistant,
            content: None,
            name: None,
            tool_call_id: None,
            tool_calls: vec![call],
            reasoning_content: None,
        });
    }
}

fn merge_chat_content(previous: Option<Value>, next: Value) -> Value {
    match (previous, next) {
        (None, next) => next,
        (Some(Value::String(mut left)), Value::String(right)) => {
            left.push_str(&right);
            Value::String(left)
        }
        (Some(Value::Array(mut left)), Value::Array(right)) => {
            left.extend(right);
            Value::Array(left)
        }
        (Some(left), right) => Value::Array(vec![left, right]),
    }
}

fn reasoning_item_text(object: &Map<String, Value>) -> Result<String> {
    let mut text = String::new();
    for key in ["summary", "content"] {
        let Some(parts) = object.get(key) else {
            continue;
        };
        for part in parts
            .as_array()
            .with_context(|| format!("reasoning item `{key}` must be an array"))?
        {
            let part = part
                .as_object()
                .context("reasoning content part must be an object")?;
            if let Some(value) = part.get("text") {
                text.push_str(&value_as_string(value, "reasoning text")?);
            }
        }
    }
    Ok(text)
}

fn tool_output_content(output: &Value) -> Result<Value> {
    match output {
        Value::String(_) => Ok(output.clone()),
        Value::Array(_) => responses_content_to_chat(output),
        Value::Object(_) => Ok(Value::String(serde_json::to_string(output)?)),
        _ => bail!("tool output must be a string, content array, or object"),
    }
}

pub fn chat_completion_to_responses(
    completion: &ChatCompletionResponse,
    requested_model: &str,
    tools: &ToolRegistry,
) -> Result<Value> {
    let response_id = responses_id(completion.id.as_deref());
    let model = completion.model.as_deref().unwrap_or(requested_model);
    let choice = completion
        .choices
        .iter()
        .find(|choice| choice.index == 0)
        .or_else(|| completion.choices.first())
        .context("Chat completion contained no choices")?;
    let mut output = Vec::new();
    append_reasoning_output(&mut output, &choice.message, &response_id);
    append_message_output(&mut output, &choice.message, &response_id);
    append_tool_outputs(&mut output, &choice.message, &response_id, tools)?;
    let status = finish_status(choice.finish_reason.as_deref());

    Ok(json!({
        "id": response_id,
        "object": "response",
        "created_at": unix_timestamp(),
        "status": status,
        "error": Value::Null,
        "incomplete_details": if status == "incomplete" {
            json!({ "reason": incomplete_reason(choice.finish_reason.as_deref()) })
        } else {
            Value::Null
        },
        "model": model,
        "output": output,
        "parallel_tool_calls": true,
        "tool_choice": "auto",
        "tools": [],
        "usage": usage_to_responses(completion.usage.as_ref())
    }))
}

fn append_reasoning_output(
    output: &mut Vec<Value>,
    message: &ChatAssistantMessage,
    response_id: &str,
) {
    let Some(text) = message
        .reasoning_content
        .as_ref()
        .and_then(value_text_lossy)
        .filter(|text| !text.is_empty())
    else {
        return;
    };
    output.push(json!({
        "id": format!("{response_id}_reasoning"),
        "type": "reasoning",
        "summary": [{ "type": "summary_text", "text": text }],
        "status": "completed"
    }));
}

fn append_message_output(
    output: &mut Vec<Value>,
    message: &ChatAssistantMessage,
    response_id: &str,
) {
    let mut content = Vec::new();
    if let Some(text) = message
        .content
        .as_ref()
        .and_then(value_text_lossy)
        .filter(|text| !text.is_empty())
    {
        content.push(json!({
            "type": "output_text",
            "text": text,
            "annotations": []
        }));
    }
    if let Some(refusal) = message
        .refusal
        .as_ref()
        .and_then(value_text_lossy)
        .filter(|text| !text.is_empty())
    {
        content.push(json!({ "type": "refusal", "refusal": refusal }));
    }
    if !content.is_empty() {
        output.push(json!({
            "id": format!("{response_id}_message"),
            "type": "message",
            "role": "assistant",
            "content": content,
            "status": "completed"
        }));
    }
}

fn append_tool_outputs(
    output: &mut Vec<Value>,
    message: &ChatAssistantMessage,
    response_id: &str,
    tools: &ToolRegistry,
) -> Result<()> {
    for (index, call) in message.tool_calls.iter().enumerate() {
        let item_id = format!("{response_id}_tool_{index}");
        let registered = tools.response_tool(&call.function.name);
        match registered
            .map(|tool| tool.kind)
            .unwrap_or(ResponsesToolKind::Function)
        {
            ResponsesToolKind::Function => output.push(response_function_call_item(
                item_id,
                call.id.clone(),
                &call.function.name,
                call.function.arguments.clone(),
                "completed",
                tools,
            )),
            ResponsesToolKind::Custom => output.push(json!({
                "id": item_id,
                "type": "custom_tool_call",
                "call_id": call.id,
                "name": call.function.name,
                "input": custom_input(&call.function.arguments),
                "status": "completed"
            })),
            ResponsesToolKind::ToolSearch => output.push(tool_search_call_item(
                item_id,
                call.id.clone(),
                &call.function.arguments,
                "completed",
            )),
        }
    }
    Ok(())
}

fn tool_search_call_item(item_id: String, call_id: String, arguments: &str, status: &str) -> Value {
    let arguments =
        serde_json::from_str(arguments).unwrap_or_else(|_| Value::String(arguments.to_owned()));
    json!({
        "id": item_id,
        "type": "tool_search_call",
        "call_id": call_id,
        "execution": "client",
        "arguments": arguments,
        "status": status
    })
}

fn response_function_call_item(
    item_id: String,
    call_id: String,
    chat_name: &str,
    arguments: String,
    status: &str,
    tools: &ToolRegistry,
) -> Value {
    let registered = tools.response_tool(chat_name);
    let name = registered
        .map(|tool| tool.original_name.as_str())
        .unwrap_or(chat_name);
    let mut item = json!({
        "id": item_id,
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": status
    });
    if let Some(namespace) = registered.and_then(|tool| tool.namespace.as_deref()) {
        item["namespace"] = json!(namespace);
    }
    item
}

fn custom_input(arguments: &str) -> String {
    serde_json::from_str::<Value>(arguments)
        .ok()
        .and_then(|value| {
            value
                .get("input")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .unwrap_or_else(|| arguments.to_owned())
}

fn finish_status(reason: Option<&str>) -> &'static str {
    match reason {
        Some("length") | Some("content_filter") => "incomplete",
        _ => "completed",
    }
}

fn incomplete_reason(reason: Option<&str>) -> &'static str {
    match reason {
        Some("content_filter") => "content_filter",
        _ => "max_output_tokens",
    }
}

fn usage_to_responses(usage: Option<&ChatUsage>) -> Value {
    let Some(usage) = usage else {
        return Value::Null;
    };
    let input = usage.prompt_tokens.unwrap_or(0);
    let output = usage.completion_tokens.unwrap_or(0);
    let total = usage.total_tokens.unwrap_or(input + output);
    let cached = usage
        .details
        .get("prompt_tokens_details")
        .and_then(|value| value.get("cached_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let reasoning = usage
        .details
        .get("completion_tokens_details")
        .and_then(|value| value.get("reasoning_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    json!({
        "input_tokens": input,
        "input_tokens_details": { "cached_tokens": cached },
        "output_tokens": output,
        "output_tokens_details": { "reasoning_tokens": reasoning },
        "total_tokens": total
    })
}

#[derive(Debug, Clone)]
struct StreamOutput {
    output_index: usize,
    item_id: String,
    text: String,
}

#[derive(Debug, Clone, Default)]
struct StreamTool {
    output_index: Option<usize>,
    item_id: Option<String>,
    call_id: String,
    name: String,
    arguments: String,
    started: bool,
}

#[derive(Debug, Clone)]
pub struct ResponsesStreamConverter {
    requested_model: String,
    response_id: String,
    tools: ToolRegistry,
    started: bool,
    completed: bool,
    next_output_index: usize,
    sequence_number: u64,
    text: Option<StreamOutput>,
    reasoning: Option<StreamOutput>,
    tool_calls: BTreeMap<usize, StreamTool>,
    usage: Option<ChatUsage>,
    finish_reason: Option<String>,
}

impl ResponsesStreamConverter {
    pub fn new(requested_model: impl Into<String>, tools: ToolRegistry) -> Self {
        Self::with_response_id(requested_model, tools, generated_id("resp"))
    }

    pub fn with_response_id(
        requested_model: impl Into<String>,
        tools: ToolRegistry,
        response_id: impl Into<String>,
    ) -> Self {
        Self {
            requested_model: requested_model.into(),
            response_id: response_id.into(),
            tools,
            started: false,
            completed: false,
            next_output_index: 0,
            sequence_number: 0,
            text: None,
            reasoning: None,
            tool_calls: BTreeMap::new(),
            usage: None,
            finish_reason: None,
        }
    }

    pub fn push(&mut self, item: ChatStreamItem) -> Result<Vec<Value>> {
        if self.completed {
            return Ok(Vec::new());
        }
        let mut events = Vec::new();
        self.ensure_started(&mut events);
        match item {
            ChatStreamItem::Chunk(chunk) => self.push_chunk(chunk, &mut events)?,
            ChatStreamItem::Done => self.complete(&mut events)?,
        }
        Ok(events)
    }

    pub fn finish(&mut self) -> Result<Vec<Value>> {
        if self.completed {
            return Ok(Vec::new());
        }
        let mut events = Vec::new();
        self.ensure_started(&mut events);
        self.complete(&mut events)?;
        Ok(events)
    }

    fn ensure_started(&mut self, events: &mut Vec<Value>) {
        if self.started {
            return;
        }
        self.started = true;
        let response = self.response_snapshot("in_progress", Vec::new(), Value::Null);
        self.emit(events, "response.created", json!({ "response": response }));
        let response = self.response_snapshot("in_progress", Vec::new(), Value::Null);
        self.emit(
            events,
            "response.in_progress",
            json!({ "response": response }),
        );
    }

    fn push_chunk(&mut self, chunk: ChatCompletionChunk, events: &mut Vec<Value>) -> Result<()> {
        if let Some(usage) = chunk.usage {
            self.usage = Some(usage);
        }
        let Some(choice) = chunk
            .choices
            .iter()
            .find(|choice| choice.index == 0)
            .or_else(|| chunk.choices.first())
        else {
            return Ok(());
        };
        if let Some(reason) = &choice.finish_reason {
            self.finish_reason = Some(reason.clone());
        }

        if let Some(delta) = choice
            .delta
            .reasoning_content
            .as_ref()
            .or(choice.delta.reasoning.as_ref())
            .and_then(value_text_lossy)
            .filter(|text| !text.is_empty())
        {
            self.push_reasoning_delta(delta, events);
        }
        if let Some(delta) = choice
            .delta
            .content
            .as_ref()
            .and_then(value_text_lossy)
            .filter(|text| !text.is_empty())
        {
            self.push_text_delta(delta, events);
        }
        for delta in &choice.delta.tool_calls {
            let state = self.tool_calls.entry(delta.index).or_default();
            if let Some(id) = &delta.id {
                state.call_id.push_str(id);
            }
            if let Some(function) = &delta.function {
                if let Some(name) = &function.name {
                    state.name.push_str(name);
                }
                if let Some(arguments) = &function.arguments {
                    state.arguments.push_str(arguments);
                }
            }
            self.start_tool_if_ready(delta.index, events);
            if let Some(arguments) = delta
                .function
                .as_ref()
                .and_then(|function| function.arguments.as_deref())
            {
                let state = &self.tool_calls[&delta.index];
                if state.started
                    && self.tools.kind(&state.name) == Some(ResponsesToolKind::Function)
                    && !arguments.is_empty()
                {
                    self.emit(
                        events,
                        "response.function_call_arguments.delta",
                        json!({
                            "item_id": state.item_id,
                            "output_index": state.output_index,
                            "delta": arguments
                        }),
                    );
                }
            }
        }
        Ok(())
    }

    fn push_reasoning_delta(&mut self, delta: String, events: &mut Vec<Value>) {
        if self.reasoning.is_none() {
            let output = self.new_output("reasoning");
            self.emit(
                events,
                "response.output_item.added",
                json!({
                    "output_index": output.output_index,
                    "item": {
                        "id": output.item_id,
                        "type": "reasoning",
                        "summary": [],
                        "status": "in_progress"
                    }
                }),
            );
            self.emit(
                events,
                "response.reasoning_summary_part.added",
                json!({
                    "item_id": output.item_id,
                    "output_index": output.output_index,
                    "summary_index": 0,
                    "part": { "type": "summary_text", "text": "" }
                }),
            );
            self.reasoning = Some(output);
        }
        let output = self.reasoning.as_mut().expect("reasoning initialized");
        output.text.push_str(&delta);
        let item_id = output.item_id.clone();
        let output_index = output.output_index;
        self.emit(
            events,
            "response.reasoning_summary_text.delta",
            json!({
                "item_id": item_id,
                "output_index": output_index,
                "summary_index": 0,
                "delta": delta
            }),
        );
    }

    fn push_text_delta(&mut self, delta: String, events: &mut Vec<Value>) {
        if self.text.is_none() {
            let output = self.new_output("message");
            self.emit(
                events,
                "response.output_item.added",
                json!({
                    "output_index": output.output_index,
                    "item": {
                        "id": output.item_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "status": "in_progress"
                    }
                }),
            );
            self.emit(
                events,
                "response.content_part.added",
                json!({
                    "item_id": output.item_id,
                    "output_index": output.output_index,
                    "content_index": 0,
                    "part": {
                        "type": "output_text",
                        "text": "",
                        "annotations": []
                    }
                }),
            );
            self.text = Some(output);
        }
        let output = self.text.as_mut().expect("text initialized");
        output.text.push_str(&delta);
        let item_id = output.item_id.clone();
        let output_index = output.output_index;
        self.emit(
            events,
            "response.output_text.delta",
            json!({
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "delta": delta
            }),
        );
    }

    fn start_tool_if_ready(&mut self, index: usize, events: &mut Vec<Value>) {
        self.start_tool(index, events, false);
    }

    fn start_tool(&mut self, index: usize, events: &mut Vec<Value>, allow_unknown: bool) {
        let ready = self.tool_calls.get(&index).is_some_and(|tool| {
            !tool.started
                && !tool.name.is_empty()
                && (allow_unknown
                    || (!tool.arguments.is_empty()
                        && self.tools.response_tool(&tool.name).is_some()))
        });
        if !ready {
            return;
        }
        let output_index = self.take_output_index();
        let tool = self.tool_calls.get_mut(&index).expect("tool exists");
        tool.started = true;
        tool.output_index = Some(output_index);
        tool.item_id = Some(format!("{}_tool_{index}", self.response_id));
        if tool.call_id.is_empty() {
            tool.call_id = format!("call_{}_{}", self.response_id, index);
        }
        let item_id = tool.item_id.clone().expect("tool item ID initialized");
        let item = match self.tools.kind(&tool.name) {
            Some(ResponsesToolKind::Custom) => json!({
                "id": item_id,
                "type": "custom_tool_call",
                "call_id": tool.call_id,
                "name": tool.name,
                "input": "",
                "status": "in_progress"
            }),
            Some(ResponsesToolKind::ToolSearch) => {
                tool_search_call_item(item_id, tool.call_id.clone(), "{}", "in_progress")
            }
            _ => response_function_call_item(
                item_id,
                tool.call_id.clone(),
                &tool.name,
                String::new(),
                "in_progress",
                &self.tools,
            ),
        };
        self.emit(
            events,
            "response.output_item.added",
            json!({ "output_index": output_index, "item": item }),
        );
    }

    fn complete(&mut self, events: &mut Vec<Value>) -> Result<()> {
        let mut final_output = Vec::new();
        let tool_indices = self.tool_calls.keys().copied().collect::<Vec<_>>();
        for index in tool_indices {
            self.start_tool(index, events, true);
        }

        if let Some(reasoning) = self.reasoning.take() {
            self.emit(
                events,
                "response.reasoning_summary_text.done",
                json!({
                    "item_id": reasoning.item_id,
                    "output_index": reasoning.output_index,
                    "summary_index": 0,
                    "text": reasoning.text
                }),
            );
            self.emit(
                events,
                "response.reasoning_summary_part.done",
                json!({
                    "item_id": reasoning.item_id,
                    "output_index": reasoning.output_index,
                    "summary_index": 0,
                    "part": { "type": "summary_text", "text": reasoning.text }
                }),
            );
            let item = json!({
                "id": reasoning.item_id,
                "type": "reasoning",
                "summary": [{ "type": "summary_text", "text": reasoning.text }],
                "status": "completed"
            });
            self.emit(
                events,
                "response.output_item.done",
                json!({
                    "output_index": reasoning.output_index,
                    "item": item
                }),
            );
            final_output.push((reasoning.output_index, item));
        }
        if let Some(text) = self.text.take() {
            self.emit(
                events,
                "response.output_text.done",
                json!({
                    "item_id": text.item_id,
                    "output_index": text.output_index,
                    "content_index": 0,
                    "text": text.text
                }),
            );
            self.emit(
                events,
                "response.content_part.done",
                json!({
                    "item_id": text.item_id,
                    "output_index": text.output_index,
                    "content_index": 0,
                    "part": {
                        "type": "output_text",
                        "text": text.text,
                        "annotations": []
                    }
                }),
            );
            let item = json!({
                "id": text.item_id,
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": text.text,
                    "annotations": []
                }],
                "status": "completed"
            });
            self.emit(
                events,
                "response.output_item.done",
                json!({
                    "output_index": text.output_index,
                    "item": item
                }),
            );
            final_output.push((text.output_index, item));
        }

        let tools = std::mem::take(&mut self.tool_calls);
        for (_, tool) in tools {
            if !tool.started {
                bail!("Chat stream ended before a tool call name was received");
            }
            let output_index = tool
                .output_index
                .context("started tool lacks output index")?;
            let item_id = tool.item_id.context("started tool lacks item ID")?;
            match self.tools.kind(&tool.name) {
                Some(ResponsesToolKind::Custom) => {
                    let input = custom_input(&tool.arguments);
                    if !input.is_empty() {
                        self.emit(
                            events,
                            "response.custom_tool_call_input.delta",
                            json!({
                                "item_id": item_id,
                                "output_index": output_index,
                                "delta": input
                            }),
                        );
                    }
                    self.emit(
                        events,
                        "response.custom_tool_call_input.done",
                        json!({
                            "item_id": item_id,
                            "output_index": output_index,
                            "input": input
                        }),
                    );
                    let item = json!({
                        "id": item_id,
                        "type": "custom_tool_call",
                        "call_id": tool.call_id,
                        "name": tool.name,
                        "input": input,
                        "status": "completed"
                    });
                    self.emit(
                        events,
                        "response.output_item.done",
                        json!({
                            "output_index": output_index,
                            "item": item
                        }),
                    );
                    final_output.push((output_index, item));
                }
                Some(ResponsesToolKind::ToolSearch) => {
                    let item =
                        tool_search_call_item(item_id, tool.call_id, &tool.arguments, "completed");
                    self.emit(
                        events,
                        "response.output_item.done",
                        json!({
                            "output_index": output_index,
                            "item": item
                        }),
                    );
                    final_output.push((output_index, item));
                }
                _ => {
                    self.emit(
                        events,
                        "response.function_call_arguments.done",
                        json!({
                            "item_id": item_id,
                            "output_index": output_index,
                            "arguments": tool.arguments
                        }),
                    );
                    let item = response_function_call_item(
                        item_id.clone(),
                        tool.call_id,
                        &tool.name,
                        tool.arguments,
                        "completed",
                        &self.tools,
                    );
                    self.emit(
                        events,
                        "response.output_item.done",
                        json!({
                            "output_index": output_index,
                            "item": item
                        }),
                    );
                    final_output.push((output_index, item));
                }
            }
        }

        let status = finish_status(self.finish_reason.as_deref());
        let usage = usage_to_responses(self.usage.as_ref());
        final_output.sort_by_key(|(index, _)| *index);
        let final_output = final_output
            .into_iter()
            .map(|(_, item)| item)
            .collect::<Vec<_>>();
        let response = self.response_snapshot(status, final_output, usage);
        let event_type = if status == "completed" {
            "response.completed"
        } else {
            "response.incomplete"
        };
        self.emit(events, event_type, json!({ "response": response }));
        self.completed = true;
        Ok(())
    }

    fn response_snapshot(&self, status: &str, output: Vec<Value>, usage: Value) -> Value {
        json!({
            "id": self.response_id,
            "object": "response",
            "created_at": unix_timestamp(),
            "status": status,
            "error": Value::Null,
            "incomplete_details": if status == "incomplete" {
                json!({ "reason": incomplete_reason(self.finish_reason.as_deref()) })
            } else {
                Value::Null
            },
            "model": self.requested_model,
            "output": output,
            "parallel_tool_calls": true,
            "tool_choice": "auto",
            "tools": [],
            "usage": usage
        })
    }

    fn new_output(&mut self, suffix: &str) -> StreamOutput {
        let output_index = self.take_output_index();
        StreamOutput {
            output_index,
            item_id: format!("{}_{suffix}", self.response_id),
            text: String::new(),
        }
    }

    fn take_output_index(&mut self) -> usize {
        let index = self.next_output_index;
        self.next_output_index += 1;
        index
    }

    fn emit(&mut self, events: &mut Vec<Value>, kind: &str, payload: Value) {
        self.sequence_number += 1;
        let mut object = payload.as_object().cloned().unwrap_or_default();
        object.insert("type".to_owned(), Value::String(kind.to_owned()));
        object.insert(
            "sequence_number".to_owned(),
            Value::Number(self.sequence_number.into()),
        );
        events.push(Value::Object(object));
    }
}

pub fn responses_sse_event(event: &Value) -> Result<String> {
    let event_type = event
        .get("type")
        .and_then(Value::as_str)
        .context("Responses SSE event must contain a string `type`")?;
    Ok(format!(
        "event: {event_type}\ndata: {}\n\n",
        serde_json::to_string(event)?
    ))
}

fn required_string(object: &Map<String, Value>, key: &str, context: &str) -> Result<String> {
    object
        .get(key)
        .with_context(|| format!("{context} must contain `{key}`"))
        .and_then(|value| value_as_string(value, &format!("{context} `{key}`")))
}

fn value_as_string(value: &Value, name: &str) -> Result<String> {
    value
        .as_str()
        .map(str::to_owned)
        .with_context(|| format!("{name} must be a string"))
}

fn value_as_text(value: &Value, name: &str) -> Result<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                let object = part
                    .as_object()
                    .with_context(|| format!("{name} parts must be objects"))?;
                let part_type = required_string(object, "type", name)?;
                if !matches!(part_type.as_str(), "input_text" | "output_text" | "text") {
                    bail!("unsupported {name} content type `{part_type}`");
                }
                text.push_str(&required_string(object, "text", name)?);
            }
            Ok(text)
        }
        _ => bail!("{name} must be a string or text content array"),
    }
}

fn value_text_lossy(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                if let Some(value) = part
                    .get("text")
                    .or_else(|| part.get("content"))
                    .and_then(Value::as_str)
                {
                    text.push_str(value);
                }
            }
            Some(text)
        }
        _ => None,
    }
}

fn optional_bool(object: &Map<String, Value>, key: &str) -> Result<Option<bool>> {
    object
        .get(key)
        .map(|value| {
            value
                .as_bool()
                .with_context(|| format!("`{key}` must be a boolean"))
        })
        .transpose()
}

fn optional_u64(object: &Map<String, Value>, key: &str) -> Result<Option<u64>> {
    object
        .get(key)
        .map(|value| {
            value
                .as_u64()
                .with_context(|| format!("`{key}` must be a non-negative integer"))
        })
        .transpose()
}

fn optional_f64(object: &Map<String, Value>, key: &str) -> Result<Option<f64>> {
    object
        .get(key)
        .map(|value| {
            value
                .as_f64()
                .with_context(|| format!("`{key}` must be a number"))
        })
        .transpose()
}

fn responses_id(chat_id: Option<&str>) -> String {
    match chat_id {
        Some(id) if id.starts_with("resp_") => id.to_owned(),
        Some(id) if !id.is_empty() => format!("resp_{id}"),
        _ => generated_id("resp"),
    }
}

fn generated_id(prefix: &str) -> String {
    let sequence = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}_{:x}_{sequence:x}", unix_timestamp())
}

fn unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::{
        ChatChoice, ChatChunkChoice, ChatDelta, ChatFunctionCallDelta, ChatToolCallDelta,
    };

    #[test]
    fn converts_function_and_custom_tools_and_multiturn_history() {
        let converted = responses_request_to_chat(&json!({
            "model": "glm-test",
            "stream": true,
            "instructions": "Be precise.",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{ "type": "input_text", "text": "edit it" }]
                },
                {
                    "type": "reasoning",
                    "summary": [{ "type": "summary_text", "text": "Need a patch." }]
                },
                {
                    "type": "function_call",
                    "call_id": "call_shell",
                    "name": "shell",
                    "arguments": "{\"cmd\":\"pwd\"}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_shell",
                    "output": "/tmp"
                },
                {
                    "type": "custom_tool_call",
                    "call_id": "call_patch",
                    "name": "apply_patch",
                    "input": "*** Begin Patch"
                },
                {
                    "type": "custom_tool_call_output",
                    "call_id": "call_patch",
                    "output": "Done!"
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "shell",
                    "description": "Run a command",
                    "parameters": {
                        "type": "object",
                        "properties": { "cmd": { "type": "string" } },
                        "required": ["cmd"]
                    }
                },
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "Apply a patch"
                }
            ],
            "tool_choice": { "type": "custom", "name": "apply_patch" },
            "parallel_tool_calls": false,
            "reasoning": { "effort": "high" },
            "max_output_tokens": 2048
        }))
        .unwrap();

        assert_eq!(
            converted.tools.kind("shell"),
            Some(ResponsesToolKind::Function)
        );
        assert_eq!(
            converted.tools.kind("apply_patch"),
            Some(ResponsesToolKind::Custom)
        );
        assert_eq!(converted.request.messages.len(), 6);
        assert_eq!(converted.request.messages[0].role, ChatRole::System);
        assert_eq!(
            converted.request.messages[2].reasoning_content,
            Some(json!("Need a patch."))
        );
        assert_eq!(
            converted.request.messages[2].tool_calls[0].function.name,
            "shell"
        );
        assert_eq!(
            converted.request.messages[4].tool_calls[0]
                .function
                .arguments,
            "{\"input\":\"*** Begin Patch\"}"
        );
        assert_eq!(
            converted.request.tool_choice,
            Some(json!({
                "type": "function",
                "function": { "name": "apply_patch" }
            }))
        );
        assert_eq!(converted.request.reasoning_effort.as_deref(), Some("high"));
        assert_eq!(converted.request.max_completion_tokens, Some(2048));
        assert_eq!(
            converted.request.stream_options,
            Some(crate::chat::ChatStreamOptions {
                include_usage: true
            })
        );
    }

    #[test]
    fn omits_none_reasoning_effort_and_preserves_supported_levels() {
        let none = responses_request_to_chat(&json!({
            "model": "glm-test",
            "input": "hello",
            "reasoning": { "effort": "none" }
        }))
        .unwrap();
        assert_eq!(none.request.reasoning_effort, None);

        let high = responses_request_to_chat(&json!({
            "model": "glm-test",
            "input": "hello",
            "reasoning": { "effort": "high" }
        }))
        .unwrap();
        assert_eq!(high.request.reasoning_effort.as_deref(), Some("high"));
    }

    #[test]
    fn converts_plaintext_agent_messages_to_assistant_messages() {
        let converted = responses_request_to_chat(&json!({
            "model": "glm-test",
            "input": [{
                "type": "agent_message",
                "author": "worker",
                "recipient": "root",
                "content": [
                    { "type": "input_text", "text": "Inspection complete." },
                    { "type": "input_text", "text": "No issues found." }
                ]
            }]
        }))
        .unwrap();

        assert_eq!(converted.request.messages.len(), 1);
        assert_eq!(converted.request.messages[0].role, ChatRole::Assistant);
        assert_eq!(
            converted.request.messages[0].content,
            Some(json!(
                "Agent message from worker to root:\nInspection complete.\nNo issues found."
            ))
        );
    }

    #[test]
    fn rejects_encrypted_agent_message_content() {
        let error = responses_request_to_chat(&json!({
            "model": "glm-test",
            "input": [{
                "type": "agent_message",
                "author": "worker",
                "recipient": "root",
                "content": [{
                    "type": "encrypted_content",
                    "encrypted_content": "opaque"
                }]
            }]
        }))
        .unwrap_err();

        assert!(error
            .to_string()
            .contains("encrypted agent_message content cannot be converted"));
    }

    #[test]
    fn flattens_namespace_tools_history_and_tool_choice() {
        let converted = responses_request_to_chat(&json!({
            "model": "glm-test",
            "input": [
                {
                    "type": "function_call",
                    "namespace": "collaboration",
                    "name": "spawn_agent",
                    "call_id": "call_1",
                    "arguments": "{\"task\":\"inspect\"}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "done"
                }
            ],
            "tools": [{
                "type": "namespace",
                "name": "collaboration",
                "description": "Agent collaboration tools",
                "tools": [{
                    "type": "function",
                    "name": "spawn_agent",
                    "description": "Spawn an agent",
                    "parameters": {
                        "type": "object",
                        "properties": { "task": { "type": "string" } },
                        "required": ["task"]
                    }
                }]
            }],
            "tool_choice": {
                "type": "function",
                "namespace": "collaboration",
                "name": "spawn_agent"
            }
        }))
        .unwrap();

        assert_eq!(converted.request.tools.len(), 1);
        assert_eq!(
            converted.request.tools[0].function.name,
            "collaboration__spawn_agent"
        );
        assert_eq!(
            converted.request.messages[0].tool_calls[0].function.name,
            "collaboration__spawn_agent"
        );
        assert_eq!(
            converted.request.tool_choice,
            Some(json!({
                "type": "function",
                "function": { "name": "collaboration__spawn_agent" }
            }))
        );
    }

    #[test]
    fn preserves_mcp_namespace_separator_when_flattening() {
        let converted = responses_request_to_chat(&json!({
            "model": "glm-test",
            "input": "search",
            "tools": [{
                "type": "namespace",
                "name": "mcp__open_websearch__",
                "tools": [{
                    "type": "function",
                    "name": "search",
                    "parameters": { "type": "object", "properties": {} }
                }]
            }]
        }))
        .unwrap();

        assert_eq!(
            converted.request.tools[0].function.name,
            "mcp__open_websearch__search"
        );
    }

    #[test]
    fn converts_client_tool_search_and_omits_hosted_web_search() {
        let converted = responses_request_to_chat(&json!({
            "model": "glm-test",
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
        .unwrap();

        assert_eq!(converted.request.tools.len(), 1);
        assert_eq!(converted.request.tools[0].function.name, "tool_search");
        assert_eq!(
            converted.tools.kind("tool_search"),
            Some(ResponsesToolKind::ToolSearch)
        );
    }

    #[test]
    fn converts_additional_tools_and_tool_search_history() {
        let converted = responses_request_to_chat(&json!({
            "model": "glm-test",
            "input": [
                {
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": [{
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
                    }]
                },
                {
                    "type": "tool_search_call",
                    "call_id": "search-1",
                    "execution": "client",
                    "arguments": {
                        "query": "calendar create",
                        "limit": 1
                    }
                },
                {
                    "type": "tool_search_output",
                    "call_id": "search-1",
                    "status": "completed",
                    "execution": "client",
                    "tools": [{
                        "type": "namespace",
                        "name": "calendar",
                        "tools": [{
                            "type": "function",
                            "name": "create_event",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "title": { "type": "string" }
                                },
                                "required": ["title"]
                            }
                        }]
                    }]
                },
                {
                    "type": "web_search_call",
                    "id": "ws_1",
                    "status": "completed"
                }
            ]
        }))
        .unwrap();

        assert_eq!(
            converted
                .request
                .tools
                .iter()
                .map(|tool| tool.function.name.as_str())
                .collect::<Vec<_>>(),
            ["tool_search", "calendar__create_event"]
        );
        assert_eq!(converted.request.messages.len(), 2);
        assert_eq!(
            converted.request.messages[0].tool_calls[0].function.name,
            "tool_search"
        );
        assert_eq!(
            converted.request.messages[0].tool_calls[0]
                .function
                .arguments,
            "{\"limit\":1,\"query\":\"calendar create\"}"
        );
        assert_eq!(converted.request.messages[1].role, ChatRole::Tool);
        assert_eq!(
            converted.request.messages[1].tool_call_id.as_deref(),
            Some("search-1")
        );
    }

    #[test]
    fn rejects_collisions_between_plain_and_flattened_namespace_tools() {
        let error = responses_request_to_chat(&json!({
            "model": "glm-test",
            "input": "hello",
            "tools": [
                {
                    "type": "function",
                    "name": "collaboration__spawn_agent",
                    "parameters": { "type": "object", "properties": {} }
                },
                {
                    "type": "namespace",
                    "name": "collaboration",
                    "tools": [{
                        "type": "function",
                        "name": "spawn_agent",
                        "parameters": { "type": "object", "properties": {} }
                    }]
                }
            ]
        }))
        .unwrap_err();

        assert!(error
            .to_string()
            .contains("tool name collision after namespace flattening"));
    }

    #[test]
    fn rejects_unsupported_items_instead_of_dropping_them() {
        let error = responses_request_to_chat(&json!({
            "model": "model-a",
            "input": [{ "type": "computer_call", "id": "1" }]
        }))
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("unsupported Responses input item type `computer_call`"));
    }

    #[test]
    fn converts_non_streaming_text_reasoning_usage_and_custom_call() {
        let mut tools = ToolRegistry::default();
        tools
            .insert("apply_patch".to_owned(), ResponsesToolKind::Custom)
            .unwrap();
        let completion = ChatCompletionResponse {
            id: Some("chatcmpl-1".to_owned()),
            model: Some("glm-test".to_owned()),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatAssistantMessage {
                    role: Some(ChatRole::Assistant),
                    content: Some(json!("I will edit it.")),
                    reasoning_content: Some(json!("A patch is needed.")),
                    refusal: None,
                    tool_calls: vec![ChatToolCall {
                        id: "call_1".to_owned(),
                        kind: "function".to_owned(),
                        function: ChatFunctionCall {
                            name: "apply_patch".to_owned(),
                            arguments: "{\"input\":\"*** Begin Patch\"}".to_owned(),
                        },
                    }],
                },
                finish_reason: Some("tool_calls".to_owned()),
            }],
            usage: Some(ChatUsage {
                prompt_tokens: Some(10),
                completion_tokens: Some(4),
                total_tokens: Some(14),
                details: BTreeMap::from([
                    (
                        "prompt_tokens_details".to_owned(),
                        json!({ "cached_tokens": 3 }),
                    ),
                    (
                        "completion_tokens_details".to_owned(),
                        json!({ "reasoning_tokens": 2 }),
                    ),
                ]),
            }),
        };

        let response =
            chat_completion_to_responses(&completion, "requested-model", &tools).unwrap();
        assert_eq!(response["id"], "resp_chatcmpl-1");
        assert_eq!(response["status"], "completed");
        assert_eq!(response["output"][0]["type"], "reasoning");
        assert_eq!(response["output"][1]["type"], "message");
        assert_eq!(response["output"][2]["type"], "custom_tool_call");
        assert_eq!(response["output"][2]["input"], "*** Begin Patch");
        assert_eq!(response["usage"]["input_tokens"], 10);
        assert_eq!(
            response["usage"]["input_tokens_details"]["cached_tokens"],
            3
        );
        assert_eq!(
            response["usage"]["output_tokens_details"]["reasoning_tokens"],
            2
        );
    }

    #[test]
    fn custom_tool_malformed_wrapper_preserves_raw_arguments() {
        assert_eq!(custom_input("raw patch data"), "raw patch data");
        assert_eq!(custom_input("{\"wrong\":1}"), "{\"wrong\":1}");
    }

    #[test]
    fn restores_namespace_in_non_streaming_function_calls() {
        let mut tools = ToolRegistry::default();
        tools
            .insert_namespaced("collaboration".to_owned(), "spawn_agent".to_owned())
            .unwrap();
        let completion = ChatCompletionResponse {
            id: Some("chatcmpl-namespace".to_owned()),
            model: Some("glm-test".to_owned()),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatAssistantMessage {
                    role: Some(ChatRole::Assistant),
                    content: None,
                    reasoning_content: None,
                    refusal: None,
                    tool_calls: vec![ChatToolCall {
                        id: "call_1".to_owned(),
                        kind: "function".to_owned(),
                        function: ChatFunctionCall {
                            name: "collaboration__spawn_agent".to_owned(),
                            arguments: "{\"task\":\"inspect\"}".to_owned(),
                        },
                    }],
                },
                finish_reason: Some("tool_calls".to_owned()),
            }],
            usage: None,
        };

        let response =
            chat_completion_to_responses(&completion, "requested-model", &tools).unwrap();
        assert_eq!(response["output"][0]["type"], "function_call");
        assert_eq!(response["output"][0]["namespace"], "collaboration");
        assert_eq!(response["output"][0]["name"], "spawn_agent");
    }

    #[test]
    fn restores_non_streaming_tool_search_calls() {
        let mut tools = ToolRegistry::default();
        tools
            .insert("tool_search".to_owned(), ResponsesToolKind::ToolSearch)
            .unwrap();
        let completion = ChatCompletionResponse {
            id: Some("chatcmpl-tool-search".to_owned()),
            model: Some("glm-test".to_owned()),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatAssistantMessage {
                    role: Some(ChatRole::Assistant),
                    content: None,
                    reasoning_content: None,
                    refusal: None,
                    tool_calls: vec![ChatToolCall {
                        id: "search-1".to_owned(),
                        kind: "function".to_owned(),
                        function: ChatFunctionCall {
                            name: "tool_search".to_owned(),
                            arguments: "{\"query\":\"calendar\",\"limit\":2}".to_owned(),
                        },
                    }],
                },
                finish_reason: Some("tool_calls".to_owned()),
            }],
            usage: None,
        };

        let response =
            chat_completion_to_responses(&completion, "requested-model", &tools).unwrap();
        assert_eq!(response["output"][0]["type"], "tool_search_call");
        assert_eq!(response["output"][0]["call_id"], "search-1");
        assert_eq!(response["output"][0]["execution"], "client");
        assert_eq!(
            response["output"][0]["arguments"],
            json!({ "query": "calendar", "limit": 2 })
        );
    }

    #[test]
    fn streaming_conversion_handles_fragmented_text_reasoning_and_tools() {
        let mut tools = ToolRegistry::default();
        tools
            .insert("apply_patch".to_owned(), ResponsesToolKind::Custom)
            .unwrap();
        tools
            .insert("shell".to_owned(), ResponsesToolKind::Function)
            .unwrap();
        let mut converter =
            ResponsesStreamConverter::with_response_id("glm-test", tools, "resp_test");
        let chunks = vec![
            ChatStreamItem::Chunk(chunk(ChatDelta {
                reasoning_content: Some(json!("think ")),
                content: Some(json!("hello ")),
                tool_calls: vec![
                    tool_delta(
                        0,
                        Some("call_patch"),
                        Some("apply_patch"),
                        Some("{\"input\":\"*** "),
                    ),
                    tool_delta(1, Some("call_shell"), Some("shell"), Some("{\"cmd\":")),
                ],
                ..ChatDelta::default()
            })),
            ChatStreamItem::Chunk(chunk(ChatDelta {
                reasoning_content: Some(json!("more")),
                content: Some(json!("world")),
                tool_calls: vec![
                    tool_delta(0, None, None, Some("Begin Patch\"}")),
                    tool_delta(1, None, None, Some("\"pwd\"}")),
                ],
                ..ChatDelta::default()
            })),
            ChatStreamItem::Chunk(ChatCompletionChunk {
                id: Some("chat-1".to_owned()),
                model: Some("glm-test".to_owned()),
                choices: vec![ChatChunkChoice {
                    index: 0,
                    delta: ChatDelta::default(),
                    finish_reason: Some("tool_calls".to_owned()),
                }],
                usage: Some(ChatUsage {
                    prompt_tokens: Some(5),
                    completion_tokens: Some(7),
                    total_tokens: Some(12),
                    details: BTreeMap::new(),
                }),
            }),
            ChatStreamItem::Done,
        ];

        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(converter.push(chunk).unwrap());
        }
        let types = events
            .iter()
            .map(|event| event["type"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(types[0], "response.created");
        assert_eq!(types[1], "response.in_progress");
        assert!(types.contains(&"response.output_text.delta"));
        assert!(types.contains(&"response.reasoning_summary_text.delta"));
        assert!(types.contains(&"response.function_call_arguments.delta"));
        assert!(types.contains(&"response.custom_tool_call_input.delta"));
        assert_eq!(types.last(), Some(&"response.completed"));

        let custom_done = events
            .iter()
            .find(|event| {
                event["type"] == "response.output_item.done"
                    && event["item"]["type"] == "custom_tool_call"
            })
            .unwrap();
        assert_eq!(custom_done["item"]["input"], "*** Begin Patch");
        let function_done = events
            .iter()
            .find(|event| {
                event["type"] == "response.output_item.done"
                    && event["item"]["type"] == "function_call"
            })
            .unwrap();
        assert_eq!(function_done["item"]["arguments"], "{\"cmd\":\"pwd\"}");
        let completed = events.last().unwrap();
        assert_eq!(completed["response"]["usage"]["total_tokens"], 12);
        assert_eq!(completed["response"]["output"].as_array().unwrap().len(), 4);
        let response_ids = events
            .iter()
            .filter_map(|event| event.get("response"))
            .filter_map(|response| response.get("id"))
            .collect::<Vec<_>>();
        assert!(response_ids.iter().all(|id| **id == json!("resp_test")));
        assert!(events
            .windows(2)
            .all(|pair| pair[0]["sequence_number"].as_u64().unwrap() + 1
                == pair[1]["sequence_number"].as_u64().unwrap()));
    }

    #[test]
    fn streaming_conversion_restores_fragmented_namespace_tool_name() {
        let mut tools = ToolRegistry::default();
        tools
            .insert_namespaced("collaboration".to_owned(), "spawn_agent".to_owned())
            .unwrap();
        let mut converter =
            ResponsesStreamConverter::with_response_id("glm-test", tools, "resp_namespace");
        let chunks = [
            ChatStreamItem::Chunk(chunk(ChatDelta {
                tool_calls: vec![tool_delta(0, Some("call_1"), Some("collaboration__"), None)],
                ..ChatDelta::default()
            })),
            ChatStreamItem::Chunk(chunk(ChatDelta {
                tool_calls: vec![tool_delta(
                    0,
                    None,
                    Some("spawn_agent"),
                    Some("{\"task\":\"inspect\"}"),
                )],
                ..ChatDelta::default()
            })),
            ChatStreamItem::Done,
        ];

        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(converter.push(chunk).unwrap());
        }

        let added = events
            .iter()
            .find(|event| {
                event["type"] == "response.output_item.added"
                    && event["item"]["type"] == "function_call"
            })
            .unwrap();
        assert_eq!(added["item"]["namespace"], "collaboration");
        assert_eq!(added["item"]["name"], "spawn_agent");

        let done = events
            .iter()
            .find(|event| {
                event["type"] == "response.output_item.done"
                    && event["item"]["type"] == "function_call"
            })
            .unwrap();
        assert_eq!(done["item"]["namespace"], "collaboration");
        assert_eq!(done["item"]["name"], "spawn_agent");

        let completed = events
            .iter()
            .find(|event| event["type"] == "response.completed")
            .unwrap();
        assert_eq!(
            completed["response"]["output"][0]["namespace"],
            "collaboration"
        );
        assert_eq!(completed["response"]["output"][0]["name"], "spawn_agent");
    }

    #[test]
    fn streaming_conversion_restores_tool_search_calls() {
        let mut tools = ToolRegistry::default();
        tools
            .insert("tool_search".to_owned(), ResponsesToolKind::ToolSearch)
            .unwrap();
        let mut converter =
            ResponsesStreamConverter::with_response_id("glm-test", tools, "resp_tool_search");
        let chunks = [
            ChatStreamItem::Chunk(chunk(ChatDelta {
                tool_calls: vec![tool_delta(
                    0,
                    Some("search-1"),
                    Some("tool_search"),
                    Some("{\"query\":\"cal"),
                )],
                ..ChatDelta::default()
            })),
            ChatStreamItem::Chunk(chunk(ChatDelta {
                tool_calls: vec![tool_delta(0, None, None, Some("endar\",\"limit\":2}"))],
                ..ChatDelta::default()
            })),
            ChatStreamItem::Done,
        ];

        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(converter.push(chunk).unwrap());
        }

        assert!(!events
            .iter()
            .any(|event| event["type"] == "response.function_call_arguments.delta"));
        let done = events
            .iter()
            .find(|event| {
                event["type"] == "response.output_item.done"
                    && event["item"]["type"] == "tool_search_call"
            })
            .unwrap();
        assert_eq!(done["item"]["call_id"], "search-1");
        assert_eq!(done["item"]["execution"], "client");
        assert_eq!(
            done["item"]["arguments"],
            json!({ "query": "calendar", "limit": 2 })
        );
    }

    #[test]
    fn sse_serialization_includes_event_name_and_json() {
        let event = json!({
            "type": "response.output_text.delta",
            "sequence_number": 3,
            "delta": "hello"
        });
        let serialized = responses_sse_event(&event).unwrap();
        assert!(serialized.starts_with("event: response.output_text.delta\ndata: {"));
        assert!(serialized.ends_with("\n\n"));
    }

    fn chunk(delta: ChatDelta) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: Some("chat-1".to_owned()),
            model: Some("glm-test".to_owned()),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta,
                finish_reason: None,
            }],
            usage: None,
        }
    }

    fn tool_delta(
        index: usize,
        id: Option<&str>,
        name: Option<&str>,
        arguments: Option<&str>,
    ) -> ChatToolCallDelta {
        ChatToolCallDelta {
            index,
            id: id.map(str::to_owned),
            kind: Some("function".to_owned()),
            function: Some(ChatFunctionCallDelta {
                name: name.map(str::to_owned),
                arguments: arguments.map(str::to_owned),
            }),
        }
    }
}
