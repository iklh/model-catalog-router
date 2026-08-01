use crate::chat::{ChatFunctionDefinition, ChatTool};
use anyhow::{Context, Result};
use futures_util::future::BoxFuture;
use serde_json::{json, Value};

pub const INTERNAL_WEB_SEARCH_TOOL_NAME: &str = "mc_router__web_search";
pub const MAX_WEB_SEARCH_ROUNDS: usize = 8;

#[derive(Debug, Clone, PartialEq)]
pub struct WebSearchRequest {
    pub query: String,
}

impl WebSearchRequest {
    pub fn from_arguments(arguments: &str) -> Result<Self> {
        let value: Value =
            serde_json::from_str(arguments).context("Web Search arguments must be valid JSON")?;
        let query = value
            .get("query")
            .and_then(Value::as_str)
            .filter(|query| !query.trim().is_empty())
            .context("Web Search arguments must contain a non-empty string `query`")?;
        Ok(Self {
            query: query.to_owned(),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct WebSearchResult {
    pub output: Value,
}

impl WebSearchResult {
    pub fn into_tool_content(self) -> Result<Value> {
        match self.output {
            Value::String(_) => Ok(self.output),
            output => Ok(Value::String(serde_json::to_string(&output)?)),
        }
    }
}

pub trait WebSearchBackend: Send + Sync {
    fn search(&self, request: WebSearchRequest) -> BoxFuture<'static, Result<WebSearchResult>>;
}

pub fn chat_tool() -> ChatTool {
    ChatTool::function(ChatFunctionDefinition {
        name: INTERNAL_WEB_SEARCH_TOOL_NAME.to_owned(),
        description: Some(
            "Search the current web when up-to-date or externally sourced information is needed."
                .to_owned(),
        ),
        parameters: json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The web search query."
                }
            },
            "required": ["query"],
            "additionalProperties": false
        }),
        strict: Some(true),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_non_empty_query() {
        assert_eq!(
            WebSearchRequest::from_arguments(r#"{"query":"latest Rust release"}"#).unwrap(),
            WebSearchRequest {
                query: "latest Rust release".to_owned()
            }
        );
        assert!(WebSearchRequest::from_arguments(r#"{"query":"  "}"#).is_err());
    }

    #[test]
    fn structured_results_are_serialized_for_chat_tool_messages() {
        let result = WebSearchResult {
            output: json!({ "answer": "result", "sources": ["https://example.com"] }),
        };
        assert_eq!(
            result.into_tool_content().unwrap(),
            Value::String(r#"{"answer":"result","sources":["https://example.com"]}"#.to_owned())
        );
    }
}
