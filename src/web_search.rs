use crate::chat::{ChatFunctionDefinition, ChatTool};
use anyhow::{bail, Context, Result};
use futures_util::future::BoxFuture;
use rmcp::{
    model::{CallToolRequestParams, ClientInfo},
    service::{Peer, RoleClient},
    transport::StreamableHttpClientTransport,
    ServiceExt,
};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;
use url::Url;

pub const INTERNAL_WEB_SEARCH_TOOL_NAME: &str = "mc_router__web_search";
pub const MCP_WEB_SEARCH_TOOL_NAME: &str = "web_search";
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

#[derive(Debug)]
pub struct McpWebSearchBackend {
    client: Peer<RoleClient>,
    cancellation: CancellationToken,
}

impl McpWebSearchBackend {
    pub async fn connect(endpoint: &Url) -> Result<Self> {
        let cancellation = CancellationToken::new();
        let transport = StreamableHttpClientTransport::from_uri(endpoint.as_str());
        let service = ClientInfo::default()
            .serve_with_ct(transport, cancellation.child_token())
            .await
            .with_context(|| format!("failed to connect to Web Search MCP at {endpoint}"))?;
        let tools = service
            .list_all_tools()
            .await
            .context("failed to list Web Search MCP tools")?;
        if !tools
            .iter()
            .any(|tool| tool.name == MCP_WEB_SEARCH_TOOL_NAME)
        {
            cancellation.cancel();
            bail!("Web Search MCP at {endpoint} does not expose `{MCP_WEB_SEARCH_TOOL_NAME}`");
        }

        let client = service.peer().clone();
        tokio::spawn(async move {
            match service.waiting().await {
                Ok(reason) => tracing::debug!(?reason, "Web Search MCP connection stopped"),
                Err(error) => tracing::warn!(%error, "Web Search MCP connection task failed"),
            }
        });
        Ok(Self {
            client,
            cancellation,
        })
    }
}

impl Drop for McpWebSearchBackend {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

impl WebSearchBackend for McpWebSearchBackend {
    fn search(&self, request: WebSearchRequest) -> BoxFuture<'static, Result<WebSearchResult>> {
        let client = self.client.clone();
        Box::pin(async move {
            let arguments = serde_json::from_value(json!({ "query": request.query }))
                .context("failed to encode Web Search MCP arguments")?;
            let result = client
                .call_tool(
                    CallToolRequestParams::new(MCP_WEB_SEARCH_TOOL_NAME).with_arguments(arguments),
                )
                .await
                .context("Web Search MCP tool call failed")?;
            if result.is_error == Some(true) {
                let details = result
                    .structured_content
                    .map(|value| value.to_string())
                    .or_else(|| serde_json::to_string(&result.content).ok())
                    .unwrap_or_else(|| "unknown MCP tool error".to_owned());
                bail!("Web Search MCP returned an error: {details}");
            }
            let output = result
                .structured_content
                .context("Web Search MCP returned no structuredContent")?;
            Ok(WebSearchResult { output })
        })
    }
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
    use axum::http::StatusCode;
    use axum::response::{IntoResponse, Response};
    use axum::routing::post;
    use axum::{Json, Router};

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

    async fn fake_mcp(Json(request): Json<Value>) -> Response {
        let id = request.get("id").cloned().unwrap_or(Value::Null);
        match request.get("method").and_then(Value::as_str) {
            Some("initialize") => Json(json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "protocolVersion": request["params"]["protocolVersion"],
                    "capabilities": { "tools": {} },
                    "serverInfo": {
                        "name": "fake-web-search",
                        "version": "1.0"
                    }
                }
            }))
            .into_response(),
            Some("notifications/initialized") => StatusCode::ACCEPTED.into_response(),
            Some("tools/list") => Json(json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "tools": [{
                        "name": MCP_WEB_SEARCH_TOOL_NAME,
                        "description": "test",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": { "type": "string" }
                            },
                            "required": ["query"]
                        }
                    }]
                }
            }))
            .into_response(),
            Some("tools/call") => {
                assert_eq!(request["params"]["name"], MCP_WEB_SEARCH_TOOL_NAME);
                assert_eq!(request["params"]["arguments"]["query"], "latest Rust");
                Json(json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "structuredContent": {
                            "answer": "Rust result.",
                            "sources": [{
                                "title": "Rust",
                                "url": "https://www.rust-lang.org/"
                            }]
                        }
                    }
                }))
                .into_response()
            }
            method => panic!("unexpected MCP method: {method:?}"),
        }
    }

    #[tokio::test]
    async fn mcp_backend_connects_and_returns_structured_content() {
        let app = Router::new().route("/mcp", post(fake_mcp));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let endpoint = Url::parse(&format!("http://{address}/mcp")).unwrap();
        let backend = McpWebSearchBackend::connect(&endpoint).await.unwrap();
        let result = backend
            .search(WebSearchRequest {
                query: "latest Rust".to_owned(),
            })
            .await
            .unwrap();
        assert_eq!(
            result.output,
            json!({
                "answer": "Rust result.",
                "sources": [{
                    "title": "Rust",
                    "url": "https://www.rust-lang.org/"
                }]
            })
        );

        drop(backend);
        server.abort();
    }
}
