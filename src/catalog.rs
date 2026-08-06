use crate::config::{Config, ProviderConfig};
use anyhow::{bail, Context, Result};
use directories::BaseDirs;
use futures_util::stream::{FuturesUnordered, StreamExt};
use reqwest::header::{ACCEPT, AUTHORIZATION};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::warn;

const BASE_INSTRUCTIONS: &str = "You are a coding agent. Work with the user in the current workspace, inspect relevant files before editing, make focused changes, and verify your work with appropriate tests.";
pub const DEFAULT_REASONING_EFFORT: &str = "medium";
pub const OPENAI_COMPACT_SUFFIX: &str = "-openai-compact";

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RoutedModel {
    pub provider: String,
    pub upstream_id: String,
    pub routed_id: String,
    #[serde(skip_serializing)]
    pub uses_chat_completions: bool,
}

#[derive(Debug, Deserialize)]
struct UpstreamModelsResponse {
    data: Vec<UpstreamModel>,
}

#[derive(Debug, Deserialize)]
struct UpstreamModel {
    id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredProviderModels {
    pub models: Vec<String>,
    pub remote_compaction_models: Vec<String>,
}

pub async fn fetch_models(config: &Config) -> Result<Vec<RoutedModel>> {
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .build()
        .context("failed to build HTTP client")?;
    let mut pending = FuturesUnordered::new();

    for (name, provider) in config
        .providers
        .iter()
        .filter(|(_, provider)| provider.enabled)
    {
        let name = name.clone();
        let provider = provider.clone();
        let client = client.clone();
        let separator = config.catalog.separator.clone();
        pending.push(async move {
            let api_key = provider.resolved_api_key(&name)?;
            let endpoint = provider.endpoint("models")?;
            let response = client
                .get(endpoint)
                .header(ACCEPT, "application/json")
                .header(AUTHORIZATION, format!("Bearer {api_key}"))
                .send()
                .await
                .with_context(|| format!("provider `{name}` models request failed"))?;
            let status = response.status();
            if !status.is_success() {
                let body = response.text().await.unwrap_or_default();
                bail!(
                    "provider `{name}` returned {status} from /models: {}",
                    truncate(&body, 300)
                );
            }
            let response: UpstreamModelsResponse = response.json().await.with_context(|| {
                format!("provider `{name}` returned an invalid /models response")
            })?;
            let models = response
                .data
                .into_iter()
                .filter(|model| !model.id.is_empty())
                .map(|model| RoutedModel {
                    routed_id: format!("{name}{separator}{}", model.id),
                    provider: name.clone(),
                    uses_chat_completions: provider.chat_models.contains(&model.id),
                    upstream_id: model.id,
                })
                .collect::<Vec<_>>();
            Ok::<_, anyhow::Error>(models)
        });
    }

    let mut models = Vec::new();
    while let Some(result) = pending.next().await {
        models.extend(result?);
    }
    models.sort_by(|left, right| left.routed_id.cmp(&right.routed_id));
    models.dedup_by(|left, right| left.routed_id == right.routed_id);
    if models.is_empty() {
        bail!("enabled providers returned no models");
    }
    Ok(models)
}

pub async fn discover_provider_models(
    name: &str,
    provider: &ProviderConfig,
) -> Result<DiscoveredProviderModels> {
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .build()
        .context("failed to build HTTP client")?;
    let api_key = provider.resolved_api_key(name)?;
    let endpoint = provider.endpoint("models")?;
    let response = client
        .get(endpoint)
        .header(ACCEPT, "application/json")
        .header(AUTHORIZATION, format!("Bearer {api_key}"))
        .send()
        .await
        .with_context(|| format!("provider `{name}` models request failed"))?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        bail!(
            "provider `{name}` returned {status} from /models: {}",
            truncate(&body, 300)
        );
    }
    let response: UpstreamModelsResponse = response
        .json()
        .await
        .with_context(|| format!("provider `{name}` returned an invalid /models response"))?;
    let models = response
        .data
        .into_iter()
        .map(|model| model.id.trim().to_owned())
        .filter(|model| !model.is_empty())
        .collect::<Vec<_>>();
    let discovered = classify_provider_models(models);
    if discovered.models.is_empty() {
        bail!("provider `{name}` returned no models");
    }
    Ok(discovered)
}

pub fn configured_models(config: &Config) -> Vec<RoutedModel> {
    configured_models_by(
        config,
        |provider| &provider.models,
        |provider, model| provider.chat_models.iter().any(|chat| chat == model),
    )
}

pub fn configured_openai_compact_models(config: &Config) -> Vec<RoutedModel> {
    configured_models_by(
        config,
        |provider| &provider.remote_compaction_models,
        |_, _| false,
    )
}

fn configured_models_by<'a>(
    config: &'a Config,
    selected_models: fn(&'a ProviderConfig) -> &'a [String],
    uses_chat_completions: fn(&ProviderConfig, &str) -> bool,
) -> Vec<RoutedModel> {
    let separator = &config.catalog.separator;
    let mut models = config
        .providers
        .iter()
        .filter(|(_, provider)| provider.enabled)
        .flat_map(|(name, provider)| {
            selected_models(provider)
                .iter()
                .map(move |model| RoutedModel {
                    provider: name.clone(),
                    upstream_id: model.clone(),
                    routed_id: format!("{name}{separator}{model}"),
                    uses_chat_completions: uses_chat_completions(provider, model),
                })
        })
        .collect::<Vec<_>>();
    models.sort_by(|left, right| left.routed_id.cmp(&right.routed_id));
    models.dedup_by(|left, right| left.routed_id == right.routed_id);
    models
}

fn classify_provider_models(models: Vec<String>) -> DiscoveredProviderModels {
    let mut advertised = models
        .into_iter()
        .map(|model| model.trim().to_owned())
        .filter(|model| !model.is_empty())
        .collect::<Vec<_>>();
    advertised.sort();
    advertised.dedup();

    let base_models = advertised
        .iter()
        .filter(|model| !model.ends_with(OPENAI_COMPACT_SUFFIX))
        .cloned()
        .collect::<Vec<_>>();
    let base_set = base_models
        .iter()
        .map(String::as_str)
        .collect::<std::collections::HashSet<_>>();
    let remote_compaction_models = advertised
        .iter()
        .filter_map(|model| model.strip_suffix(OPENAI_COMPACT_SUFFIX))
        .filter(|base| base_set.contains(base))
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();

    DiscoveredProviderModels {
        models: base_models,
        remote_compaction_models,
    }
}

pub fn write_catalog(path: &Path, models: &[RoutedModel], context_window: i64) -> Result<()> {
    if models.is_empty() {
        bail!("refusing to write an empty model catalog");
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let catalog = build_codex_catalog(models, context_window);
    let bytes = serde_json::to_vec_pretty(&catalog)?;
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, bytes)
        .with_context(|| format!("failed to write {}", temporary.display()))?;
    fs::rename(&temporary, path)
        .with_context(|| format!("failed to replace {}", path.display()))?;
    Ok(())
}

pub fn build_codex_catalog(models: &[RoutedModel], context_window: i64) -> Value {
    let original_models = load_codex_models().unwrap_or_else(|error| {
        warn!("could not load the Codex model cache: {error:#}");
        BTreeMap::new()
    });
    build_codex_catalog_with_originals(models, context_window, &original_models)
}

fn build_codex_catalog_with_originals(
    models: &[RoutedModel],
    context_window: i64,
    original_models: &BTreeMap<String, Value>,
) -> Value {
    let entries = models
        .iter()
        .enumerate()
        .map(|(index, model)| {
            let original = original_models.get(&model.upstream_id);
            let supported_reasoning_levels = if !is_openai_model(&model.upstream_id) {
                chat_reasoning_levels()
            } else {
                original
                    .and_then(|entry| entry.get("supported_reasoning_levels"))
                    .cloned()
                    .unwrap_or_else(|| json!([]))
            };
            let default_reasoning_level = if reasoning_level_is_supported(
                &supported_reasoning_levels,
                DEFAULT_REASONING_EFFORT,
            ) {
                Some(json!(DEFAULT_REASONING_EFFORT))
            } else {
                original
                    .and_then(|entry| entry.get("default_reasoning_level"))
                    .filter(|level| {
                        level.as_str().is_some_and(|effort| {
                            reasoning_level_is_supported(&supported_reasoning_levels, effort)
                        })
                    })
                    .cloned()
            };
            let mut entry = json!({
                "slug": model.routed_id,
                "display_name": format!("{} / {}", model.provider, model.upstream_id),
                "description": format!("{} via {}", model.upstream_id, model.provider),
                "supported_reasoning_levels": supported_reasoning_levels,
                "shell_type": "shell_command",
                "visibility": "list",
                "supported_in_api": true,
                "priority": index as i32,
                "availability_nux": null,
                "upgrade": null,
                "base_instructions": BASE_INSTRUCTIONS,
                "support_verbosity": false,
                "default_verbosity": null,
                "apply_patch_tool_type": "freeform",
                "truncation_policy": { "mode": "tokens", "limit": 10000 },
                "supports_parallel_tool_calls": true,
                "context_window": context_window,
                "max_context_window": context_window,
                "auto_compact_token_limit": context_window * 9 / 10,
                "effective_context_window_percent": 95,
                "experimental_supported_tools": [],
                "input_modalities": ["text"]
            });
            if let Some(default_reasoning_level) = default_reasoning_level {
                entry["default_reasoning_level"] = default_reasoning_level;
            }
            entry
        })
        .collect::<Vec<_>>();
    json!({ "models": entries })
}

fn is_openai_model(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    let model = model.rsplit('/').next().unwrap_or(&model);
    ["gpt", "o1", "o3", "o4", "codex"]
        .iter()
        .any(|prefix| model.starts_with(prefix))
}

fn chat_reasoning_levels() -> Value {
    json!([
        {
            "effort": "none",
            "description": "Do not specify a reasoning effort."
        },
        {
            "effort": "low",
            "description": "Faster responses with less reasoning."
        },
        {
            "effort": "medium",
            "description": "Balances speed and reasoning."
        },
        {
            "effort": "high",
            "description": "More reasoning for complex tasks."
        }
    ])
}

fn reasoning_level_is_supported(levels: &Value, effort: &str) -> bool {
    levels.as_array().is_some_and(|levels| {
        levels
            .iter()
            .any(|level| level.get("effort").and_then(Value::as_str) == Some(effort))
    })
}

fn load_codex_models() -> Result<BTreeMap<String, Value>> {
    let path = codex_models_cache_path()?;
    if !path.exists() {
        return Ok(BTreeMap::new());
    }
    let contents =
        fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let catalog: Value = serde_json::from_str(&contents)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    let entries = catalog
        .get("models")
        .and_then(Value::as_array)
        .with_context(|| format!("{} does not contain a models array", path.display()))?;
    Ok(entries
        .iter()
        .filter_map(|entry| {
            entry
                .get("slug")
                .and_then(Value::as_str)
                .map(|slug| (slug.to_owned(), entry.clone()))
        })
        .collect())
}

fn codex_models_cache_path() -> Result<PathBuf> {
    let codex_home = match std::env::var_os("CODEX_HOME") {
        Some(path) => PathBuf::from(path),
        None => BaseDirs::new()
            .context("could not determine the user home directory")?
            .home_dir()
            .join(".codex"),
    };
    Ok(codex_home.join("models_cache.json"))
}

fn truncate(value: &str, limit: usize) -> String {
    let mut chars = value.chars();
    let shortened = chars.by_ref().take(limit).collect::<String>();
    if chars.next().is_some() {
        format!("{shortened}...")
    } else {
        shortened
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use url::Url;

    #[test]
    fn catalog_uses_routed_slug_and_required_wrapper() {
        let models = vec![RoutedModel {
            provider: "alpha".into(),
            upstream_id: "gpt-test".into(),
            routed_id: "alpha/gpt-test".into(),
            uses_chat_completions: false,
        }];
        let catalog = build_codex_catalog(&models, 64_000);
        assert_eq!(catalog["models"][0]["slug"], "alpha/gpt-test");
        assert_eq!(catalog["models"][0]["context_window"], 64_000);
    }

    #[test]
    fn discovery_hides_compact_aliases_and_ignores_orphans() {
        let discovered = classify_provider_models(vec![
            "sol".into(),
            "sol-openai-compact".into(),
            "luna-openai-compact".into(),
            "terra".into(),
            "gpt-5.5-openai-compact".into(),
        ]);
        assert_eq!(discovered.models, vec!["sol", "terra"]);
        assert_eq!(discovered.remote_compaction_models, vec!["sol"]);
    }

    #[test]
    fn compact_catalog_uses_selected_base_ids_only() {
        let provider = ProviderConfig {
            base_url: Url::parse("https://example.com/v1").unwrap(),
            api_key: Some("secret".into()),
            api_key_env: None,
            enabled: true,
            models: vec!["sol".into(), "terra".into()],
            chat_models: Vec::new(),
            messages_models: Vec::new(),
            remote_compaction_models: vec!["sol".into()],
        };
        let config = Config {
            providers: BTreeMap::from([("alpha".to_owned(), provider)]),
            ..Config::default()
        };
        let compact = configured_openai_compact_models(&config);
        assert_eq!(compact.len(), 1);
        assert_eq!(compact[0].upstream_id, "sol");
        assert_eq!(compact[0].routed_id, "alpha/sol");
        assert!(!compact[0].uses_chat_completions);
    }

    #[test]
    fn chat_models_use_fixed_reasoning_levels() {
        let provider = ProviderConfig {
            base_url: Url::parse("https://example.com/v1").unwrap(),
            api_key: Some("secret".into()),
            api_key_env: None,
            enabled: true,
            models: vec!["glm-test".into(), "gpt-test".into()],
            chat_models: vec!["glm-test".into()],
            messages_models: Vec::new(),
            remote_compaction_models: Vec::new(),
        };
        let config = Config {
            providers: BTreeMap::from([("alpha".to_owned(), provider)]),
            ..Config::default()
        };
        let models = configured_models(&config);
        let catalog = build_codex_catalog_with_originals(&models, 64_000, &BTreeMap::new());
        let entries = catalog["models"].as_array().unwrap();
        let chat = entries
            .iter()
            .find(|entry| entry["slug"] == "alpha/glm-test")
            .unwrap();
        let native = entries
            .iter()
            .find(|entry| entry["slug"] == "alpha/gpt-test")
            .unwrap();

        assert_eq!(
            chat["supported_reasoning_levels"]
                .as_array()
                .unwrap()
                .iter()
                .map(|level| level["effort"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["none", "low", "medium", "high"]
        );
        assert_eq!(chat["default_reasoning_level"], DEFAULT_REASONING_EFFORT);
        assert_eq!(native["supported_reasoning_levels"], json!([]));
    }

    #[test]
    fn non_openai_models_use_fixed_reasoning_levels_regardless_of_transport() {
        let models = vec![
            RoutedModel {
                provider: "alpha".into(),
                upstream_id: "grok-4".into(),
                routed_id: "alpha/grok-4".into(),
                uses_chat_completions: false,
            },
            RoutedModel {
                provider: "alpha".into(),
                upstream_id: "claude-sonnet".into(),
                routed_id: "alpha/claude-sonnet".into(),
                uses_chat_completions: true,
            },
        ];
        let catalog = build_codex_catalog_with_originals(
            &models,
            64_000,
            &BTreeMap::from([
                (
                    "grok-4".to_owned(),
                    json!({ "supported_reasoning_levels": [] }),
                ),
                (
                    "claude-sonnet".to_owned(),
                    json!({ "supported_reasoning_levels": [] }),
                ),
            ]),
        );

        for entry in catalog["models"].as_array().unwrap() {
            assert_eq!(entry["supported_reasoning_levels"], chat_reasoning_levels());
            assert_eq!(entry["default_reasoning_level"], DEFAULT_REASONING_EFFORT);
        }
    }

    #[test]
    fn openai_model_families_keep_upstream_reasoning_levels() {
        let models = ["gpt-5", "o3", "o4-mini", "codex-mini"]
            .into_iter()
            .map(|model| RoutedModel {
                provider: "alpha".into(),
                upstream_id: model.into(),
                routed_id: format!("alpha/{model}"),
                uses_chat_completions: false,
            })
            .collect::<Vec<_>>();
        let originals = models
            .iter()
            .map(|model| {
                (
                    model.upstream_id.clone(),
                    json!({
                        "supported_reasoning_levels": [
                            { "effort": "high", "description": "Deep" }
                        ],
                        "default_reasoning_level": "high"
                    }),
                )
            })
            .collect::<BTreeMap<_, _>>();

        let catalog = build_codex_catalog_with_originals(&models, 64_000, &originals);
        for entry in catalog["models"].as_array().unwrap() {
            assert_eq!(
                entry["supported_reasoning_levels"],
                json!([{ "effort": "high", "description": "Deep" }])
            );
            assert_eq!(entry["default_reasoning_level"], "high");
        }
    }

    #[test]
    fn catalog_inherits_reasoning_levels_from_matching_codex_model() {
        let models = vec![RoutedModel {
            provider: "alpha".into(),
            upstream_id: "gpt-test".into(),
            routed_id: "alpha/gpt-test".into(),
            uses_chat_completions: false,
        }];
        let originals = BTreeMap::from([(
            "gpt-test".to_owned(),
            json!({
                "slug": "gpt-test",
                "default_reasoning_level": "high",
                "supported_reasoning_levels": [
                    { "effort": "low", "description": "Fast" },
                    { "effort": "high", "description": "Deep" }
                ]
            }),
        )]);

        let catalog = build_codex_catalog_with_originals(&models, 64_000, &originals);

        assert_eq!(catalog["models"][0]["default_reasoning_level"], "high");
        assert_eq!(
            catalog["models"][0]["supported_reasoning_levels"][1]["effort"],
            "high"
        );
    }

    #[test]
    fn catalog_prefers_router_default_when_supported() {
        let models = vec![RoutedModel {
            provider: "alpha".into(),
            upstream_id: "gpt-test".into(),
            routed_id: "alpha/gpt-test".into(),
            uses_chat_completions: false,
        }];
        let originals = BTreeMap::from([(
            "gpt-test".to_owned(),
            json!({
                "slug": "gpt-test",
                "default_reasoning_level": "low",
                "supported_reasoning_levels": [
                    { "effort": "low", "description": "Fast" },
                    { "effort": "medium", "description": "Balanced" },
                    { "effort": "high", "description": "Deep" }
                ]
            }),
        )]);

        let catalog = build_codex_catalog_with_originals(&models, 64_000, &originals);

        assert_eq!(
            catalog["models"][0]["default_reasoning_level"],
            DEFAULT_REASONING_EFFORT
        );
    }

    #[test]
    fn catalog_omits_invalid_inherited_reasoning_default() {
        let models = vec![RoutedModel {
            provider: "alpha".into(),
            upstream_id: "gpt-test".into(),
            routed_id: "alpha/gpt-test".into(),
            uses_chat_completions: false,
        }];
        let originals = BTreeMap::from([(
            "gpt-test".to_owned(),
            json!({
                "slug": "gpt-test",
                "default_reasoning_level": "high",
                "supported_reasoning_levels": [
                    { "effort": "low", "description": "Fast" }
                ]
            }),
        )]);

        let catalog = build_codex_catalog_with_originals(&models, 64_000, &originals);

        assert!(catalog["models"][0]
            .get("default_reasoning_level")
            .is_none());
    }
}
