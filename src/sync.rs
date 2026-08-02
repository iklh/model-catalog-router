use crate::catalog::{
    configured_models, configured_openai_compact_models, write_catalog, RoutedModel,
    DEFAULT_REASONING_EFFORT,
};
use crate::config::{
    default_catalog_path, default_openai_compact_catalog_path, write_private_file, Config,
};
use anyhow::{bail, Context, Result};
use directories::BaseDirs;
use std::fs;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use toml_edit::{value, DocumentMut, Item, Table};

pub const PROFILE_NAME: &str = "mc-router";
pub const OPENAI_COMPACT_PROFILE_NAME: &str = "mc-router-openai-compact";
const BASE_PROVIDER_ID: &str = "model-catalog-router";
const OPENAI_COMPACT_PROVIDER_ID: &str = "model-catalog-router-openai-compact";

#[derive(Debug)]
pub struct SyncedProfile {
    pub catalog_path: PathBuf,
    pub profile_path: PathBuf,
    pub model: String,
}

#[derive(Debug)]
pub struct SyncResult {
    pub base: SyncedProfile,
    pub openai_compact: Option<SyncedProfile>,
}

pub fn default_profile_path() -> Result<PathBuf> {
    profile_path(PROFILE_NAME)
}

pub fn default_openai_compact_profile_path() -> Result<PathBuf> {
    profile_path(OPENAI_COMPACT_PROFILE_NAME)
}

fn profile_path(name: &str) -> Result<PathBuf> {
    let codex_home = match std::env::var_os("CODEX_HOME") {
        Some(path) => PathBuf::from(path),
        None => BaseDirs::new()
            .context("could not determine the user home directory")?
            .home_dir()
            .join(".codex"),
    };
    Ok(codex_home.join(format!("{name}.config.toml")))
}

pub fn current_profile_model(profile_path: &Path) -> Result<Option<String>> {
    if !profile_path.exists() {
        return Ok(None);
    }
    let document = read_profile(profile_path)?;
    Ok(document
        .get("model")
        .and_then(|item| item.as_str())
        .map(ToOwned::to_owned))
}

pub fn sync_files(
    config: &Config,
    config_path: &Path,
    selected_model: Option<&str>,
    selected_openai_compact_model: Option<&str>,
) -> Result<SyncResult> {
    let base = sync_profile(
        config,
        &configured_models(config),
        &absolute_path(&default_catalog_path(config_path))?,
        &default_profile_path()?,
        selected_model,
        BASE_PROVIDER_ID,
        "Model Catalog Router",
        config.server.listen,
    )?;

    let compact_models = configured_openai_compact_models(config);
    let compact_catalog_path = absolute_path(&default_openai_compact_catalog_path(config_path))?;
    let compact_profile_path = default_openai_compact_profile_path()?;
    let openai_compact = if compact_models.is_empty() {
        remove_if_exists(&compact_catalog_path)?;
        remove_if_exists(&compact_profile_path)?;
        None
    } else {
        Some(sync_profile(
            config,
            &compact_models,
            &compact_catalog_path,
            &compact_profile_path,
            selected_openai_compact_model,
            OPENAI_COMPACT_PROVIDER_ID,
            "OpenAI",
            config.server.openai_compact_listen,
        )?)
    };

    Ok(SyncResult {
        base,
        openai_compact,
    })
}

#[allow(clippy::too_many_arguments)]
fn sync_profile(
    config: &Config,
    models: &[RoutedModel],
    catalog_path: &Path,
    profile_path: &Path,
    selected_model: Option<&str>,
    provider_id: &str,
    provider_name: &str,
    listen: SocketAddr,
) -> Result<SyncedProfile> {
    if models.is_empty() {
        bail!("cannot synchronize an empty model catalog");
    }
    write_catalog(catalog_path, models, config.catalog.context_window)?;

    let mut document = if profile_path.exists() {
        read_profile(profile_path)?
    } else {
        DocumentMut::new()
    };
    let current_model = document
        .get("model")
        .and_then(|item| item.as_str())
        .map(ToOwned::to_owned);
    let model = select_profile_model(models, selected_model, current_model.as_deref())?;

    document["model"] = value(&model);
    document["model_provider"] = value(provider_id);
    document["model_catalog_json"] = value(catalog_path.to_string_lossy().to_string());
    if document.get("model_reasoning_effort").is_none() {
        document["model_reasoning_effort"] = value(DEFAULT_REASONING_EFFORT);
    }
    ensure_provider_table(&mut document, provider_id)?;
    document["model_providers"][provider_id]["name"] = value(provider_name);
    document["model_providers"][provider_id]["base_url"] = value(format!("http://{listen}/v1"));
    document["model_providers"][provider_id]["wire_api"] = value("responses");

    write_private_file(profile_path, document.to_string().as_bytes())?;
    Ok(SyncedProfile {
        catalog_path: catalog_path.to_path_buf(),
        profile_path: profile_path.to_path_buf(),
        model,
    })
}

fn ensure_provider_table(document: &mut DocumentMut, provider_id: &str) -> Result<()> {
    const PROVIDERS: &str = "model_providers";

    if document.get(PROVIDERS).is_none() {
        document[PROVIDERS] = Item::Table(Table::new());
    } else if document[PROVIDERS].is_inline_table() {
        let inline = document[PROVIDERS]
            .as_inline_table()
            .expect("checked inline table")
            .clone();
        let mut table = Table::new();
        for (key, value) in inline.iter() {
            table.insert(key, Item::Value(value.clone()));
        }
        document[PROVIDERS] = Item::Table(table);
    } else if !document[PROVIDERS].is_table() {
        bail!("profile `model_providers` must be a TOML table");
    }

    let providers = document[PROVIDERS]
        .as_table_mut()
        .expect("model_providers is a table");
    providers.set_implicit(true);
    if !providers.contains_key(provider_id) || !providers[provider_id].is_table() {
        providers.insert(provider_id, Item::Table(Table::new()));
    }
    Ok(())
}

fn select_profile_model(
    models: &[RoutedModel],
    selected: Option<&str>,
    current: Option<&str>,
) -> Result<String> {
    if let Some(selected) = selected {
        if models.iter().any(|model| model.routed_id == selected) {
            return Ok(selected.to_owned());
        }
        bail!("selected profile model `{selected}` is not in the generated catalog");
    }
    if let Some(current) = current {
        if models.iter().any(|model| model.routed_id == current) {
            return Ok(current.to_owned());
        }
    }
    Ok(models[0].routed_id.clone())
}

fn read_profile(path: &Path) -> Result<DocumentMut> {
    fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?
        .parse::<DocumentMut>()
        .with_context(|| format!("failed to parse {}", path.display()))
}

fn absolute_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()
            .context("failed to determine current directory")?
            .join(path))
    }
}

fn remove_if_exists(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("failed to remove {}", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CatalogConfig, ProviderConfig, ServerConfig};
    use std::collections::BTreeMap;
    use url::Url;

    #[test]
    fn keeps_current_profile_model_when_still_available() {
        let models = vec![
            RoutedModel {
                provider: "a".into(),
                upstream_id: "one".into(),
                routed_id: "a/one".into(),
                uses_chat_completions: false,
            },
            RoutedModel {
                provider: "b".into(),
                upstream_id: "two".into(),
                routed_id: "b/two".into(),
                uses_chat_completions: false,
            },
        ];
        assert_eq!(
            select_profile_model(&models, None, Some("b/two")).unwrap(),
            "b/two"
        );
        assert_eq!(
            select_profile_model(&models, None, Some("gone/model")).unwrap(),
            "a/one"
        );
    }

    #[test]
    fn writes_base_and_openai_compact_profiles_without_upstream_secret() {
        let root = std::env::temp_dir().join(format!(
            "model-catalog-router-sync-test-{}",
            std::process::id()
        ));
        let base_profile_path = root.join("codex/mc-router.config.toml");
        let compact_profile_path = root.join("codex/mc-router-openai-compact.config.toml");
        let mut providers = BTreeMap::new();
        providers.insert(
            "alpha".to_owned(),
            ProviderConfig {
                base_url: Url::parse("https://example.com/v1").unwrap(),
                api_key: Some("upstream-secret".to_owned()),
                api_key_env: None,
                enabled: true,
                models: vec!["gpt-test".to_owned()],
                chat_models: Vec::new(),
                remote_compaction_models: vec!["gpt-test".to_owned()],
            },
        );
        let config = Config {
            server: ServerConfig::default(),
            catalog: CatalogConfig::default(),
            web_search: None,
            providers,
        };

        let base_models = configured_models(&config);
        sync_profile(
            &config,
            &base_models,
            &root.join("router/model-catalog.json"),
            &base_profile_path,
            None,
            BASE_PROVIDER_ID,
            "Model Catalog Router",
            config.server.listen,
        )
        .unwrap();
        let compact_models = configured_openai_compact_models(&config);
        sync_profile(
            &config,
            &compact_models,
            &root.join("router/model-catalog-openai-compact.json"),
            &compact_profile_path,
            None,
            OPENAI_COMPACT_PROVIDER_ID,
            "OpenAI",
            config.server.openai_compact_listen,
        )
        .unwrap();

        let base_profile = fs::read_to_string(&base_profile_path).unwrap();
        assert!(base_profile.contains("[model_providers.model-catalog-router]"));
        assert!(base_profile.contains("model = \"alpha/gpt-test\""));
        assert!(!base_profile.contains("upstream-secret"));

        let compact_profile = fs::read_to_string(&compact_profile_path).unwrap();
        assert!(compact_profile.contains("[model_providers.model-catalog-router-openai-compact]"));
        assert!(compact_profile.contains("name = \"OpenAI\""));
        assert!(compact_profile.contains("http://127.0.0.1:8788/v1"));
        assert!(!compact_profile.contains("gpt-test-openai-compact"));
        assert!(!compact_profile.contains("upstream-secret"));
        let _ = fs::remove_dir_all(root);
    }
}
