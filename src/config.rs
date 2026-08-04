use anyhow::{bail, Context, Result};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use url::Url;

const EXAMPLE_CONFIG: &str = r#"# Model Catalog Router configuration

[server]
listen = "127.0.0.1:8787"
openai_compact_listen = "127.0.0.1:8788"

[catalog]
separator = "/"
context_window = 128000

# Optional Web Search MCP URL override. Enable it at runtime with `serve --web-search`.
# [web_search]
# mcp_url = "http://127.0.0.1:9091/mcp"

[providers.example]
base_url = "https://new-api.example.com/v1"
api_key_env = "EXAMPLE_NEW_API_KEY"
enabled = true
models = ["gpt-5"]
chat_models = []
remote_compaction_models = []
"#;

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub catalog: CatalogConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing: Option<RoutingConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub web_search: Option<WebSearchConfig>,
    #[serde(default)]
    pub providers: BTreeMap<String, ProviderConfig>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    #[serde(default = "default_listen")]
    pub listen: SocketAddr,
    #[serde(default = "default_openai_compact_listen")]
    pub openai_compact_listen: SocketAddr,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            listen: default_listen(),
            openai_compact_listen: default_openai_compact_listen(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogConfig {
    #[serde(default = "default_separator")]
    pub separator: String,
    #[serde(default = "default_context_window")]
    pub context_window: i64,
}

impl Default for CatalogConfig {
    fn default() -> Self {
        Self {
            separator: default_separator(),
            context_window: default_context_window(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WebSearchConfig {
    pub mcp_url: Url,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RoutingConfig {
    pub unprefixed_model_provider: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderConfig {
    pub base_url: Url,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub models: Vec<String>,
    #[serde(default)]
    pub chat_models: Vec<String>,
    #[serde(default)]
    pub remote_compaction_models: Vec<String>,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        load_dotenv(path)?;
        let text = fs::read_to_string(path).with_context(|| {
            format!(
                "failed to read {} (run `model-catalog-router config`)",
                path.display()
            )
        })?;
        toml::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))
    }

    pub fn load_or_default(path: &Path) -> Result<Self> {
        if path.exists() {
            Self::load(path)
        } else {
            load_dotenv(path)?;
            Ok(Self::default())
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let text = toml::to_string_pretty(self).context("failed to serialize configuration")?;
        write_private_file(path, text.as_bytes())
    }

    pub fn validate(&self) -> Result<()> {
        self.validate_inner(true)
    }

    pub fn validate_for_save(&self) -> Result<()> {
        self.validate_inner(false)
    }

    pub fn provider_for_check(&self, name: &str) -> Result<&ProviderConfig> {
        if self.catalog.separator.is_empty() {
            bail!("catalog.separator must not be empty");
        }
        let provider = self
            .providers
            .get(name)
            .with_context(|| format!("provider `{name}` is not configured"))?;
        validate_provider_name(name, &self.catalog.separator)?;
        validate_provider_url(name, provider)?;
        validate_provider(name, provider, true)?;
        Ok(provider)
    }

    pub fn web_search_mcp_url(&self) -> Url {
        self.web_search
            .as_ref()
            .map(|web_search| web_search.mcp_url.clone())
            .unwrap_or_else(default_web_search_mcp_url)
    }

    fn validate_inner(&self, resolve_keys: bool) -> Result<()> {
        if self.providers.is_empty() {
            bail!("at least one provider must be configured");
        }
        if self.catalog.separator.is_empty() {
            bail!("catalog.separator must not be empty");
        }
        if self.catalog.context_window <= 0 {
            bail!("catalog.context_window must be positive");
        }
        if !is_loopback(self.server.listen.ip()) {
            bail!("server.listen must use a loopback address in this version");
        }
        if !is_loopback(self.server.openai_compact_listen.ip()) {
            bail!("server.openai_compact_listen must use a loopback address in this version");
        }
        if self.server.listen == self.server.openai_compact_listen {
            bail!("server.listen and server.openai_compact_listen must be different");
        }
        if let Some(web_search) = &self.web_search {
            validate_web_search_mcp_url(&web_search.mcp_url)?;
        }
        if let Some(routing) = &self.routing {
            let provider = self
                .providers
                .get(&routing.unprefixed_model_provider)
                .with_context(|| {
                    format!(
                        "routing.unprefixed_model_provider `{}` is not configured",
                        routing.unprefixed_model_provider
                    )
                })?;
            if !provider.enabled {
                bail!(
                    "routing.unprefixed_model_provider `{}` is disabled",
                    routing.unprefixed_model_provider
                );
            }
        }

        let mut folded_names = HashSet::new();
        for (name, provider) in &self.providers {
            validate_provider_name(name, &self.catalog.separator)?;
            if !folded_names.insert(name.to_ascii_lowercase()) {
                bail!("provider names must be unique ignoring ASCII case: `{name}`");
            }
            validate_provider_url(name, provider)?;
            if !provider.enabled {
                continue;
            }
            validate_provider(name, provider, resolve_keys)?;
        }
        if !self.providers.values().any(|provider| provider.enabled) {
            bail!("at least one provider must be enabled");
        }
        Ok(())
    }
}

pub fn default_web_search_mcp_url() -> Url {
    Url::parse("http://127.0.0.1:9091/mcp").expect("valid default Web Search MCP URL")
}

pub fn validate_web_search_mcp_url(url: &Url) -> Result<()> {
    if !matches!(url.scheme(), "http" | "https") {
        bail!("web_search.mcp_url must use http or https");
    }
    if url.host_str().is_none() {
        bail!("web_search.mcp_url must contain a host");
    }
    if url.fragment().is_some() {
        bail!("web_search.mcp_url must not contain a fragment");
    }
    Ok(())
}

fn validate_provider_url(name: &str, provider: &ProviderConfig) -> Result<()> {
    if provider.base_url.scheme() != "http" && provider.base_url.scheme() != "https" {
        bail!("provider `{name}` base_url must use http or https");
    }
    Ok(())
}

pub fn validate_provider_name(name: &str, separator: &str) -> Result<()> {
    if name.is_empty() || name.contains(separator) {
        bail!("provider name is empty or contains the catalog separator");
    }
    if !name
        .chars()
        .all(|character| character.is_ascii_alphanumeric() || character == '-' || character == '_')
    {
        bail!("provider name may contain only ASCII letters, digits, hyphens, and underscores");
    }
    Ok(())
}

fn validate_provider(name: &str, provider: &ProviderConfig, resolve_key: bool) -> Result<()> {
    match (&provider.api_key, &provider.api_key_env) {
        (Some(_), Some(_)) => {
            bail!("provider `{name}` must set only one of api_key and api_key_env")
        }
        (None, None) => bail!("provider `{name}` must set api_key or api_key_env"),
        _ => {}
    }
    if provider.models.is_empty() {
        bail!("enabled provider `{name}` must contain at least one model");
    }
    if provider.models.iter().any(|model| model.trim().is_empty()) {
        bail!("provider `{name}` contains an empty model name");
    }
    if provider
        .chat_models
        .iter()
        .any(|model| model.trim().is_empty())
    {
        bail!("provider `{name}` contains an empty Chat model name");
    }
    if let Some(model) = provider
        .chat_models
        .iter()
        .find(|model| !provider.models.contains(model))
    {
        bail!("provider `{name}` Chat model `{model}` is not present in models");
    }
    if provider
        .models
        .iter()
        .any(|model| model.ends_with(crate::catalog::OPENAI_COMPACT_SUFFIX))
    {
        bail!(
            "provider `{name}` models must contain base model names, not `{}` aliases",
            crate::catalog::OPENAI_COMPACT_SUFFIX
        );
    }
    if provider
        .remote_compaction_models
        .iter()
        .any(|model| model.trim().is_empty())
    {
        bail!("provider `{name}` contains an empty remote compaction model name");
    }
    if provider
        .remote_compaction_models
        .iter()
        .any(|model| model.ends_with(crate::catalog::OPENAI_COMPACT_SUFFIX))
    {
        bail!("provider `{name}` remote_compaction_models must contain base model names");
    }
    if let Some(model) = provider
        .remote_compaction_models
        .iter()
        .find(|model| !provider.models.contains(model))
    {
        bail!("provider `{name}` remote compaction model `{model}` is not present in models");
    }
    if resolve_key {
        provider.resolved_api_key(name)?;
    }
    Ok(())
}

impl ProviderConfig {
    pub fn resolved_api_key(&self, name: &str) -> Result<String> {
        if let Some(key) = &self.api_key {
            if key.is_empty() {
                bail!("provider `{name}` api_key must not be empty");
            }
            return Ok(key.clone());
        }
        let env_name = self
            .api_key_env
            .as_deref()
            .context("missing api_key source")?;
        let key = std::env::var(env_name).with_context(|| {
            format!("provider `{name}` requires environment variable `{env_name}`")
        })?;
        if key.is_empty() {
            bail!("environment variable `{env_name}` is empty");
        }
        Ok(key)
    }

    pub fn endpoint(&self, path: &str) -> Result<Url> {
        let mut base = self.base_url.clone();
        let mut base_path = base.path().trim_end_matches('/').to_string();
        base_path.push('/');
        base_path.push_str(path.trim_start_matches('/'));
        base.set_path(&base_path);
        Ok(base)
    }
}

pub fn default_config_path() -> PathBuf {
    ProjectDirs::from("org", "model-catalog-router", "model-catalog-router")
        .map(|dirs| dirs.config_dir().join("config.toml"))
        .unwrap_or_else(|| PathBuf::from(".config/model-catalog-router/config.toml"))
}

pub fn default_catalog_path(config_path: &Path) -> PathBuf {
    config_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("model-catalog.json")
}

pub fn default_openai_compact_catalog_path(config_path: &Path) -> PathBuf {
    config_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("model-catalog-openai-compact.json")
}

pub fn init(path: &Path, force: bool) -> Result<()> {
    if path.exists() && !force {
        bail!(
            "{} already exists; use --force to replace it",
            path.display()
        );
    }
    write_private_file(path, EXAMPLE_CONFIG.as_bytes())?;
    println!("created {}", path.display());
    Ok(())
}

pub fn write_private_file(path: &Path, contents: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .context("configuration path has no parent directory")?;
    fs::create_dir_all(parent).with_context(|| format!("failed to create {}", parent.display()))?;
    set_directory_permissions(parent)?;
    let temporary = path.with_extension("toml.tmp");
    fs::write(&temporary, contents)
        .with_context(|| format!("failed to write {}", temporary.display()))?;
    set_file_permissions(&temporary)?;
    fs::rename(&temporary, path)
        .with_context(|| format!("failed to replace {}", path.display()))?;
    Ok(())
}

fn load_dotenv(config_path: &Path) -> Result<()> {
    let env_path = config_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(".env");
    if env_path.is_file() {
        dotenvy::from_path(&env_path)
            .with_context(|| format!("failed to load {}", env_path.display()))?;
    }
    Ok(())
}

#[cfg(unix)]
fn set_directory_permissions(directory: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(directory, fs::Permissions::from_mode(0o700))?;
    Ok(())
}

#[cfg(not(unix))]
fn set_directory_permissions(_directory: &Path) -> Result<()> {
    Ok(())
}

#[cfg(unix)]
fn set_file_permissions(file: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(file, fs::Permissions::from_mode(0o600))?;
    Ok(())
}

#[cfg(not(unix))]
fn set_file_permissions(_file: &Path) -> Result<()> {
    Ok(())
}

fn is_loopback(ip: IpAddr) -> bool {
    ip.is_loopback()
}
fn default_listen() -> SocketAddr {
    "127.0.0.1:8787"
        .parse()
        .expect("valid default listen address")
}
fn default_openai_compact_listen() -> SocketAddr {
    "127.0.0.1:8788"
        .parse()
        .expect("valid default OpenAI compact listen address")
}
fn default_separator() -> String {
    "/".to_owned()
}
fn default_context_window() -> i64 {
    128_000
}
fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_endpoint_preserves_base_prefix() {
        let provider = ProviderConfig {
            base_url: Url::parse("https://example.com/openai/v1/").unwrap(),
            api_key: Some("secret".into()),
            api_key_env: None,
            enabled: true,
            models: vec!["gpt-test".into()],
            chat_models: Vec::new(),
            remote_compaction_models: Vec::new(),
        };
        assert_eq!(
            provider.endpoint("responses").unwrap().as_str(),
            "https://example.com/openai/v1/responses"
        );
    }

    #[test]
    fn generated_example_is_valid_toml() {
        let config: Config = toml::from_str(EXAMPLE_CONFIG).unwrap();
        assert!(config.providers.contains_key("example"));
        assert_eq!(
            config.server.openai_compact_listen,
            "127.0.0.1:8788".parse().unwrap()
        );
    }

    #[test]
    fn default_serialization_omits_routing_section() {
        let text = toml::to_string_pretty(&Config::default()).unwrap();
        assert!(!text.contains("[routing]"));
        assert!(!text.contains("unprefixed_model_provider"));
    }

    #[test]
    fn existing_config_defaults_new_compact_fields() {
        let config: Config = toml::from_str(
            r#"
[server]
listen = "127.0.0.1:8787"

[providers.example]
base_url = "https://example.com/v1"
api_key = "secret"
models = ["sol"]
"#,
        )
        .unwrap();
        assert_eq!(
            config.server.openai_compact_listen,
            "127.0.0.1:8788".parse().unwrap()
        );
        assert!(config.providers["example"]
            .remote_compaction_models
            .is_empty());
        assert!(config.providers["example"].chat_models.is_empty());
        assert!(config.routing.is_none());
        assert!(config.web_search.is_none());
        assert_eq!(
            config.web_search_mcp_url().as_str(),
            "http://127.0.0.1:9091/mcp"
        );
    }

    #[test]
    fn web_search_mcp_is_optional_and_validates_its_url() {
        let provider = ProviderConfig {
            base_url: Url::parse("https://example.com/v1").unwrap(),
            api_key: Some("secret".into()),
            api_key_env: None,
            enabled: true,
            models: vec!["glm-test".into()],
            chat_models: vec!["glm-test".into()],
            remote_compaction_models: Vec::new(),
        };
        let mut config = Config {
            web_search: Some(WebSearchConfig {
                mcp_url: Url::parse("http://127.0.0.1:9091/mcp").unwrap(),
            }),
            providers: BTreeMap::from([("example".to_owned(), provider)]),
            ..Config::default()
        };
        assert!(config.validate_for_save().is_ok());

        config.web_search.as_mut().unwrap().mcp_url = Url::parse("ftp://127.0.0.1/mcp").unwrap();
        assert!(config.validate_for_save().is_err());
    }

    #[test]
    fn configured_web_search_url_overrides_the_default() {
        let config = Config {
            web_search: Some(WebSearchConfig {
                mcp_url: Url::parse("http://127.0.0.1:9092/custom").unwrap(),
            }),
            ..Config::default()
        };

        assert_eq!(
            config.web_search_mcp_url().as_str(),
            "http://127.0.0.1:9092/custom"
        );
    }

    #[test]
    fn unprefixed_model_provider_must_exist_and_be_enabled() {
        let provider = ProviderConfig {
            base_url: Url::parse("https://example.com/v1").unwrap(),
            api_key: Some("secret".into()),
            api_key_env: None,
            enabled: true,
            models: vec!["sol".into()],
            chat_models: Vec::new(),
            remote_compaction_models: Vec::new(),
        };
        let mut config = Config {
            routing: Some(RoutingConfig {
                unprefixed_model_provider: "example".into(),
            }),
            providers: BTreeMap::from([("example".to_owned(), provider)]),
            ..Config::default()
        };
        assert!(config.validate_for_save().is_ok());

        config.providers.get_mut("example").unwrap().enabled = false;
        assert!(config.validate_for_save().is_err());

        config.routing.as_mut().unwrap().unprefixed_model_provider = "missing".into();
        assert!(config.validate_for_save().is_err());
    }

    #[test]
    fn provider_names_are_safe_catalog_prefixes() {
        assert!(validate_provider_name("newapi-a", "/").is_ok());
        assert!(validate_provider_name("bad/name", "/").is_err());
        assert!(validate_provider_name("bad name", "/").is_err());
    }

    #[test]
    fn explicitly_targeted_disabled_provider_can_be_checked() {
        let provider = ProviderConfig {
            base_url: Url::parse("https://example.com/v1").unwrap(),
            api_key: Some("secret".into()),
            api_key_env: None,
            enabled: false,
            models: vec!["gpt-test".into()],
            chat_models: Vec::new(),
            remote_compaction_models: Vec::new(),
        };
        let config = Config {
            providers: BTreeMap::from([("disabled".to_owned(), provider)]),
            ..Config::default()
        };

        assert!(config.provider_for_check("disabled").is_ok());
    }

    #[test]
    fn remote_compaction_models_must_be_selected_base_models() {
        let provider = ProviderConfig {
            base_url: Url::parse("https://example.com/v1").unwrap(),
            api_key: Some("secret".into()),
            api_key_env: None,
            enabled: true,
            models: vec!["sol".into()],
            chat_models: Vec::new(),
            remote_compaction_models: vec!["terra".into()],
        };
        let config = Config {
            providers: BTreeMap::from([("example".to_owned(), provider)]),
            ..Config::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn chat_models_must_be_selected_base_models() {
        let provider = ProviderConfig {
            base_url: Url::parse("https://example.com/v1").unwrap(),
            api_key: Some("secret".into()),
            api_key_env: None,
            enabled: true,
            models: vec!["sol".into()],
            chat_models: vec!["terra".into()],
            remote_compaction_models: Vec::new(),
        };
        let config = Config {
            providers: BTreeMap::from([("example".to_owned(), provider)]),
            ..Config::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn server_listeners_must_be_distinct() {
        let provider = ProviderConfig {
            base_url: Url::parse("https://example.com/v1").unwrap(),
            api_key: Some("secret".into()),
            api_key_env: None,
            enabled: true,
            models: vec!["sol".into()],
            chat_models: Vec::new(),
            remote_compaction_models: Vec::new(),
        };
        let mut config = Config {
            providers: BTreeMap::from([("example".to_owned(), provider)]),
            ..Config::default()
        };
        config.server.openai_compact_listen = config.server.listen;
        assert!(config.validate().is_err());
    }
}
