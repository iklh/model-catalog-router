use crate::catalog::{
    configured_models, configured_openai_compact_models, discover_provider_models,
    DiscoveredProviderModels, OPENAI_COMPACT_SUFFIX,
};
use crate::config::{
    default_web_search_mcp_url, validate_provider_name, validate_web_search_mcp_url, Config,
    ProviderConfig, RoutingConfig, WebSearchConfig,
};
use crate::sync;
use anyhow::{bail, Context, Result};
use std::collections::BTreeSet;
use std::io::{self, Write};
use std::path::Path;
use url::Url;

pub async fn run(config_path: &Path) -> Result<()> {
    let mut config = Config::load_or_default(config_path)?;
    loop {
        print_menu(&config);
        let choice = prompt("e/n/d/r/m/t/l/u/w/q> ", None)?.to_ascii_lowercase();
        if choice == "q" {
            return Ok(());
        }
        let result = match choice.as_str() {
            "e" => edit_provider(&config).await,
            "n" => new_provider(&config).await,
            "d" => delete_provider(&config),
            "r" => refresh_provider(&config).await,
            "m" => edit_provider_models(&config),
            "t" => toggle_provider(&config),
            "u" => configure_unprefixed_model_provider(&config),
            "w" => configure_web_search(&config),
            "l" => {
                show_provider_models(&config)?;
                continue;
            }
            "" => continue,
            _ => {
                println!("Unknown choice. Enter e, n, d, r, m, t, l, u, w, or q.");
                continue;
            }
        };
        match result {
            Ok(updated) => {
                if let Err(error) = apply_change(&mut config, config_path, updated).await {
                    println!("Configuration was not changed: {error:#}");
                }
            }
            Err(error) => println!("Configuration was not changed: {error:#}"),
        }
    }
}

async fn apply_change(
    config: &mut Config,
    config_path: &Path,
    updated: Option<Config>,
) -> Result<()> {
    let Some(updated) = updated else {
        return Ok(());
    };
    updated.validate_for_save()?;
    let selected_models = profile_models_if_replacement_needed(&updated)?;
    updated.save(config_path)?;
    match sync::sync_files(
        &updated,
        config_path,
        selected_models.base.as_deref(),
        selected_models.openai_compact.as_deref(),
    ) {
        Ok(result) => {
            println!("Saved {}", config_path.display());
            println!("Updated {}", result.base.catalog_path.display());
            println!(
                "Updated {} (model: {})",
                result.base.profile_path.display(),
                result.base.model
            );
            if let Some(compact) = result.openai_compact {
                println!("Updated {}", compact.catalog_path.display());
                println!(
                    "Updated {} (model: {})",
                    compact.profile_path.display(),
                    compact.model
                );
            } else {
                println!("Remote compaction profile is not configured.");
            }
            *config = updated;
            Ok(())
        }
        Err(error) => {
            println!("Configuration was saved, but Codex synchronization failed: {error:#}");
            println!(
                "Fix the problem and make another configuration change to retry synchronization."
            );
            *config = updated;
            Ok(())
        }
    }
}

fn print_menu(config: &Config) {
    println!("\nCurrent providers:\n");
    if config.providers.is_empty() {
        println!("  No providers configured.");
    } else {
        println!("  #  Name                         Models  Enabled");
        println!("  -  ----                         ------  -------");
        for (index, (name, provider)) in config.providers.iter().enumerate() {
            println!(
                "  {:<2} {:<28} {:>6}  {}",
                index + 1,
                name,
                provider.models.len(),
                if provider.enabled { "yes" } else { "no" }
            );
        }
    }
    println!(
        "\nWeb Search MCP URL: {}",
        config
            .web_search
            .as_ref()
            .map(|web_search| web_search.mcp_url.as_str())
            .unwrap_or("http://127.0.0.1:9091/mcp (default)")
    );
    println!(
        "Unprefixed model provider: {}",
        config
            .routing
            .as_ref()
            .map(|routing| routing.unprefixed_model_provider.as_str())
            .unwrap_or("disabled")
    );
    println!("\ne) Edit existing provider");
    println!("n) New provider");
    println!("d) Delete provider");
    println!("r) Refresh provider models");
    println!("m) Edit provider models");
    println!("t) Toggle provider enabled");
    println!("l) List provider models");
    println!("u) Configure unprefixed model provider");
    println!("w) Configure Web Search MCP URL");
    println!("q) Quit configuration\n");
}

fn configure_unprefixed_model_provider(config: &Config) -> Result<Option<Config>> {
    let names = config
        .providers
        .iter()
        .filter(|(_, provider)| provider.enabled)
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    if names.is_empty() {
        println!("No enabled providers are configured.");
        return Ok(None);
    }

    println!("\nUnprefixed model provider:");
    println!("0) Disabled");
    for (index, name) in names.iter().enumerate() {
        let current = config
            .routing
            .as_ref()
            .is_some_and(|routing| routing.unprefixed_model_provider == *name);
        println!(
            "{}) {}{}",
            index + 1,
            name,
            if current { " (current)" } else { "" }
        );
    }
    let input = prompt("Number (Enter to cancel)", None)?;
    if input.is_empty() {
        return Ok(None);
    }
    let index = input
        .parse::<usize>()
        .context("provider choice must be a number")?;
    let mut updated = config.clone();
    if index == 0 {
        updated.routing = None;
    } else {
        let name = names
            .get(index - 1)
            .context("provider choice is out of range")?;
        updated.routing = Some(RoutingConfig {
            unprefixed_model_provider: name.clone(),
        });
    }
    Ok(Some(updated))
}

fn configure_web_search(config: &Config) -> Result<Option<Config>> {
    let mut updated = config.clone();
    let default_url = default_web_search_mcp_url();
    let current = config
        .web_search
        .as_ref()
        .map(|web_search| &web_search.mcp_url)
        .unwrap_or(&default_url);
    let mcp_url = prompt_web_search_mcp_url(current)?;
    updated.web_search = Some(WebSearchConfig { mcp_url });
    Ok(Some(updated))
}

fn prompt_web_search_mcp_url(current: &Url) -> Result<Url> {
    loop {
        let input = prompt("Web Search MCP URL", Some(current.as_str()))?;
        match Url::parse(&input) {
            Ok(url) if validate_web_search_mcp_url(&url).is_ok() => return Ok(url),
            _ => println!(
                "Web Search MCP URL must be an http or https URL with a host and no fragment."
            ),
        }
    }
}

fn show_provider_models(config: &Config) -> Result<()> {
    print_provider_models(config);
    println!();
    loop {
        match prompt("b) Back to menu", Some("b"))?
            .to_ascii_lowercase()
            .as_str()
        {
            "b" => return Ok(()),
            _ => println!("Unknown choice. Enter b to return to the menu."),
        }
    }
}

fn print_provider_models(config: &Config) {
    println!("\nCurrent providers with model names:\n");
    if config.providers.is_empty() {
        println!("  No providers configured.");
        return;
    }

    let rows = config
        .providers
        .iter()
        .map(|(name, provider)| {
            let models = provider.models.join(", ");
            (name, provider, models)
        })
        .collect::<Vec<_>>();
    let model_list_width = rows
        .iter()
        .map(|(_, _, models)| models.len())
        .max()
        .unwrap_or_default()
        .max("Model list".len());

    println!(
        "  #  Name                         Models  {:<model_list_width$}  Enabled",
        "Model list"
    );
    println!(
        "  -  ----                         ------  {:<model_list_width$}  -------",
        "----------"
    );
    for (index, (name, provider, models)) in rows.iter().enumerate() {
        println!(
            "  {:<2} {:<28} {:>6}  {:<model_list_width$}  {}",
            index + 1,
            name,
            provider.models.len(),
            models,
            if provider.enabled { "yes" } else { "no" }
        );
    }
}

async fn new_provider(config: &Config) -> Result<Option<Config>> {
    let mut candidate: Option<(String, ProviderConfig)> = None;
    loop {
        let Some((name, provider)) = provider_wizard(config, candidate.as_ref(), false).await?
        else {
            return Ok(None);
        };
        print_provider_summary(&name, &provider);
        match prompt("y) Save  e) Edit  c) Cancel\ny/e/c> ", Some("y"))?
            .to_ascii_lowercase()
            .as_str()
        {
            "y" => {
                let mut updated = config.clone();
                updated.providers.insert(name, provider);
                return Ok(Some(updated));
            }
            "e" => candidate = Some((name, provider)),
            "c" => return Ok(None),
            _ => println!("Unknown choice."),
        }
    }
}

async fn edit_provider(config: &Config) -> Result<Option<Config>> {
    let Some(name) = choose_provider(config, "Provider to edit")? else {
        return Ok(None);
    };
    let original = config
        .providers
        .get(&name)
        .expect("selected provider")
        .clone();
    let mut candidate = (name.clone(), original);
    loop {
        let Some(updated_candidate) = provider_wizard(config, Some(&candidate), true).await? else {
            return Ok(None);
        };
        candidate = updated_candidate;
        print_provider_summary(&candidate.0, &candidate.1);
        match prompt("y) Save  e) Edit again  c) Cancel\ny/e/c> ", Some("y"))?
            .to_ascii_lowercase()
            .as_str()
        {
            "y" => {
                let mut updated = config.clone();
                updated.providers.insert(name, candidate.1);
                return Ok(Some(updated));
            }
            "e" => {}
            "c" => return Ok(None),
            _ => println!("Unknown choice."),
        }
    }
}

async fn provider_wizard(
    config: &Config,
    existing: Option<&(String, ProviderConfig)>,
    lock_name: bool,
) -> Result<Option<(String, ProviderConfig)>> {
    let existing_name = existing.map(|(name, _)| name.as_str());
    let name = if lock_name {
        existing_name
            .context("missing existing provider name")?
            .to_owned()
    } else {
        loop {
            let name = prompt("Provider name", existing_name)?;
            if let Err(error) = validate_provider_name(&name, &config.catalog.separator) {
                println!("Invalid provider name: {error}");
                continue;
            }
            let belongs_to_current = existing_name == Some(name.as_str());
            if config.providers.contains_key(&name) && !belongs_to_current {
                println!("Provider `{name}` already exists.");
                continue;
            }
            break name;
        }
    };

    let current = existing.map(|(_, provider)| provider);
    let base_url = prompt_base_url(current.map(|provider| &provider.base_url))?;
    let (api_key, api_key_env) = prompt_credentials(current)?;
    let enabled = prompt_yes_no(
        "Enabled",
        current.map(|provider| provider.enabled).unwrap_or(true),
    )?;
    let proxy_url = prompt_proxy(current.and_then(|provider| provider.proxy_url.as_ref()))?;
    let mut provider = ProviderConfig {
        base_url,
        proxy_url,
        api_key,
        api_key_env,
        enabled,
        models: current
            .map(|provider| provider.models.clone())
            .unwrap_or_default(),
        chat_models: current
            .map(|provider| provider.chat_models.clone())
            .unwrap_or_default(),
        messages_models: current
            .map(|provider| provider.messages_models.clone())
            .unwrap_or_default(),
        remote_compaction_models: current
            .map(|provider| provider.remote_compaction_models.clone())
            .unwrap_or_default(),
    };

    let configure_models = current.is_none() || prompt_yes_no("Change model selection", false)?;
    if configure_models {
        if prompt_yes_no("Automatically discover models", true)? {
            loop {
                match discover_models_with_recovery(&name, &provider, true).await? {
                    DiscoveryOutcome::Discovered(discovered) => {
                        let current_models = current.map(|provider| provider.models.as_slice());
                        match select_discovered_models(
                            &discovered.models,
                            current_models,
                            current.is_none(),
                        )? {
                            ModelSelection::Discovered(models) => {
                                provider.models = models;
                                provider.remote_compaction_models = configure_remote_compaction(
                                    &discovered,
                                    &provider.models,
                                    current.map(|provider| {
                                        provider.remote_compaction_models.as_slice()
                                    }),
                                )?;
                            }
                            ModelSelection::Manual(models) => {
                                provider.models = models;
                                provider.remote_compaction_models.clear();
                            }
                        }
                        provider.chat_models = select_chat_models(
                            &provider.models,
                            current_models,
                            current.map(|provider| provider.chat_models.as_slice()),
                        )?;
                        provider.messages_models = select_messages_models(
                            &provider.models,
                            &provider.chat_models,
                            current.map(|provider| provider.models.as_slice()),
                            current.map(|provider| provider.messages_models.as_slice()),
                        )?;
                        break;
                    }
                    DiscoveryOutcome::EditConnection => {
                        edit_connection_details(&mut provider)?;
                    }
                    DiscoveryOutcome::Manual => {
                        provider.models = manual_models()?;
                        provider.chat_models = select_chat_models(
                            &provider.models,
                            current.map(|provider| provider.models.as_slice()),
                            current.map(|provider| provider.chat_models.as_slice()),
                        )?;
                        provider.messages_models = select_messages_models(
                            &provider.models,
                            &provider.chat_models,
                            current.map(|provider| provider.models.as_slice()),
                            current.map(|provider| provider.messages_models.as_slice()),
                        )?;
                        provider.remote_compaction_models.clear();
                        break;
                    }
                    DiscoveryOutcome::Cancel => return Ok(None),
                }
            }
        } else {
            provider.models = manual_models()?;
            provider.chat_models = select_chat_models(
                &provider.models,
                current.map(|provider| provider.models.as_slice()),
                current.map(|provider| provider.chat_models.as_slice()),
            )?;
            provider.messages_models = select_messages_models(
                &provider.models,
                &provider.chat_models,
                current.map(|provider| provider.models.as_slice()),
                current.map(|provider| provider.messages_models.as_slice()),
            )?;
            provider.remote_compaction_models.clear();
        }
    } else if prompt_yes_no("Change Chat compatibility selection", false)? {
        provider.chat_models = select_chat_models(
            &provider.models,
            Some(&provider.models),
            Some(&provider.chat_models),
        )?;
        provider.messages_models = select_messages_models(
            &provider.models,
            &provider.chat_models,
            Some(&provider.models),
            Some(&provider.messages_models),
        )?;
    }
    if provider.models.is_empty() {
        bail!("at least one model must be selected");
    }
    Ok(Some((name, provider)))
}

fn prompt_base_url(current: Option<&Url>) -> Result<Url> {
    loop {
        let input = prompt("Base URL", current.map(Url::as_str))?;
        match Url::parse(&input) {
            Ok(url) if url.scheme() == "http" || url.scheme() == "https" => return Ok(url),
            _ => println!("Base URL must be a valid http or https URL."),
        }
    }
}

fn edit_connection_details(provider: &mut ProviderConfig) -> Result<()> {
    provider.base_url = prompt_base_url(Some(&provider.base_url))?;
    let (api_key, api_key_env) = prompt_credentials(Some(provider))?;
    provider.api_key = api_key;
    provider.api_key_env = api_key_env;
    provider.proxy_url = prompt_proxy(provider.proxy_url.as_ref())?;
    Ok(())
}

fn prompt_proxy(current: Option<&Url>) -> Result<Option<Url>> {
    if let Some(current) = current {
        println!("\nProxy:");
        println!("1) Keep current ({current})");
        println!("2) No proxy");
        println!("3) Custom proxy URL");
        return match prompt("Choice", Some("1"))?.as_str() {
            "1" => Ok(Some(current.clone())),
            "2" => Ok(None),
            "3" => prompt_custom_proxy(),
            _ => bail!("invalid proxy choice"),
        };
    }

    println!("\nProxy:");
    println!("1) No proxy");
    println!("2) Custom proxy URL");
    match prompt("Choice", Some("1"))?.as_str() {
        "1" => Ok(None),
        "2" => prompt_custom_proxy(),
        _ => bail!("invalid proxy choice"),
    }
}

fn prompt_custom_proxy() -> Result<Option<Url>> {
    loop {
        let input = prompt("Proxy URL", None)?;
        match Url::parse(&input) {
            Ok(url)
                if matches!(url.scheme(), "http" | "https" | "socks5" | "socks5h")
                    && url.host_str().is_some() =>
            {
                return Ok(Some(url));
            }
            _ => println!("Proxy URL must be a valid http, https, socks5, or socks5h URL."),
        }
    }
}

fn prompt_credentials(
    current: Option<&ProviderConfig>,
) -> Result<(Option<String>, Option<String>)> {
    if let Some(current) = current {
        println!("\nAPI key source:");
        println!("1) Keep current ({})", credential_summary(current));
        println!("2) API key");
        println!("3) Environment variable");
        match prompt("Choice", Some("1"))?.as_str() {
            "1" => return Ok((current.api_key.clone(), current.api_key_env.clone())),
            "2" => return Ok((Some(prompt_api_key()?), None)),
            "3" => return Ok((None, Some(prompt_required("Environment variable")?))),
            _ => bail!("invalid API key source choice"),
        }
    }

    println!("\nAPI key source:");
    println!("1) API key");
    println!("2) Environment variable");
    match prompt("Choice", Some("1"))?.as_str() {
        "1" => Ok((Some(prompt_api_key()?), None)),
        "2" => Ok((None, Some(prompt_required("Environment variable")?))),
        _ => bail!("invalid API key source choice"),
    }
}

enum DiscoveryOutcome {
    Discovered(DiscoveredProviderModels),
    EditConnection,
    Manual,
    Cancel,
}

async fn discover_models_with_recovery(
    name: &str,
    provider: &ProviderConfig,
    allow_manual: bool,
) -> Result<DiscoveryOutcome> {
    loop {
        println!("Discovering models from `{name}`...");
        match discover_provider_models(name, provider).await {
            Ok(models) => return Ok(DiscoveryOutcome::Discovered(models)),
            Err(error) => {
                println!("\nModel discovery failed:\n{error:#}\n");
                println!("r) Retry");
                println!("e) Edit Base URL and API key");
                if allow_manual {
                    println!("m) Enter models manually");
                }
                println!("c) Cancel");
                let choices = if allow_manual { "r/e/m/c" } else { "r/e/c" };
                loop {
                    match prompt(choices, Some("r"))?.to_ascii_lowercase().as_str() {
                        "r" => break,
                        "e" => return Ok(DiscoveryOutcome::EditConnection),
                        "m" if allow_manual => return Ok(DiscoveryOutcome::Manual),
                        "c" => return Ok(DiscoveryOutcome::Cancel),
                        _ => println!("Unknown choice."),
                    }
                }
            }
        }
    }
}

fn edit_provider_models(config: &Config) -> Result<Option<Config>> {
    let Some(name) = choose_provider(config, "Provider whose models to edit")? else {
        return Ok(None);
    };
    let mut provider = config
        .providers
        .get(&name)
        .expect("selected provider")
        .clone();
    let original = provider.clone();

    loop {
        let changed = provider_model_selections_changed(&provider, &original);
        print_model_editor_list(&name, &provider, changed);
        let prompt_label = if changed {
            "Number (Enter to save)"
        } else {
            "Number (Enter to back)"
        };
        let choice = prompt(prompt_label, None)?.to_ascii_lowercase();
        match choice.as_str() {
            "" => {
                if changed {
                    let mut updated = config.clone();
                    updated.providers.insert(name, provider);
                    return Ok(Some(updated));
                }
                return Ok(None);
            }
            "n" => {
                let model = prompt_required("New model name")?;
                let default_api = if recommend_chat_protocol(&model) {
                    ModelApi::ChatCompletions
                } else {
                    ModelApi::Responses
                };
                let api = prompt_model_api(default_api)?;
                add_model_to_provider(&mut provider, &model, api)?;
            }
            "s" if changed => {
                let mut updated = config.clone();
                updated.providers.insert(name, provider);
                return Ok(Some(updated));
            }
            "r" if changed => return Ok(None),
            "b" if !changed => return Ok(None),
            _ => {
                let index = choice
                    .parse::<usize>()
                    .context("model choice must be a number")?;
                let old_model = provider
                    .models
                    .get(index.saturating_sub(1))
                    .cloned()
                    .context("model choice is out of range")?;
                edit_provider_model(&mut provider, &old_model)?;
            }
        }
    }
}

fn print_model_editor_list(name: &str, provider: &ProviderConfig, changed: bool) {
    println!("\nModels for provider `{name}`:\n");
    if provider.models.is_empty() {
        println!("  No models configured.");
    } else {
        for (index, model) in provider.models.iter().enumerate() {
            println!(
                "{:>3}) {} ({})",
                index + 1,
                model,
                model_api(provider, model).label()
            );
        }
    }
    println!("\nn) Add model manually");
    if changed {
        println!("s) Save and return");
        println!("r) Discard changes and return");
    } else {
        println!("b) Back");
    }
}

fn provider_model_selections_changed(provider: &ProviderConfig, original: &ProviderConfig) -> bool {
    !same_model_selection(&provider.models, &original.models)
        || !same_model_selection(&provider.chat_models, &original.chat_models)
        || !same_model_selection(&provider.messages_models, &original.messages_models)
        || !same_model_selection(
            &provider.remote_compaction_models,
            &original.remote_compaction_models,
        )
}

fn same_model_selection(left: &[String], right: &[String]) -> bool {
    let mut left = left.to_vec();
    let mut right = right.to_vec();
    left.sort();
    right.sort();
    left == right
}

fn edit_provider_model(provider: &mut ProviderConfig, old_model: &str) -> Result<()> {
    let current_api = model_api(provider, old_model);
    println!("\nEditing model: {old_model}");
    println!("Current API: {}\n", current_api.label());
    println!("n) Rename model");
    println!("a) Change model API");
    println!("d) Delete model");
    println!("b) Back\n");
    let choice = prompt("Choice", Some("b"))?.to_ascii_lowercase();
    match choice.as_str() {
        "n" => {
            let new_model = prompt("New model name", Some(old_model))?;
            rename_model_in_provider(provider, old_model, &new_model)?;
        }
        "a" => {
            let api = prompt_model_api(current_api)?;
            set_model_api(provider, old_model, api)?;
        }
        "d" => {
            if prompt_yes_no(&format!("Delete model `{old_model}`"), false)? {
                remove_model_from_provider(provider, old_model)?;
            }
        }
        "b" => {}
        _ => println!("Unknown choice. Enter n, a, d, or b."),
    }
    Ok(())
}

fn prompt_model_api(default: ModelApi) -> Result<ModelApi> {
    println!("\nModel API:");
    for (index, api) in ModelApi::ALL.iter().enumerate() {
        println!(
            "{}) {}{}",
            index + 1,
            api.label(),
            if *api == default { " (current)" } else { "" }
        );
    }
    let default_choice = default.choice_number().to_string();
    let choice = prompt("Choice", Some(&default_choice))?;
    let index = choice
        .parse::<usize>()
        .context("API choice must be a number")?;
    ModelApi::ALL
        .get(index.saturating_sub(1))
        .copied()
        .context("API choice is out of range")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelApi {
    Responses,
    ChatCompletions,
    AnthropicMessages,
}

impl ModelApi {
    const ALL: [Self; 3] = [
        Self::Responses,
        Self::ChatCompletions,
        Self::AnthropicMessages,
    ];

    fn choice_number(self) -> usize {
        Self::ALL
            .iter()
            .position(|candidate| *candidate == self)
            .expect("known API has a menu choice")
            + 1
    }

    fn label(self) -> &'static str {
        match self {
            Self::Responses => "OpenAI Responses",
            Self::ChatCompletions => "Chat Completions",
            Self::AnthropicMessages => "Anthropic Messages",
        }
    }
}

fn model_api(provider: &ProviderConfig, model: &str) -> ModelApi {
    if provider
        .chat_models
        .iter()
        .any(|candidate| candidate == model)
    {
        ModelApi::ChatCompletions
    } else if provider
        .messages_models
        .iter()
        .any(|candidate| candidate == model)
    {
        ModelApi::AnthropicMessages
    } else {
        ModelApi::Responses
    }
}

fn validate_model_name(model: &str) -> Result<()> {
    if model.trim().is_empty() {
        bail!("model name must not be empty");
    }
    if model.ends_with(OPENAI_COMPACT_SUFFIX) {
        bail!("model name must not end with the reserved `{OPENAI_COMPACT_SUFFIX}` suffix");
    }
    Ok(())
}

fn add_model_to_provider(provider: &mut ProviderConfig, model: &str, api: ModelApi) -> Result<()> {
    validate_model_name(model)?;
    if provider.models.iter().any(|candidate| candidate == model) {
        bail!("model `{model}` already exists");
    }
    provider.models.push(model.to_owned());
    provider.models.sort();
    match api {
        ModelApi::Responses => {}
        ModelApi::ChatCompletions => {
            provider.chat_models.push(model.to_owned());
            provider.chat_models.sort();
        }
        ModelApi::AnthropicMessages => {
            provider.messages_models.push(model.to_owned());
            provider.messages_models.sort();
        }
    }
    Ok(())
}

fn rename_model_in_provider(
    provider: &mut ProviderConfig,
    old_model: &str,
    new_model: &str,
) -> Result<()> {
    validate_model_name(new_model)?;
    if old_model == new_model {
        return Ok(());
    }
    if provider
        .models
        .iter()
        .any(|candidate| candidate != old_model && candidate == new_model)
    {
        bail!("model `{new_model}` already exists");
    }

    let api = model_api(provider, old_model);
    replace_or_remove_model(&mut provider.models, old_model, Some(new_model));
    replace_or_remove_model(&mut provider.chat_models, old_model, None);
    replace_or_remove_model(&mut provider.messages_models, old_model, None);
    replace_or_remove_model(
        &mut provider.remote_compaction_models,
        old_model,
        Some(new_model),
    );
    match api {
        ModelApi::Responses => {}
        ModelApi::ChatCompletions => {
            provider.chat_models.push(new_model.to_owned());
            provider.chat_models.sort();
        }
        ModelApi::AnthropicMessages => {
            provider.messages_models.push(new_model.to_owned());
            provider.messages_models.sort();
        }
    }
    Ok(())
}

fn set_model_api(provider: &mut ProviderConfig, model: &str, api: ModelApi) -> Result<()> {
    if !provider.models.iter().any(|candidate| candidate == model) {
        bail!("model `{model}` does not exist");
    }
    provider.chat_models.retain(|candidate| candidate != model);
    provider
        .messages_models
        .retain(|candidate| candidate != model);
    match api {
        ModelApi::Responses => {}
        ModelApi::ChatCompletions => {
            provider.chat_models.push(model.to_owned());
            provider.chat_models.sort();
        }
        ModelApi::AnthropicMessages => {
            provider.messages_models.push(model.to_owned());
            provider.messages_models.sort();
        }
    }
    Ok(())
}

fn remove_model_from_provider(provider: &mut ProviderConfig, model: &str) -> Result<()> {
    if provider.models.len() <= 1 {
        bail!("provider must retain at least one model");
    }
    if !provider.models.iter().any(|candidate| candidate == model) {
        bail!("model `{model}` does not exist");
    }
    replace_or_remove_model(&mut provider.models, model, None);
    replace_or_remove_model(&mut provider.chat_models, model, None);
    replace_or_remove_model(&mut provider.messages_models, model, None);
    replace_or_remove_model(&mut provider.remote_compaction_models, model, None);
    Ok(())
}

fn replace_or_remove_model(models: &mut Vec<String>, old_model: &str, new_model: Option<&str>) {
    models.retain(|candidate| candidate != old_model);
    if let Some(new_model) = new_model {
        models.push(new_model.to_owned());
        models.sort();
    }
}

async fn refresh_provider(config: &Config) -> Result<Option<Config>> {
    let Some(name) = choose_provider(config, "Provider to refresh")? else {
        return Ok(None);
    };
    let mut provider = config
        .providers
        .get(&name)
        .expect("selected provider")
        .clone();
    let (discovered, selection) = loop {
        match discover_models_with_recovery(&name, &provider, false).await? {
            DiscoveryOutcome::Discovered(models) => {
                let selection =
                    select_discovered_models(&models.models, Some(&provider.models), false)?;
                break (models, selection);
            }
            DiscoveryOutcome::EditConnection => edit_connection_details(&mut provider)?,
            DiscoveryOutcome::Cancel => return Ok(None),
            DiscoveryOutcome::Manual => unreachable!("manual entry is disabled during refresh"),
        }
    };
    let (selected, remote_compaction_models) = match selection {
        ModelSelection::Discovered(selected) => {
            let remote_compaction_models = configure_remote_compaction(
                &discovered,
                &selected,
                Some(&provider.remote_compaction_models),
            )?;
            (selected, remote_compaction_models)
        }
        ModelSelection::Manual(selected) => (selected, Vec::new()),
    };
    if selected.is_empty() {
        bail!("at least one model must be selected");
    }
    let chat_models = select_chat_models(
        &selected,
        Some(&provider.models),
        Some(&provider.chat_models),
    )?;
    let messages_models = select_messages_models(
        &selected,
        &chat_models,
        Some(&provider.models),
        Some(&provider.messages_models),
    )?;
    let mut updated = config.clone();
    provider.models = selected;
    provider.chat_models = chat_models;
    provider.messages_models = messages_models;
    provider.remote_compaction_models = remote_compaction_models;
    updated.providers.insert(name, provider);
    Ok(Some(updated))
}

fn select_messages_models(
    selected_models: &[String],
    chat_models: &[String],
    previous_models: Option<&[String]>,
    previous_messages_models: Option<&[String]>,
) -> Result<Vec<String>> {
    let candidates = selected_models
        .iter()
        .filter(|model| !chat_models.contains(model))
        .cloned()
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return Ok(Vec::new());
    }
    let previous_models = previous_models.unwrap_or_default();
    let previous = previous_messages_models.unwrap_or_default();
    let defaults = candidates
        .iter()
        .enumerate()
        .filter_map(|(i, model)| {
            let selected = if previous_models.contains(model) {
                previous.contains(model)
            } else {
                false
            };
            selected.then_some(i + 1)
        })
        .collect::<Vec<_>>();
    println!("\nAnthropic Messages compatibility models:\n");
    for (index, model) in candidates.iter().enumerate() {
        println!(
            "{:>3}) [{}] {}",
            index + 1,
            if defaults.contains(&(index + 1)) {
                "x"
            } else {
                " "
            },
            model
        );
    }
    println!("\nModels marked [x] will use Anthropic Messages compatibility.");
    println!(
        "Press Enter to keep the marked models, or enter a selection such as 1,3, all, or none."
    );
    let input = prompt("Select Messages models", Some("keep marked"))?;
    let indices = if input.eq_ignore_ascii_case("none") {
        Vec::new()
    } else {
        resolve_model_selection(&input, defaults, candidates.len())?
    };
    Ok(indices
        .into_iter()
        .map(|index| candidates[index - 1].clone())
        .collect())
}

fn configure_remote_compaction(
    discovered: &DiscoveredProviderModels,
    selected_models: &[String],
    current: Option<&[String]>,
) -> Result<Vec<String>> {
    let candidates = discovered
        .remote_compaction_models
        .iter()
        .filter(|model| selected_models.contains(model))
        .cloned()
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    let current_candidates = current
        .unwrap_or_default()
        .iter()
        .filter(|model| candidates.contains(model))
        .cloned()
        .collect::<Vec<_>>();
    if !prompt_yes_no(
        "Enable remote compaction for supported selected models",
        !current_candidates.is_empty(),
    )? {
        return Ok(Vec::new());
    }

    println!(
        "\nThe Provider advertises matching `{}` aliases for these base models.",
        crate::catalog::OPENAI_COMPACT_SUFFIX
    );
    println!("Only the generated `mc-router-openai-compact` profile will use them.");
    select_models(
        &candidates,
        (!current_candidates.is_empty()).then_some(current_candidates.as_slice()),
        current_candidates.is_empty(),
    )
}

enum ModelSelection {
    Discovered(Vec<String>),
    Manual(Vec<String>),
}

fn select_discovered_models(
    discovered: &[String],
    current: Option<&[String]>,
    default_all: bool,
) -> Result<ModelSelection> {
    if discovered.len() <= 10 {
        return Ok(ModelSelection::Discovered(select_models(
            discovered,
            current,
            default_all,
        )?));
    }

    println!("\nDetected {} models.", discovered.len());
    if prompt_yes_no("Show all discovered models", false)? {
        return Ok(ModelSelection::Discovered(select_models(
            discovered,
            current,
            default_all,
        )?));
    }

    println!("\n1) Search models");
    println!("2) Enter models manually");
    loop {
        match prompt("Choice", Some("1"))?.as_str() {
            "1" => {
                return Ok(ModelSelection::Discovered(search_models(
                    discovered, current,
                )?))
            }
            "2" => return Ok(ModelSelection::Manual(manual_models()?)),
            _ => println!("Unknown choice. Enter 1 or 2."),
        }
    }
}

fn search_models(discovered: &[String], current: Option<&[String]>) -> Result<Vec<String>> {
    let mut selected = current
        .unwrap_or_default()
        .iter()
        .filter(|model| discovered.contains(*model))
        .cloned()
        .collect::<BTreeSet<_>>();

    loop {
        let keyword = prompt_search_keyword()?;
        let keyword = keyword.to_ascii_lowercase();
        let matches = discovered
            .iter()
            .filter(|model| !selected.contains(*model))
            .filter(|model| model.to_ascii_lowercase().contains(&keyword))
            .cloned()
            .collect::<Vec<_>>();
        if matches.is_empty() {
            println!("No matching models. Try another keyword.");
            continue;
        }

        println!(
            "\n{} matching models; showing the first {}:\n",
            matches.len(),
            matches.len().min(10)
        );
        let displayed = &matches[..matches.len().min(10)];
        for (index, model) in displayed.iter().enumerate() {
            println!("{:>3}) {}", index + 1, model);
        }
        println!("\nEnter a selection such as 1,3-5, all, or none.");
        let input = prompt("Select models", None)?;
        let indices = if input.is_empty() || input.eq_ignore_ascii_case("none") {
            Vec::new()
        } else {
            resolve_model_selection(&input, Vec::new(), displayed.len())?
        };
        selected.extend(
            indices
                .into_iter()
                .map(|index| displayed[index - 1].clone()),
        );

        if !prompt_yes_no("Continue filtering models", false)? {
            return Ok(discovered
                .iter()
                .filter(|model| selected.contains(*model))
                .cloned()
                .collect());
        }
    }
}

fn prompt_search_keyword() -> Result<String> {
    loop {
        let keyword = remove_whitespace(&prompt("Search keyword", None)?);
        if !keyword.is_empty() {
            return Ok(keyword);
        }
    }
}

fn remove_whitespace(input: &str) -> String {
    input
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect()
}

fn select_models(
    discovered: &[String],
    current: Option<&[String]>,
    default_all: bool,
) -> Result<Vec<String>> {
    let current = current.unwrap_or_default().iter().collect::<BTreeSet<_>>();
    println!("\nAvailable models:\n");
    for (index, model) in discovered.iter().enumerate() {
        let selected = default_all || current.contains(model);
        println!(
            "{:>3}) [{}] {}",
            index + 1,
            if selected { "x" } else { " " },
            model
        );
    }
    let default_indices = discovered
        .iter()
        .enumerate()
        .filter_map(|(index, model)| (default_all || current.contains(model)).then_some(index + 1))
        .collect::<Vec<_>>();
    if default_all {
        println!("\nAll discovered models are selected by default.");
        println!("Press Enter to keep all, or enter a complete selection such as 1,3-5 or all.");
    } else {
        let missing = current
            .iter()
            .filter(|model| !discovered.contains(*model))
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            println!("\nPreviously selected models no longer advertised by the Provider:");
            for model in missing {
                println!("  - {model}");
            }
            println!("These models will be removed if you continue.");
        }
        println!("\nPreviously selected models are marked [x].");
        println!(
            "Press Enter to keep the marked models, or enter a complete selection such as 1,3 or all."
        );
    }
    let input = prompt(
        "Select models",
        Some(if default_all { "all" } else { "keep marked" }),
    )?;
    let indices = resolve_model_selection(&input, default_indices, discovered.len())?;
    Ok(indices
        .into_iter()
        .map(|index| discovered[index - 1].clone())
        .collect())
}

fn select_chat_models(
    selected_models: &[String],
    previous_models: Option<&[String]>,
    previous_chat_models: Option<&[String]>,
) -> Result<Vec<String>> {
    let previous_models = previous_models.unwrap_or_default();
    let previous_chat_models = previous_chat_models.unwrap_or_default();
    let default_indices =
        default_chat_model_indices(selected_models, previous_models, previous_chat_models);

    println!("\nChat compatibility models:\n");
    for (index, model) in selected_models.iter().enumerate() {
        println!(
            "{:>3}) [{}] {}",
            index + 1,
            if default_indices.contains(&(index + 1)) {
                "x"
            } else {
                " "
            },
            model
        );
    }
    println!("\nModels marked [x] will use Chat Completions compatibility.");
    println!("GPT, o-series, Codex, Claude, Gemini, and Grok model names are excluded by default.");
    println!(
        "Press Enter to keep the marked models, or enter a complete selection such as 1,3-5, all, or none."
    );
    let input = prompt("Select Chat models", Some("keep marked"))?;
    let indices = if input.eq_ignore_ascii_case("none") {
        Vec::new()
    } else {
        resolve_model_selection(&input, default_indices, selected_models.len())?
    };
    Ok(indices
        .into_iter()
        .map(|index| selected_models[index - 1].clone())
        .collect())
}

fn default_chat_model_indices(
    selected_models: &[String],
    previous_models: &[String],
    previous_chat_models: &[String],
) -> Vec<usize> {
    selected_models
        .iter()
        .enumerate()
        .filter_map(|(index, model)| {
            let selected = if previous_models.contains(model) {
                previous_chat_models.contains(model)
            } else {
                recommend_chat_protocol(model)
            };
            selected.then_some(index + 1)
        })
        .collect()
}

fn recommend_chat_protocol(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    let model = model.rsplit('/').next().unwrap_or(&model);
    !["gpt", "o1", "o3", "o4", "codex", "claude", "gemini", "grok"]
        .iter()
        .any(|prefix| model.starts_with(prefix))
}

fn resolve_model_selection(
    input: &str,
    default_indices: Vec<usize>,
    maximum: usize,
) -> Result<Vec<usize>> {
    if input.is_empty() || input.eq_ignore_ascii_case("keep marked") {
        Ok(default_indices)
    } else if input.eq_ignore_ascii_case("all") {
        Ok((1..=maximum).collect())
    } else {
        parse_selection(input, maximum)
    }
}

fn manual_models() -> Result<Vec<String>> {
    println!("Enter model names one per line. Submit an empty line to finish.");
    let mut models = BTreeSet::new();
    loop {
        let model = prompt("Model", None)?;
        if model.is_empty() {
            break;
        }
        models.insert(model);
    }
    if models.is_empty() {
        bail!("at least one model must be entered");
    }
    Ok(models.into_iter().collect())
}

fn delete_provider(config: &Config) -> Result<Option<Config>> {
    let Some(name) = choose_provider(config, "Provider to delete")? else {
        return Ok(None);
    };
    if !prompt_yes_no(&format!("Delete provider `{name}`"), false)? {
        return Ok(None);
    }
    let mut updated = config.clone();
    updated.providers.remove(&name);
    if updated
        .routing
        .as_ref()
        .is_some_and(|routing| routing.unprefixed_model_provider == name)
    {
        updated.routing = None;
        println!("Unprefixed model provider was cleared.");
    }
    Ok(Some(updated))
}

fn toggle_provider(config: &Config) -> Result<Option<Config>> {
    let Some(name) = choose_provider(config, "Provider to toggle")? else {
        return Ok(None);
    };
    let mut updated = config.clone();
    let provider = updated.providers.get_mut(&name).expect("selected provider");
    provider.enabled = !provider.enabled;
    if !provider.enabled
        && updated
            .routing
            .as_ref()
            .is_some_and(|routing| routing.unprefixed_model_provider == name)
    {
        updated.routing = None;
        println!("Unprefixed model provider was cleared.");
    }
    println!(
        "Provider `{name}` will be {}.",
        if provider.enabled {
            "enabled"
        } else {
            "disabled"
        }
    );
    Ok(Some(updated))
}

fn choose_provider(config: &Config, label: &str) -> Result<Option<String>> {
    if config.providers.is_empty() {
        println!("No providers are configured.");
        return Ok(None);
    }
    println!("\n{label}:");
    let names = config.providers.keys().cloned().collect::<Vec<_>>();
    for (index, name) in names.iter().enumerate() {
        println!("{}) {}", index + 1, name);
    }
    let input = prompt("Number (Enter to cancel)", None)?;
    if input.is_empty() {
        return Ok(None);
    }
    let index = input
        .parse::<usize>()
        .context("provider choice must be a number")?;
    names
        .get(index.saturating_sub(1))
        .cloned()
        .map(Some)
        .context("provider choice is out of range")
}

#[derive(Default)]
struct ProfileModelSelections {
    base: Option<String>,
    openai_compact: Option<String>,
}

fn profile_models_if_replacement_needed(config: &Config) -> Result<ProfileModelSelections> {
    Ok(ProfileModelSelections {
        base: profile_model_if_replacement_needed(
            &sync::default_profile_path()?,
            &configured_models(config),
            sync::PROFILE_NAME,
        )?,
        openai_compact: profile_model_if_replacement_needed(
            &sync::default_openai_compact_profile_path()?,
            &configured_openai_compact_models(config),
            sync::OPENAI_COMPACT_PROFILE_NAME,
        )?,
    })
}

fn profile_model_if_replacement_needed(
    profile_path: &Path,
    models: &[crate::catalog::RoutedModel],
    profile_name: &str,
) -> Result<Option<String>> {
    let Some(current) = sync::current_profile_model(profile_path)? else {
        return Ok(None);
    };
    if models.is_empty() {
        return Ok(None);
    }
    if models.iter().any(|model| model.routed_id == current) {
        return Ok(None);
    }
    println!("\nThe `{profile_name}` profile model `{current}` is no longer available.");
    println!("Choose a replacement:\n");
    for (index, model) in models.iter().enumerate() {
        println!("{}) {}", index + 1, model.routed_id);
    }
    let input = prompt("Model number", Some("1"))?;
    let index = input
        .parse::<usize>()
        .context("model choice must be a number")?;
    models
        .get(index.saturating_sub(1))
        .map(|model| Some(model.routed_id.clone()))
        .context("model choice is out of range")
}

fn print_provider_summary(name: &str, provider: &ProviderConfig) {
    println!("\nProvider configuration:\n");
    println!("Name:     {name}");
    println!("Base URL: {}", provider.base_url);
    println!("API key:  {}", credential_summary(provider));
    if let Some(proxy_url) = &provider.proxy_url {
        println!("Proxy:    {proxy_url}");
    }
    println!("Models:   {} selected", provider.models.len());
    println!("Chat:     {} selected", provider.chat_models.len());
    println!(
        "Remote compaction: {} selected",
        provider.remote_compaction_models.len()
    );
    println!(
        "Enabled:  {}\n",
        if provider.enabled { "yes" } else { "no" }
    );
}

fn credential_summary(provider: &ProviderConfig) -> String {
    match (&provider.api_key, &provider.api_key_env) {
        (Some(_), _) => "stored API key".to_owned(),
        (_, Some(name)) => format!("environment variable {name}"),
        _ => "not configured".to_owned(),
    }
}

fn parse_selection(input: &str, maximum: usize) -> Result<Vec<usize>> {
    let mut selected = BTreeSet::new();
    for part in input
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        if let Some((start, end)) = part.split_once('-') {
            let start = parse_index(start, maximum)?;
            let end = parse_index(end, maximum)?;
            if start > end {
                bail!("invalid descending range `{part}`");
            }
            selected.extend(start..=end);
        } else {
            selected.insert(parse_index(part, maximum)?);
        }
    }
    if selected.is_empty() {
        bail!("no models selected");
    }
    Ok(selected.into_iter().collect())
}

fn parse_index(value: &str, maximum: usize) -> Result<usize> {
    let index = value
        .parse::<usize>()
        .with_context(|| format!("invalid model number `{value}`"))?;
    if index == 0 || index > maximum {
        bail!("model number `{index}` is out of range 1-{maximum}");
    }
    Ok(index)
}

fn prompt_yes_no(label: &str, default: bool) -> Result<bool> {
    let suffix = if default { "Y/n" } else { "y/N" };
    loop {
        let answer = prompt(&format!("{label} [{suffix}]"), None)?;
        match answer.to_ascii_lowercase().as_str() {
            "" => return Ok(default),
            "y" | "yes" => return Ok(true),
            "n" | "no" => return Ok(false),
            _ => println!("Enter y or n."),
        }
    }
}

fn prompt_required(label: &str) -> Result<String> {
    loop {
        let value = prompt(label, None)?;
        if !value.is_empty() {
            return Ok(value);
        }
        println!("A value is required.");
    }
}

fn prompt_api_key() -> Result<String> {
    println!("Enter API key (input will be visible).");
    prompt_required("API key")
}

fn prompt(label: &str, default: Option<&str>) -> Result<String> {
    match default {
        Some(default) => print!("{label} [{default}]: "),
        None => print!("{label}: "),
    }
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let input = input.trim().to_owned();
    if input.is_empty() {
        Ok(default.unwrap_or_default().to_owned())
    } else {
        Ok(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_model_numbers_and_ranges() {
        assert_eq!(parse_selection("1,3-5,3", 5).unwrap(), vec![1, 3, 4, 5]);
        assert!(parse_selection("0", 5).is_err());
        assert!(parse_selection("5-3", 5).is_err());
    }

    #[test]
    fn resolves_displayed_model_selection_defaults() {
        assert_eq!(
            resolve_model_selection("keep marked", vec![1, 3], 4).unwrap(),
            vec![1, 3]
        );
        assert_eq!(
            resolve_model_selection("all", vec![1, 3], 4).unwrap(),
            vec![1, 2, 3, 4]
        );
    }

    #[test]
    fn removes_all_whitespace_from_search_keywords() {
        assert_eq!(remove_whitespace(" hello you "), "helloyou");
        assert_eq!(remove_whitespace(" \t\n "), "");
    }

    #[test]
    fn matches_search_keywords_case_insensitively() {
        let keyword = remove_whitespace(" HEL lo ").to_ascii_lowercase();
        let models = ["hello3", "33HELLO44", "88hello", "other"];
        let matches = models
            .iter()
            .filter(|model| model.to_ascii_lowercase().contains(&keyword))
            .collect::<Vec<_>>();
        assert_eq!(matches, vec![&"hello3", &"33HELLO44", &"88hello"]);
    }

    #[test]
    fn recommends_chat_except_for_known_native_model_families() {
        for model in [
            "gpt-5",
            "o3",
            "o4-mini",
            "codex-mini",
            "claude-sonnet-4",
            "gemini-2.5-pro",
            "grok-4",
            "openai/gpt-5",
            "xai/grok-4",
        ] {
            assert!(!recommend_chat_protocol(model), "{model}");
        }
        for model in ["glm-4.5", "deepseek-v3", "qwen3-coder", "vendor/glm-4.5"] {
            assert!(recommend_chat_protocol(model), "{model}");
        }
    }

    fn model_editor_provider() -> ProviderConfig {
        ProviderConfig {
            base_url: Url::parse("https://example.com/v1").unwrap(),
            proxy_url: None,
            api_key: Some("test-key".to_owned()),
            api_key_env: None,
            enabled: true,
            models: vec!["gpt-test".to_owned()],
            chat_models: Vec::new(),
            messages_models: Vec::new(),
            remote_compaction_models: vec!["gpt-test".to_owned()],
        }
    }

    #[test]
    fn adds_models_with_the_selected_api() {
        let mut provider = model_editor_provider();
        add_model_to_provider(
            &mut provider,
            "z-ai/glm-5.3-flash:free",
            ModelApi::ChatCompletions,
        )
        .unwrap();
        add_model_to_provider(
            &mut provider,
            "anthropic/model",
            ModelApi::AnthropicMessages,
        )
        .unwrap();

        assert_eq!(
            provider.models,
            vec![
                "anthropic/model".to_owned(),
                "gpt-test".to_owned(),
                "z-ai/glm-5.3-flash:free".to_owned()
            ]
        );
        assert_eq!(
            provider.chat_models,
            vec!["z-ai/glm-5.3-flash:free".to_owned()]
        );
        assert_eq!(provider.messages_models, vec!["anthropic/model".to_owned()]);
        assert!(add_model_to_provider(
            &mut provider,
            "z-ai/glm-5.3-flash:free",
            ModelApi::Responses
        )
        .is_err());
    }

    #[test]
    fn renames_models_across_related_model_lists() {
        let mut provider = model_editor_provider();
        provider.models.push("glm-test".to_owned());
        provider.chat_models.push("glm-test".to_owned());

        rename_model_in_provider(&mut provider, "glm-test", "z-ai/glm-5.3-flash:free").unwrap();

        assert_eq!(
            provider.models,
            vec!["gpt-test".to_owned(), "z-ai/glm-5.3-flash:free".to_owned()]
        );
        assert_eq!(
            provider.chat_models,
            vec!["z-ai/glm-5.3-flash:free".to_owned()]
        );
        assert_eq!(provider.messages_models, Vec::<String>::new());
    }

    #[test]
    fn switching_api_removes_the_previous_compatibility_choice() {
        let mut provider = model_editor_provider();
        provider.models.push("glm-test".to_owned());
        provider.chat_models.push("glm-test".to_owned());

        set_model_api(&mut provider, "glm-test", ModelApi::AnthropicMessages).unwrap();
        assert!(!provider.chat_models.contains(&"glm-test".to_owned()));
        assert_eq!(provider.messages_models, vec!["glm-test".to_owned()]);

        set_model_api(&mut provider, "glm-test", ModelApi::Responses).unwrap();
        assert!(!provider.chat_models.contains(&"glm-test".to_owned()));
        assert!(!provider.messages_models.contains(&"glm-test".to_owned()));
    }

    #[test]
    fn deletes_models_from_every_related_list() {
        let mut provider = model_editor_provider();
        provider.models.push("glm-test".to_owned());
        provider.chat_models.push("glm-test".to_owned());
        provider
            .remote_compaction_models
            .push("glm-test".to_owned());

        remove_model_from_provider(&mut provider, "glm-test").unwrap();

        assert_eq!(provider.models, vec!["gpt-test".to_owned()]);
        assert_eq!(provider.chat_models, Vec::<String>::new());
        assert_eq!(provider.messages_models, Vec::<String>::new());
        assert_eq!(
            provider.remote_compaction_models,
            vec!["gpt-test".to_owned()]
        );
    }

    #[test]
    fn detects_actual_model_selection_changes_only() {
        let original = model_editor_provider();

        assert!(!provider_model_selections_changed(&original, &original));

        let reordered = ProviderConfig {
            models: original.models.iter().rev().cloned().collect(),
            ..original.clone()
        };
        assert!(!provider_model_selections_changed(&reordered, &original));

        let changed = ProviderConfig {
            chat_models: vec!["gpt-test".to_owned()],
            ..original.clone()
        };
        assert!(provider_model_selections_changed(&changed, &original));

        let compact_changed = ProviderConfig {
            remote_compaction_models: Vec::new(),
            ..original.clone()
        };
        assert!(provider_model_selections_changed(
            &compact_changed,
            &original
        ));
    }

    #[test]
    fn rejects_invalid_quick_model_edits() {
        let mut provider = model_editor_provider();

        assert!(
            add_model_to_provider(&mut provider, "model-openai-compact", ModelApi::Responses)
                .is_err()
        );
        assert!(rename_model_in_provider(&mut provider, "gpt-test", "").is_err());
        assert!(remove_model_from_provider(&mut provider, "gpt-test").is_err());
    }

    #[test]
    fn preserves_existing_chat_choices_and_classifies_only_new_models() {
        let selected = vec![
            "glm-old".to_owned(),
            "deepseek-old".to_owned(),
            "glm-new".to_owned(),
            "gpt-new".to_owned(),
        ];
        let previous = vec!["glm-old".to_owned(), "deepseek-old".to_owned()];
        let previous_chat = vec!["deepseek-old".to_owned()];

        assert_eq!(
            default_chat_model_indices(&selected, &previous, &previous_chat),
            vec![2, 3]
        );
    }
}
