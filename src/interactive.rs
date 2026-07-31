use crate::catalog::{
    configured_models, configured_openai_compact_models, discover_provider_models,
    DiscoveredProviderModels,
};
use crate::config::{validate_provider_name, Config, ProviderConfig};
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
        let choice = prompt("e/n/d/r/t/l/q> ", None)?.to_ascii_lowercase();
        if choice == "q" {
            return Ok(());
        }
        let result = match choice.as_str() {
            "e" => edit_provider(&config).await,
            "n" => new_provider(&config).await,
            "d" => delete_provider(&config),
            "r" => refresh_provider(&config).await,
            "t" => toggle_provider(&config),
            "l" => {
                show_provider_models(&config)?;
                continue;
            }
            "" => continue,
            _ => {
                println!("Unknown choice. Enter e, n, d, r, t, l, or q.");
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
    println!("\ne) Edit existing provider");
    println!("n) New provider");
    println!("d) Delete provider");
    println!("r) Refresh provider models");
    println!("t) Toggle provider enabled");
    println!("l) List provider models");
    println!("q) Quit configuration\n");
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
    let mut provider = ProviderConfig {
        base_url,
        api_key,
        api_key_env,
        enabled,
        models: current
            .map(|provider| provider.models.clone())
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
                        provider.models =
                            select_models(&discovered.models, current_models, current.is_none())?;
                        provider.remote_compaction_models = configure_remote_compaction(
                            &discovered,
                            &provider.models,
                            current.map(|provider| provider.remote_compaction_models.as_slice()),
                        )?;
                        break;
                    }
                    DiscoveryOutcome::EditConnection => {
                        edit_connection_details(&mut provider)?;
                    }
                    DiscoveryOutcome::Manual => {
                        provider.models = manual_models()?;
                        provider.remote_compaction_models.clear();
                        break;
                    }
                    DiscoveryOutcome::Cancel => return Ok(None),
                }
            }
        } else {
            provider.models = manual_models()?;
            provider.remote_compaction_models.clear();
        }
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
    Ok(())
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

async fn refresh_provider(config: &Config) -> Result<Option<Config>> {
    let Some(name) = choose_provider(config, "Provider to refresh")? else {
        return Ok(None);
    };
    let mut provider = config
        .providers
        .get(&name)
        .expect("selected provider")
        .clone();
    let discovered = loop {
        match discover_models_with_recovery(&name, &provider, false).await? {
            DiscoveryOutcome::Discovered(models) => break models,
            DiscoveryOutcome::EditConnection => edit_connection_details(&mut provider)?,
            DiscoveryOutcome::Cancel => return Ok(None),
            DiscoveryOutcome::Manual => unreachable!("manual entry is disabled during refresh"),
        }
    };
    let selected = select_models(&discovered.models, Some(&provider.models), false)?;
    if selected.is_empty() {
        bail!("at least one model must be selected");
    }
    let remote_compaction_models = configure_remote_compaction(
        &discovered,
        &selected,
        Some(&provider.remote_compaction_models),
    )?;
    let mut updated = config.clone();
    provider.models = selected;
    provider.remote_compaction_models = remote_compaction_models;
    updated.providers.insert(name, provider);
    Ok(Some(updated))
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
    Ok(Some(updated))
}

fn toggle_provider(config: &Config) -> Result<Option<Config>> {
    let Some(name) = choose_provider(config, "Provider to toggle")? else {
        return Ok(None);
    };
    let mut updated = config.clone();
    let provider = updated.providers.get_mut(&name).expect("selected provider");
    provider.enabled = !provider.enabled;
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
    let Some(current) = sync::current_profile_model(&profile_path)? else {
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
    println!("Models:   {} selected", provider.models.len());
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
}
