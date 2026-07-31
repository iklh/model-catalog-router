mod catalog;
pub mod chat;
mod cli;
mod config;
mod interactive;
mod proxy;
pub mod responses;
mod sync;

use anyhow::{bail, Result};
use clap::Parser;
use cli::{Cli, Command};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();
    let config_path = cli.config.unwrap_or_else(config::default_config_path);

    match cli.command {
        Command::Config => interactive::run(&config_path).await,
        Command::Init { force } => config::init(&config_path, force),
        Command::Check { target } => {
            let loaded = config::Config::load(&config_path)?;
            if let Some(target) = target {
                check_target(&loaded, &target).await
            } else {
                check_all(&loaded).await
            }
        }
        Command::ListModels { json } => {
            let loaded = config::Config::load(&config_path)?;
            loaded.validate()?;
            let models = catalog::configured_models(&loaded);
            if json {
                println!("{}", serde_json::to_string_pretty(&models)?);
            } else {
                for model in models {
                    println!("{}", model.routed_id);
                }
            }
            Ok(())
        }
        Command::ListProviders => {
            let loaded = config::Config::load(&config_path)?;
            let width = loaded.providers.keys().map(String::len).max().unwrap_or(0);
            for (name, provider) in &loaded.providers {
                let status = if provider.enabled {
                    "enabled"
                } else {
                    "disabled"
                };
                println!("{name:<width$}  {status}");
            }
            Ok(())
        }
        Command::Regenerate { output } => {
            let loaded = config::Config::load(&config_path)?;
            loaded.validate()?;
            let models = catalog::configured_models(&loaded);
            if let Some(output) = output {
                catalog::write_catalog(&output, &models, loaded.catalog.context_window)?;
                println!("wrote {} models to {}", models.len(), output.display());
            } else {
                let result = sync::sync_files(&loaded, &config_path, None, None)?;
                println!(
                    "wrote {} models to {}",
                    models.len(),
                    result.base.catalog_path.display()
                );
                println!("updated {}", result.base.profile_path.display());
                if let Some(compact) = result.openai_compact {
                    println!(
                        "wrote {} remote-compaction models to {}",
                        catalog::configured_openai_compact_models(&loaded).len(),
                        compact.catalog_path.display()
                    );
                    println!("updated {}", compact.profile_path.display());
                }
            }
            Ok(())
        }
        Command::Serve { listen } => {
            let mut loaded = config::Config::load(&config_path)?;
            if let Some(listen) = listen {
                loaded.server.listen = listen;
            }
            loaded.validate()?;
            proxy::serve(loaded).await
        }
        Command::ServeOpenaiCompact { listen } => {
            let mut loaded = config::Config::load(&config_path)?;
            if let Some(listen) = listen {
                loaded.server.openai_compact_listen = listen;
            }
            loaded.validate()?;
            proxy::serve_openai_compact(loaded).await
        }
        Command::ServeAll {
            listen,
            openai_compact_listen,
        } => {
            let mut loaded = config::Config::load(&config_path)?;
            if let Some(listen) = listen {
                loaded.server.listen = listen;
            }
            if let Some(listen) = openai_compact_listen {
                loaded.server.openai_compact_listen = listen;
            }
            loaded.validate()?;
            proxy::serve_all(loaded).await
        }
    }
}

async fn check_all(config: &config::Config) -> Result<()> {
    config.validate()?;
    let discovered = catalog::fetch_models(config).await?;
    let configured = catalog::configured_models(config);
    let discovered_ids = discovered
        .iter()
        .map(|model| model.routed_id.as_str())
        .collect::<std::collections::HashSet<_>>();
    let missing = configured
        .iter()
        .filter(|model| !discovered_ids.contains(model.routed_id.as_str()))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        bail!(
            "{} configured models are missing upstream: {}",
            missing.len(),
            missing
                .iter()
                .map(|model| model.routed_id.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    let mut missing_compact = Vec::new();
    for (name, provider) in config
        .providers
        .iter()
        .filter(|(_, provider)| provider.enabled)
    {
        for model in &provider.remote_compaction_models {
            let routed = format!(
                "{name}{}{}{}",
                config.catalog.separator,
                model,
                catalog::OPENAI_COMPACT_SUFFIX
            );
            if !discovered_ids.contains(routed.as_str()) {
                missing_compact.push(routed);
            }
        }
    }
    if !missing_compact.is_empty() {
        bail!(
            "{} configured remote compaction aliases are missing upstream: {}",
            missing_compact.len(),
            missing_compact.join(", ")
        );
    }
    println!(
        "configuration is valid; checked {} configured models",
        configured.len()
    );
    Ok(())
}

async fn check_target(config: &config::Config, target: &str) -> Result<()> {
    match parse_check_target(target, &config.catalog.separator)? {
        CheckTarget::Provider(name) => {
            let provider = config.provider_for_check(name)?;
            let discovered = catalog::discover_provider_models(name, provider).await?;
            let missing = provider
                .models
                .iter()
                .filter(|model| !discovered.models.contains(model))
                .collect::<Vec<_>>();
            if !missing.is_empty() {
                bail!(
                    "provider `{name}` is missing {} configured models upstream: {}",
                    missing.len(),
                    missing
                        .iter()
                        .map(|model| format!("{name}{}{model}", config.catalog.separator))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            let missing_compact = provider
                .remote_compaction_models
                .iter()
                .filter(|model| !discovered.remote_compaction_models.contains(model))
                .collect::<Vec<_>>();
            if !missing_compact.is_empty() {
                bail!(
                    "provider `{name}` is missing {} configured remote compaction aliases upstream: {}",
                    missing_compact.len(),
                    missing_compact
                        .iter()
                        .map(|model| format!(
                            "{name}{}{model}{}",
                            config.catalog.separator,
                            catalog::OPENAI_COMPACT_SUFFIX
                        ))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            let noun = if provider.models.len() == 1 {
                "model"
            } else {
                "models"
            };
            println!(
                "checked provider `{name}`: all {} configured {noun} available",
                provider.models.len()
            );
            Ok(())
        }
        CheckTarget::Model { provider, model } => {
            let provider_config = config.provider_for_check(provider)?;
            if !provider_config
                .models
                .iter()
                .any(|configured| configured == model)
            {
                bail!("model `{model}` is not configured for provider `{provider}`");
            }
            let discovered = catalog::discover_provider_models(provider, provider_config).await?;
            if !discovered
                .models
                .iter()
                .any(|advertised| advertised == model)
            {
                bail!("provider `{provider}` does not advertise model `{model}`");
            }
            if provider_config
                .remote_compaction_models
                .iter()
                .any(|configured| configured == model)
                && !discovered
                    .remote_compaction_models
                    .iter()
                    .any(|advertised| advertised == model)
            {
                bail!(
                    "provider `{provider}` does not advertise remote compaction alias `{model}{}`",
                    catalog::OPENAI_COMPACT_SUFFIX
                );
            }
            println!("checked model `{target}`: available upstream");
            Ok(())
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum CheckTarget<'a> {
    Provider(&'a str),
    Model { provider: &'a str, model: &'a str },
}

fn parse_check_target<'a>(target: &'a str, separator: &str) -> Result<CheckTarget<'a>> {
    if target.is_empty() {
        bail!("check target must not be empty");
    }
    if separator.is_empty() {
        bail!("catalog.separator must not be empty");
    }
    if let Some((provider, model)) = target.split_once(separator) {
        if provider.is_empty() || model.is_empty() {
            bail!("model check target must use `<provider>{separator}<model>`");
        }
        Ok(CheckTarget::Model { provider, model })
    } else {
        Ok(CheckTarget::Provider(target))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_provider_and_model_check_targets() {
        assert_eq!(
            parse_check_target("alpha", "/").unwrap(),
            CheckTarget::Provider("alpha")
        );
        assert_eq!(
            parse_check_target("alpha/org/model", "/").unwrap(),
            CheckTarget::Model {
                provider: "alpha",
                model: "org/model"
            }
        );
        assert_eq!(
            parse_check_target("alpha::model", "::").unwrap(),
            CheckTarget::Model {
                provider: "alpha",
                model: "model"
            }
        );
    }

    #[test]
    fn rejects_incomplete_model_check_target() {
        assert!(parse_check_target("alpha/", "/").is_err());
        assert!(parse_check_target("/model", "/").is_err());
    }
}
