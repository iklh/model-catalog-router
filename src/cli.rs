use clap::{Parser, Subcommand};
use std::net::SocketAddr;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(version, about)]
pub struct Cli {
    /// Path to config.toml. Defaults to the XDG configuration directory.
    #[arg(long, global = true)]
    pub config: Option<PathBuf>,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Open the interactive provider configuration menu.
    Config,
    /// Create an example configuration file.
    Init {
        /// Replace an existing configuration file.
        #[arg(long)]
        force: bool,
    },
    /// Check all enabled providers, one provider, or one routed model.
    Check {
        /// Optional provider name or routed model name.
        target: Option<String>,
    },
    /// List the routed model names exposed to Codex.
    #[command(name = "listmodels")]
    ListModels {
        /// Print structured JSON.
        #[arg(long)]
        json: bool,
    },
    /// List all configured provider names.
    #[command(name = "listproviders")]
    ListProviders,
    /// Regenerate the Codex catalog and Router profile.
    Regenerate {
        /// Output path. Defaults next to config.toml.
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Start the local OpenAI-compatible router.
    Serve {
        /// Override server.listen from config.toml.
        #[arg(long)]
        listen: Option<SocketAddr>,
    },
    /// Start only the OpenAI remote-compaction router.
    ServeOpenaiCompact {
        /// Override server.openai_compact_listen from config.toml.
        #[arg(long)]
        listen: Option<SocketAddr>,
    },
    /// Start both the base and OpenAI remote-compaction routers.
    ServeAll {
        /// Override server.listen from config.toml.
        #[arg(long)]
        listen: Option<SocketAddr>,
        /// Override server.openai_compact_listen from config.toml.
        #[arg(long)]
        openai_compact_listen: Option<SocketAddr>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_list_commands_without_hyphens() {
        let models = Cli::try_parse_from(["model-catalog-router", "listmodels"]).unwrap();
        assert!(matches!(
            models.command,
            Command::ListModels { json: false }
        ));

        let providers = Cli::try_parse_from(["model-catalog-router", "listproviders"]).unwrap();
        assert!(matches!(providers.command, Command::ListProviders));
    }

    #[test]
    fn old_command_names_are_not_accepted() {
        assert!(Cli::try_parse_from(["model-catalog-router", "models"]).is_err());
        assert!(Cli::try_parse_from(["model-catalog-router", "catalog"]).is_err());
    }

    #[test]
    fn parses_optional_check_target() {
        let all = Cli::try_parse_from(["model-catalog-router", "check"]).unwrap();
        assert!(matches!(all.command, Command::Check { target: None }));

        let model =
            Cli::try_parse_from(["model-catalog-router", "check", "alpha/gpt-test"]).unwrap();
        assert!(matches!(
            model.command,
            Command::Check { target: Some(target) } if target == "alpha/gpt-test"
        ));
    }

    #[test]
    fn parses_openai_compact_service_commands() {
        let compact = Cli::try_parse_from([
            "model-catalog-router",
            "serve-openai-compact",
            "--listen",
            "127.0.0.1:9001",
        ])
        .unwrap();
        assert!(matches!(
            compact.command,
            Command::ServeOpenaiCompact { listen: Some(_) }
        ));

        let all = Cli::try_parse_from([
            "model-catalog-router",
            "serve-all",
            "--listen",
            "127.0.0.1:9000",
            "--openai-compact-listen",
            "127.0.0.1:9001",
        ])
        .unwrap();
        assert!(matches!(
            all.command,
            Command::ServeAll {
                listen: Some(_),
                openai_compact_listen: Some(_)
            }
        ));
    }
}
