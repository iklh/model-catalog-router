# Model Catalog Router

Model Catalog Router gives Codex one local OpenAI-compatible endpoint and routes each request to a different new-api provider according to its namespaced model name.

```text
newapi-a/gpt-5 -> newapi-a base URL + API key -> gpt-5
newapi-b/gpt-5 -> newapi-b base URL + API key -> gpt-5
```

Router models always include a Provider prefix. The dedicated Codex profile keeps them separate from the normal Codex model catalog and its saved model/reasoning selection.

## Requirements

Rust 1.88 or newer is required. Debian 13's base Rust 1.85 packages are too old;
after enabling `trixie-backports`, install the backported toolchain:

```bash
sudo apt install -t trixie-backports cargo rustc
cargo install --locked --path . --root ~/.local --force
```

This builds an optimized release and installs `model-catalog-router` at `~/.local/bin/model-catalog-router`. Ensure `~/.local/bin` is in `PATH` before using the commands below. Cargo downloads the Rust dependencies recorded in `Cargo.lock`; rustup is not required. To build without installing, use `cargo build --release`.

## Interactive Configuration

Open the rclone-style configuration menu:

```bash
model-catalog-router config
```

The menu supports:

- Adding and editing Providers.
- Deleting Providers after confirmation.
- Enabling or disabling Providers without deleting their settings.
- Refreshing a Provider's model list.
- Storing an API key directly or naming an environment variable.

When adding a Provider, automatic discovery calls its OpenAI-compatible `GET /v1/models` endpoint. A failed request can be retried, can return to the Base URL and API key prompts, or can fall back to manual entry. Press Enter to select all discovered models, or enter a complete selection such as `1,3-5`. Manual mode accepts model names without contacting the Provider or validating them.

Discovery treats model IDs ending in `-openai-compact` as remote-compaction aliases rather than normal selectable models. An alias is recognized only when the same response also contains its exact base model:

```text
sol
sol-openai-compact
```

The configuration menu displays only `sol`. Orphan aliases such as `luna-openai-compact` without `luna` are silently ignored. After the normal model selection, the menu offers remote compaction only when at least one selected base model has a matching alias. If enabled, all supported selected base models are initially marked and can be reduced to a partial selection. Manual model entry clears remote-compaction selection because it provides no discovery evidence.

Every saved change automatically updates:

```text
~/.config/model-catalog-router/config.toml
~/.config/model-catalog-router/model-catalog.json
~/.codex/mc-router.config.toml
```

When at least one enabled Provider has a selected remote-compaction model, synchronization also updates:

```text
~/.config/model-catalog-router/model-catalog-openai-compact.json
~/.codex/mc-router-openai-compact.config.toml
```

If no enabled remote-compaction models remain, these two generated compact files are removed.

`$XDG_CONFIG_HOME` and `$CODEX_HOME` are respected when set. Configuration and profile files are written with mode `0600` on Unix.

The resulting Router configuration resembles:

```toml
[server]
listen = "127.0.0.1:8787"
openai_compact_listen = "127.0.0.1:8788"

[catalog]
separator = "/"
context_window = 128000

[providers.newapi-a]
base_url = "https://newapi-a.example.com/v1"
api_key_env = "NEWAPI_A_API_KEY"
enabled = true
models = ["gpt-5", "gpt-5-mini"]
remote_compaction_models = ["gpt-5"]

[providers.newapi-b]
base_url = "https://newapi-b.example.com/v1"
api_key = "sk-replace-me"
enabled = false
models = ["claude-sonnet"]
remote_compaction_models = []
```

`remote_compaction_models` always contains base model IDs, never suffixed aliases. It must be a subset of `models`. Both listen addresses must be distinct loopback addresses.

For environment-backed keys, create `~/.config/model-catalog-router/.env` or export the variable before starting the Router:

```dotenv
NEWAPI_A_API_KEY=sk-replace-me
```

The process environment takes precedence over the `.env` file. Logs never include API keys or Authorization headers.

## Run With Codex

Start the Router:

```bash
model-catalog-router serve
```

Start Codex with its isolated Router profile:

```bash
codex -p mc-router
```

This base service listens on `127.0.0.1:8787` by default.

To run only the remote-compaction service:

```bash
model-catalog-router serve-openai-compact
codex -p mc-router-openai-compact
```

It listens on `127.0.0.1:8788` by default. The compact profile intentionally generates a Provider whose displayed name is `OpenAI`, which enables Codex's remote-compaction behavior. The profile and its catalog still show base routed IDs such as `newapi-a/gpt-5`; `-openai-compact` aliases are never exposed to the user.

The compact service handles routing transparently:

```text
POST /v1/responses
newapi-a/gpt-5 -> upstream gpt-5

POST /v1/responses/compact
newapi-a/gpt-5 -> upstream gpt-5-openai-compact
```

Only models selected in `remote_compaction_models` are accepted by this service. A Provider merely declaring the alias is trusted during configuration; the Router does not send a test compaction request. If the upstream compact request later fails, Codex reports the remote-compaction failure rather than falling back locally to ordinary `/responses`.

Start both listeners atomically in one foreground process with:

```bash
model-catalog-router serve-all
```

`serve-all` binds both addresses before serving either one, and one Ctrl+C stops both. It fails clearly when no remote-compaction models are configured. The separate `serve` and `serve-openai-compact` commands can instead be run as independent processes so that either service can be started or stopped without affecting the other. Switching profiles requires exiting Codex and launching it again with the desired `-p` option.

Normal `codex` continues to use `~/.codex/config.toml`, its normal model catalog, and its previous model/reasoning selection. Router sessions use `~/.codex/mc-router.config.toml`. Changes made with `/model` in a Router session are persisted to that profile by Codex.

Each generated profile contains its own local Provider settings and catalog path. On first creation, its initial model is the first enabled model in that catalog. Later synchronization preserves each profile's current model while it remains available and prompts separately for a replacement during interactive configuration if it was removed or disabled.

Generated Router entries match each upstream model ID against `$CODEX_HOME/models_cache.json` (or `~/.codex/models_cache.json`) and inherit that Codex model's supported reasoning levels. The Router uses `medium` as the catalog default when the model supports it; otherwise it keeps a valid default from the matching Codex model. Models that Codex does not recognize remain available without invented reasoning options.

Unprefixed model requests are rejected. There is no default Router Provider; `newapi-a/gpt-5` and `newapi-b/gpt-5` always have unambiguous destinations.

## Run as a User systemd Service on Debian 13

For a long-running Router on Debian 13, use a user-level systemd service. The Router remains a normal foreground process; systemd starts it in the background, restarts it after failures, and collects its logs. A user service also keeps the Router configuration and API keys under the same Unix account that runs Codex.

Install or update the release binary in the user's local binary directory:

```bash
cargo install --locked --path . --root ~/.local --force
```

Create `~/.config/systemd/user/model-catalog-router.service` with the following contents:

```ini
[Unit]
Description=Model Catalog Router

[Service]
Type=simple
ExecStart=%h/.local/bin/model-catalog-router --config %h/.config/model-catalog-router/config.toml serve
Restart=on-failure
RestartSec=3s

[Install]
WantedBy=default.target
```

If `$XDG_CONFIG_HOME` is set to a non-default location, adjust the `--config` path accordingly. The Router automatically loads `.env` from the directory containing `config.toml`, so the service does not need a separate `EnvironmentFile` directive.

Load the unit, start it immediately, and enable it for future user sessions:

```bash
systemctl --user daemon-reload
systemctl --user enable --now model-catalog-router.service
```

Inspect its status, follow its logs, or verify its health endpoint with:

```bash
systemctl --user status model-catalog-router.service
journalctl --user -u model-catalog-router.service -f
curl --fail http://127.0.0.1:8787/health
```

The health response reports only that the selected local listener is alive:

```json
{"status":"ok","mode":"base"}
```

The compact listener reports `"mode":"openai-compact"`. Health checks do not contact any upstream Provider or verify model availability.

The service reads its configuration when it starts. Restart it after changing Providers, keys, models, or the listen address:

```bash
systemctl --user restart model-catalog-router.service
```

After updating the source, reinstall the binary and restart the service:

```bash
cargo install --locked --path . --root ~/.local --force
systemctl --user restart model-catalog-router.service
```

By default, an enabled user service starts when that user logs in. On a server where the Router must start during boot and continue without an interactive login, enable lingering once for the account:

```bash
sudo loginctl enable-linger "$USER"
```

Disable and stop the service with:

```bash
systemctl --user disable --now model-catalog-router.service
```

After disabling the service, uninstall the binary with:

```bash
cargo uninstall --root ~/.local model-catalog-router
```

## Commands

```text
model-catalog-router config
model-catalog-router check [TARGET]
model-catalog-router listmodels [--json]
model-catalog-router listproviders
model-catalog-router regenerate [--output PATH]
model-catalog-router serve [--listen ADDRESS]
model-catalog-router serve-openai-compact [--listen ADDRESS]
model-catalog-router serve-all [--listen ADDRESS] [--openai-compact-listen ADDRESS]
model-catalog-router init [--force]
```

All commands accept `--config PATH` for an alternate Router configuration. `regenerate` rebuilds the base catalog/profile and, when configured, the compact catalog/profile; with `--output`, it writes only the requested base catalog file. `listproviders` prints every configured Provider with its `enabled` or `disabled` status.

```text
newapi-a  enabled
newapi-b  disabled
```

Without a target, `check` contacts every enabled Provider and verifies that all configured base models and selected remote-compaction aliases are still advertised upstream. A Provider target checks only that Provider, including when it is disabled. A routed model target checks that configured base model and, if selected for remote compaction, its matching alias:

```bash
model-catalog-router check
model-catalog-router check newapi-a
model-catalog-router check newapi-a/gpt-5
```

Model targets use `catalog.separator` (the default is `/`) and split only at its first occurrence, so upstream model IDs may contain the separator. Checks use each Provider's `GET /v1/models` endpoint; they do not send an inference request. `init` remains available for non-interactive example configuration creation.

## Current Scope

- Routes selected models through OpenAI-compatible JSON endpoints under `/v1/*`.
- Replaces namespaced model names and upstream bearer tokens per request.
- Uses a separate compact listener/profile/catalog and rewrites only `/responses/compact` requests to matching `-openai-compact` aliases.
- Streams upstream response bodies without buffering, including Responses API SSE.
- Allows only loopback listeners and requires no local API key.
- Inherits supported reasoning levels from Codex's model cache when an upstream model ID matches a Codex model slug, preferring the Router's `medium` default when supported; unmatched models keep conservative capability metadata because standard `/v1/models` responses do not describe reasoning features.

WebSocket transport, per-model capability overrides, automatic service installation, and automatic Router process startup are outside the first version. The user-level systemd service above can be installed manually.
