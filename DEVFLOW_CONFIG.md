# Devflow configuration

Devflow merges configuration in this order, with later values taking precedence:

1. Built-in defaults
2. `~/.config/devflow/config.toml`
3. `<repo>/.devflow.toml`
4. CLI `--provider` and `--model` overrides

## Global configuration

Create the shared configuration once:

```bash
mkdir -p ~/.config/devflow
cp /path/to/devflow/.devflow.global.example.toml \
  ~/.config/devflow/config.toml
```

Use the global file for provider URLs and model defaults:

```toml
[model]
provider = "ollama"
temperature = 0

[providers.ollama]
model = "your-local-model"
base_url = "http://192.168.1.50:11434"

[providers.openrouter]
model = "your-openrouter-model"
base_url = "https://openrouter.ai/api/v1"
```

Keep `OPENROUTER_API_KEY` in the environment rather than the TOML file.

## Repository configuration

Keep project-specific commands and Git settings in `<repo>/.devflow.toml`:

```toml
[review]
base_ref = "upstream/beta"
output_dir = ".devflow/reviews"
max_diff_chars = 40000
max_command_output_chars = 12000

[commands]
check = ["npm run check"]
test = ["npm test --workspace @solaris/common"]
```

Development planning uses a separate section:

```toml
[plan]
output_dir = ".devflow/plans"
max_context_chars = 30000
max_requested_files = 8
max_searches = 6
max_search_results_chars = 12000
save_model_exchange = false
```

Run a read-only plan with:

```bash
devflow plan "Describe the development outcome"
```

A repository may override `[model]` or `[providers.*]`, but normally it does not need to.

## One-run overrides

```bash
devflow review --provider openrouter --model provider/model-name
```

At startup and in the Prefect task log, Devflow prints the resolved provider, model, and endpoint.
