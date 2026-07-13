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
base_url = "http://127.0.0.1:11434"

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
save_model_exchange = false
max_output_tokens = 8000
compact_retry_output_tokens = 4000
```

Planning currently uses LangChain function calling with the `portable-v1` plan schema.
The schema requires every field and rejects unknown fields for broad provider compatibility.
Text-length and list-size limits are model guidance rather than provider-enforced JSON Schema
constraints; the output token limits above remain enforced by the model client.

Run a read-only plan with:

```bash
devflow plan "Describe the development outcome"
```

By default, planning runs Serena context discovery first. Reuse an existing context
or revise a prior structured plan with:

```bash
devflow plan --context /path/to/context.json "Describe the development outcome"
devflow plan --from-plan /path/to/plan.json "Describe the development outcome"
devflow plan --context /path/to/context.json --from-plan /path/to/plan.json \
  "Describe the development outcome"
```

Reused context must belong to the same repository and current commit. Its sibling
`evidence.json` supplies the repository identity.

## Serena repository context

Install and initialize Serena before using repository-context discovery:

```bash
uv tool install -p 3.13 serena-agent
serena init
```

Configure the read-only context-discovery process:

```toml
[serena]
output_dir = ".devflow/serena-context"
command = "serena"
args = ["start-mcp-server", "--context", "ide", "--project", "{repo}"]
max_rounds = 3
max_tool_calls_per_round = 12
max_total_tool_calls = 36
max_tool_result_chars = 8000
max_transcript_chars = 60000
max_report_output_tokens = 5000
model_request_min_interval_seconds = 2.0
```

Run Serena context discovery independently with:

```bash
devflow serena-context "Describe the development outcome"
```

Devflow exposes only Serena retrieval tools during this workflow. Editing, shell,
memory-writing, and project-mutation tools are not available to the model.
Model requests are started no more frequently than the configured minimum interval.
Serena server diagnostics are saved to `serena.log` in the run directory instead of
being printed live; tool events and errors remain available in the transcript.

A repository may override `[model]` or `[providers.*]`, but normally it does not need to.

## One-run overrides

```bash
devflow review --provider openrouter --model provider/model-name
```

At startup and in the Prefect task log, Devflow prints the resolved provider, model, and endpoint.
