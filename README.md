
Setup:
===============================
cd ~/projects/devflow
source .venv/bin/activate

pip install -U prefect langgraph

#for ollama
python -m pip install -U langchain-ollama

#openrouter
python -m pip install -U langchain_openai


when using
===============================
#to start server
cd ~/projects/devflow
source .venv/bin/activate
prefect server start

#when running commands
cd ~/projects/devflow
source .venv/bin/activate


Script List:
===========
devflow review - code review (and some relevant CI tasks like running tests)
devflow plan - Serena-backed structured implementation planning and plan refinement
devflow serena-context - relevant file selection using serena MCP
devflow implement - propose, approve, apply, and validate edits from a ready plan

# Run Serena, then create a plan
devflow plan "request"

# Auto-approve supplemental context and replanning gates
devflow plan --yes "request"

# Open plan.json after the run without the final prompt
devflow plan --open-plan "request"

# Propose edits from an approved plan, then ask before applying them
devflow implement /path/to/plan.json

# Apply a valid proposal without the apply prompt
devflow implement --yes /path/to/plan.json

# Reuse Serena context
devflow plan --context /path/to/context.json "request"

# Run Serena with the previous plan, then refine it
devflow plan --from-plan /path/to/plan.json "request"

# Reuse context and refine a plan
devflow plan \
  --context /path/to/context.json \
  --from-plan /path/to/plan.json \
  "request"

#example with specified model
devflow plan \
  --context /path/to/context.json \
  --from-plan /path/to/plan.json \
  --provider openrouter \
  --model provider/model-name \
  "request"
  
  
  
Proof-of-concept text review
==============================
#how to run:
cd ~/projects/devflow
OLLAMA_BASE_URL=http://YOUR_OLLAMA_IP_HERE:11434 \
OLLAMA_MODEL=your_modelname_here \
PYTHONPATH=src python scripts/run_text_review.py


#what it does
has an AI review text (for testing basic flow of Prefect, does not do anything useful)


develop workflow
=================
devflow serena-context "development goal here"
(do coding)



Recommended code-review workflow
================================

Install devflow in editable mode from this repository:

cd ~/projects/devflow
clone the devflow repo


#when using
source .venv/bin/activate

#one-time to do editable install
pip install -e .


#Copy the example configuration into the Git repository you want to review:
#e.g.    cp /path/to/devflow/.devflow.example.toml /path/to/project/.devflow.toml
#Edit `.devflow.toml`, then from within your project folder you can use devflow commands.


#before running any of the devflow commands
cd ~/projects/projectname
source .venv/bin/activate


#in the project you want to review, run devflow command
devflow review

#To open the generated Markdown report after the run:
devflow review --open


Each run writes `review.md`, `review.json`, and `evidence.json` beneath the configured output directory. A `latest` symlink points to the newest run.

For Ollama, set the model and server URL in `.devflow.toml` or use `OLLAMA_MODEL` and `OLLAMA_BASE_URL`.

For OpenRouter, set `provider = "openrouter"` and export `OPENROUTER_API_KEY`.


Configuration setup
===================
#global config (preferred model/provider, etc.)
#copy from example: .devflow.global.example.toml to
~/.config/devflow/config.toml
#end edit

#project level config (define repo path, testing commands, etc.)
#copy from example: .devflow.example.toml to 
$PROJECT_FOLDER/.devflow.toml
#and edit
#can override global

#temporary overrides can be used in shell commands

#parameters can also be specified in the command
