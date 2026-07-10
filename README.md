
Setup:
===============================
cd ~/projects/devflow
source .venv/bin/activate

pip install -U prefect langgraph

#for ollama
python -m pip install -U langchain-ollama


when using
===============================
cd ~/projects/devflow
source .venv/bin/activate


Scripts:
================================

text_review
------------
how to run:

cd ~/projects/devflow
OLLAMA_BASE_URL=http://YOUR_OLLAMA_IP_HERE:11434 \
OLLAMA_MODEL=your_modelname_here \
PYTHONPATH=src python scripts/run_text_review.py


what it does:

has an AI review text (for testing basic flow of Prefect, does not do anything useful)


code_review
------------
how to run:

cd ~/projects/devflow
OLLAMA_BASE_URL=http://YOUR_OLLAMA_IP_HERE:11434 \
OLLAMA_MODEL=your_modelname_here \
PYTHONPATH=src python scripts/run_code_review.py \
  --repo ~/projects/REPOPATH \
  --base upstream/master \
  --check "echo 'CI CHECK COMMAND GOES HERE including echo'" \
  --test "echo 'CI TEST COMMAND GOES HERE including echo'"

what it does:

-code review 
-run some CI steps specified in check and test
-output evidence and results to: output/code-review/


