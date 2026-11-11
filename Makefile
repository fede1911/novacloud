# Blueprint commands: run them from the blueprint folder
PYTHON ?= .venv/bin/python

.PHONY: setup ingest search chat chat-drive bench bench-offline bench-retrieval test lint format

setup:  ## Create the virtualenv with the pinned dependencies
	uv venv --python 3.13 .venv && uv pip install --python $(PYTHON) -r requirements-dev.txt

ingest:  ## Rebuild the knowledge base index with Titan embeddings
	$(PYTHON) -m ingestion.cli

search:  ## Search the existing index: make search Q="mi ridate i soldi del piano annuale?" SCOPE=novabill
	@test -n "$(Q)" -a -n "$(SCOPE)" || { echo 'usage: make search Q="question" SCOPE=novabill'; exit 1; }
	$(PYTHON) -m ingestion.cli --query "$(Q)" --scope "$(SCOPE)"

chat:  ## Chat through the triage
	$(PYTHON) -m agent.cli

chat-drive:  ## Chat with the scope fixed to NovaDrive (no triage)
	$(PYTHON) -m agent.cli --scope novadrive

bench:  ## Run the benchmark with the LLM judges
	$(PYTHON) -m benchmark.cli

bench-offline:  ## Run the benchmark with only the deterministic metrics
	$(PYTHON) -m benchmark.cli --no-judge

bench-retrieval:  ## Evaluate the retrieval alone (embedding calls only, no LLM)
	$(PYTHON) -m benchmark.retrieval_cli

test:  ## Unit and integration tests (no network)
	$(PYTHON) -m pytest -m "not slow"

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format . && $(PYTHON) -m ruff check --fix .
