.PHONY: help sync test lint format demo clean data baseline forecast regime signal graph agent scheduled-session

help:
	@echo "AutoSignal-X — modular research instrument for predictive structure discovery"
	@echo ""
	@echo "Setup:"
	@echo "  make sync          Install dependencies via uv (creates .venv, writes uv.lock)"
	@echo ""
	@echo "Quality:"
	@echo "  make test          Run all tests"
	@echo "  make lint          Lint with ruff"
	@echo "  make format        Auto-format with ruff"
	@echo ""
	@echo "Demo:"
	@echo "  make demo          Launch the Streamlit research cockpit"
	@echo ""
	@echo "Pipeline targets are added per iteration; see README.md for the full list."
	@echo "No make? Each target maps to one 'uv run …' command — see README."

sync:
	uv sync --all-extras

test:
	uv run pytest

lint:
	uv run ruff check src/ tests/ app/

format:
	uv run ruff format src/ tests/ app/
	uv run ruff check --fix src/ tests/ app/

demo:
	uv run streamlit run app/streamlit_app.py

# --- pipeline targets (added per iteration) ---

data:
	uv run autosignalx data fetch

baseline:
	uv run autosignalx eval baseline

forecast:
	uv run autosignalx eval chronos

regime:
	uv run autosignalx regime fit

signal:
	uv run autosignalx signal rank

signal-stability:
	uv run autosignalx signal stability

graph:
	uv run autosignalx graph build

graph-per-regime:
	uv run autosignalx graph build-per-regime

agent:
	uv run autosignalx agent run --fresh

chat-index:
	uv run autosignalx chat index

chat-eval:
	uv run autosignalx chat eval

snapshot:
	uv run autosignalx snapshot build

harden:
	uv run autosignalx agent harden

scheduled-session:
	bash scripts/run_session.sh

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
