# One entry point for people, CI and Claude Code. Run `make` to list targets.
.DEFAULT_GOAL := help
UV       ?= uv
IMAGE    ?= ticket-agent:dev
PLATFORM ?= linux/arm64

.PHONY: help sync lock-check lint fmt test check run up down eval image

help: ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-17s %s\n", $$1, $$2}'

sync: ## Install exactly what uv.lock pins (with dev tools)
	$(UV) sync --locked

lock-check: ## Fail if uv.lock does not match pyproject.toml
	$(UV) lock --check

lint: ## Ruff lint and format check (no changes)
	$(UV) run --locked ruff check .
	$(UV) run --locked ruff format --check .

fmt: ## Fix lint findings and format the code
	$(UV) run --locked ruff check --fix .
	$(UV) run --locked ruff format .

test: ## Unit tests: no network, no real LLM
	$(UV) run --locked pytest -q

check: lock-check lint test ## Everything a gate needs

run: ## Run the service on :8080 with .env
	$(UV) run --locked --env-file .env uvicorn ticket_agent.main:app --reload --host 0.0.0.0 --port 8080

up: ## Start agent + mock-ticketing (FLAKY=1 make up for chaos mode)
	docker compose up --build

down: ## Stop the stack
	docker compose down

eval: ## Run the eval suite against the running stack
	$(UV) run --locked python eval/run_eval.py

image: ## Build the AgentCore image (linux/arm64)
	docker buildx build --platform $(PLATFORM) -t $(IMAGE) --load .
