# ===========================================================================
# Makefile — Santander ML Pipeline
# Atalhos para o ciclo de desenvolvimento, qualidade e demo.
# Uso: make <target>   |   make help
# ===========================================================================
.DEFAULT_GOAL := help
PYTHON ?= python
export PYTHONPATH := .

.PHONY: help install test lint security pipeline api docker demo loadtest drift

help: ## Lista os targets disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Instala dependências de produção e desenvolvimento
	$(PYTHON) -m pip install --upgrade pip
	pip install -r requirements.txt -r requirements-dev.txt

test: ## Roda a suíte completa com cobertura (gate 80%)
	pytest -v --cov=src --cov-report=term-missing --cov-fail-under=80

lint: ## Formatação (black) + linting (flake8)
	black --check --diff src/
	flake8 src/ --max-line-length=120 --ignore=E501,W503,E402 --statistics --count

security: ## Gates de segurança locais (bandit + pip-audit)
	bandit -r src/ -ll -ii -x src/test_api.py --format txt
	pip-audit -r requirements.txt --progress-spinner off || true

pipeline: ## Executa o pipeline de ML (ingestão → preproc → treino)
	$(PYTHON) src/pipeline_manager.py

api: ## Sobe a API FastAPI em http://localhost:8000
	uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

docker: ## Build da imagem Docker
	docker build -t pima-diabetes-api:latest .

demo: ## Executa a demo end-to-end
	bash scripts/demo_end_to_end.sh

drift: ## Gera o relatório de Data Drift (PSI)
	$(PYTHON) src/generate_report.py

loadtest: ## Sobe o Locust (teste de carga) — ver docs/performance.md
	locust -f tests/performance/locustfile.py --host http://localhost:8000
