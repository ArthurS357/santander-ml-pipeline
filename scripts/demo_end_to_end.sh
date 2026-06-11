#!/usr/bin/env bash
###############################################################################
# Demo End-to-End — Santander ML Pipeline (Linux / macOS)
#
# Reproduz o ciclo completo para a banca:
#   deps → testes → pipeline → API → healthchecks → predição → métricas → stop
#
# Uso:
#   chmod +x scripts/demo_end_to_end.sh
#   ./scripts/demo_end_to_end.sh
###############################################################################
set -euo pipefail

# Raiz do projeto (pasta pai de scripts/)
cd "$(dirname "$0")/.."
export PYTHONPATH=.

echo "==> 1/8  Instalando dependências"
pip install -r requirements.txt -r requirements-dev.txt

echo "==> 2/8  Executando suíte de testes"
pytest -v

echo "==> 3/8  Executando pipeline (treino + registro MLflow)"
python src/pipeline_manager.py

echo "==> 4/8  Subindo API (uvicorn em background)"
uvicorn src.api:app --host 0.0.0.0 --port 8000 &
API_PID=$!
# Garante o encerramento da API mesmo em erro/Ctrl+C
trap 'echo "==> 8/8  Encerrando API (pid $API_PID)"; kill "$API_PID" 2>/dev/null || true' EXIT

echo "==> 5/8  Aguardando readiness"
for _ in $(seq 1 30); do
  if curl -fs http://localhost:8000/health/live >/dev/null 2>&1; then break; fi
  sleep 1
done
echo "Liveness : $(curl -s http://localhost:8000/health/live)"
echo "Readiness: $(curl -s http://localhost:8000/health/ready)"

echo "==> 6/8  Predição de exemplo (POST /predict)"
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"preg":1.0,"plas":85.0,"pres":66.0,"skin":29.0,"test":0.0,"mass":26.6,"pedi":0.351,"age":31.0}'
echo

echo "==> 7/8  Métricas Prometheus (ML customizadas)"
curl -s http://localhost:8000/metrics \
  | grep -E "diabetes_predictions_total|diabetes_prediction_confidence" \
  | head -n 10 || echo "(métricas ainda não populadas)"

echo "Demo concluída com sucesso."
