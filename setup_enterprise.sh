#!/usr/bin/env bash
# setup_enterprise.sh — Ambiente corporativo restrito (proxy / SSL interceptado)
# ─────────────────────────────────────────────────────────────────────────────
# Para mirror interno, substitua o bloco pip por:
#   pip install --index-url http://pypi.intranet/simple \
#               --trusted-host pypi.intranet \
#               -r requirements.txt
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

echo "[1/4] Criando ambiente virtual..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

echo "[2/4] Ativando venv e instalando dependencias..."
# shellcheck disable=SC1091
source venv/bin/activate

pip install \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host pypi.python.org \
    -r requirements.txt

echo "[3/4] Criando estrutura de pastas..."
mkdir -p data/raw data/processed data/logs reports mlruns

echo "[4/4] Exportando variaveis de ambiente..."
# RAW_DATA_URL: altere para .xlsx ou .parquet conforme necessario
export RAW_DATA_URL="data/raw/pima_diabetes.csv"
export MLFLOW_TRACKING_URI="sqlite:///./mlflow.db"
export PYTHONPATH="."

echo ""
echo "Ambiente pronto. Execute o pipeline com:"
echo "  python src/pipeline_manager.py"
