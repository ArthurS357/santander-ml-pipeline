# setup_enterprise.ps1 — Ambiente corporativo restrito (proxy / SSL interceptado)
# ─────────────────────────────────────────────────────────────────────────────
# Para mirror interno, substitua o bloco pip por:
#   pip install --index-url http://pypi.intranet/simple `
#               --trusted-host pypi.intranet `
#               -r requirements.txt
# ─────────────────────────────────────────────────────────────────────────────

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "[1/4] Criando ambiente virtual..." -ForegroundColor Cyan
if (-Not (Test-Path "venv")) {
    python -m venv venv
}

Write-Host "[2/4] Ativando venv e instalando dependencias..." -ForegroundColor Cyan
& "venv\Scripts\Activate.ps1"

pip install `
    --trusted-host pypi.org `
    --trusted-host files.pythonhosted.org `
    --trusted-host pypi.python.org `
    -r requirements.txt

Write-Host "[3/4] Criando estrutura de pastas..." -ForegroundColor Cyan
@("data\raw", "data\processed", "data\logs", "reports", "mlruns") | ForEach-Object {
    New-Item -ItemType Directory -Force -Path $_ | Out-Null
}

Write-Host "[4/4] Definindo variaveis de ambiente da sessao..." -ForegroundColor Cyan
# RAW_DATA_URL: altere para .xlsx ou .parquet conforme necessario
$env:RAW_DATA_URL        = "data\raw\pima_diabetes.csv"
$env:MLFLOW_TRACKING_URI = "sqlite:///./mlflow.db"
$env:PYTHONPATH          = "."

Write-Host ""
Write-Host "Ambiente pronto. Execute o pipeline com:" -ForegroundColor Green
Write-Host "  python src/pipeline_manager.py" -ForegroundColor Yellow
