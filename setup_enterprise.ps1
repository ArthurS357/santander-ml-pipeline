# setup_enterprise.ps1 — Ambiente corporativo restrito (proxy / SSL interceptado)
# ─────────────────────────────────────────────────────────────────────────────
# Para mirror interno, substitua o bloco pip por:
#   pip install --index-url http://pypi.intranet/simple `
#               --trusted-host pypi.intranet `
#               -r requirements.txt
# ─────────────────────────────────────────────────────────────────────────────

# setup_enterprise.ps1 — Ambiente corporativo restrito
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# TODO: Em ambientes corporativos, substitua as URLs abaixo pelo Nexus/Artifactory interno.
$INTERNAL_PYPI_URL = "https://pypi.org/simple"
$TRUSTED_HOST = "pypi.org"
$TRUSTED_HOST_2 = "files.pythonhosted.org"

Write-Host "[1/4] Criando ambiente virtual..." -ForegroundColor Cyan
if (-Not (Test-Path "venv")) {
    python -m venv venv
}

Write-Host "[2/4] Ativando venv e instalando dependencias..." -ForegroundColor Cyan
& "venv\Scripts\Activate.ps1"

python -m pip install --upgrade pip --index-url $INTERNAL_PYPI_URL --trusted-host $TRUSTED_HOST --trusted-host $TRUSTED_HOST_2

Write-Host "Instalando requirements..." -ForegroundColor Cyan
pip install --index-url $INTERNAL_PYPI_URL --trusted-host $TRUSTED_HOST --trusted-host $TRUSTED_HOST_2 -r requirements.txt
pip install --index-url $INTERNAL_PYPI_URL --trusted-host $TRUSTED_HOST --trusted-host $TRUSTED_HOST_2 -r requirements-dev.txt

Write-Host "[3/4] Criando estrutura de pastas..." -ForegroundColor Cyan
@("data\raw", "data\processed", "data\logs", "reports", "mlruns") | ForEach-Object {
    New-Item -ItemType Directory -Force -Path $_ | Out-Null
}

Write-Host "[4/4] Definindo variaveis de ambiente da sessao..." -ForegroundColor Cyan
$env:RAW_DATA_URL        = "data\raw\pima_diabetes.csv"
$env:MLFLOW_TRACKING_URI = "sqlite:///./mlflow.db"
$env:PYTHONPATH          = "."

Write-Host "`nAmbiente pronto. Execute o pipeline com: python src/pipeline_manager.py" -ForegroundColor Green