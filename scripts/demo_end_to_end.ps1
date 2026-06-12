###############################################################################
# Demo End-to-End — Santander ML Pipeline (Windows / PowerShell)
#
# Reproduz o ciclo completo para a banca:
#   deps -> testes -> pipeline -> API -> healthchecks -> predicao -> metricas -> stop
#
# Uso:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\scripts\demo_end_to_end.ps1
###############################################################################
$ErrorActionPreference = "Stop"

# Raiz do projeto (pasta pai de scripts/)
Set-Location (Join-Path $PSScriptRoot "..")
$env:PYTHONPATH = "."

Write-Host "==> 0/8  Preparando dataset (download se ausente - mesma fonte do CI)"
New-Item -ItemType Directory -Force -Path "data\raw" | Out-Null
if (-not (Test-Path "data\raw\pima_diabetes.csv")) {
    Write-Host "Dataset nao encontrado. Baixando do mirror publico UCI..."
    Invoke-WebRequest `
        -Uri "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv" `
        -OutFile "data\raw\pima_diabetes_raw.tmp"
    # Prepend do cabecalho - o arquivo publico vem sem nomes de colunas
    $header = "preg,plas,pres,skin,test,mass,pedi,age,class"
    $body = Get-Content "data\raw\pima_diabetes_raw.tmp"
    @($header) + $body | Set-Content "data\raw\pima_diabetes.csv" -Encoding utf8
    Remove-Item "data\raw\pima_diabetes_raw.tmp" -Force
    Write-Host "Dataset baixado: $((Get-Content 'data\raw\pima_diabetes.csv').Count) linhas"
}

Write-Host "==> 1/8  Instalando dependencias"
pip install -r requirements.txt -r requirements-dev.txt

Write-Host "==> 2/8  Executando suite de testes"
pytest -v

Write-Host "==> 3/8  Executando pipeline (treino + registro MLflow)"
python src/pipeline_manager.py

Write-Host "==> 4/8  Subindo API (uvicorn em background)"
$api = Start-Process -FilePath "uvicorn" `
  -ArgumentList "src.api:app", "--host", "0.0.0.0", "--port", "8000" -PassThru

try {
    Write-Host "==> 5/8  Aguardando readiness"
    for ($i = 0; $i -lt 30; $i++) {
        try {
            Invoke-RestMethod "http://localhost:8000/health/live" -TimeoutSec 2 | Out-Null
            break
        } catch { Start-Sleep -Seconds 1 }
    }
    Write-Host "Liveness :" (Invoke-RestMethod "http://localhost:8000/health/live" | ConvertTo-Json -Compress)
    try {
        Write-Host "Readiness:" (Invoke-RestMethod "http://localhost:8000/health/ready" | ConvertTo-Json -Compress)
    } catch {
        Write-Warning "Readiness retornou erro (modelo nao carregado?): $_"
    }

    Write-Host "==> 6/8  Predicao de exemplo (POST /predict)"
    $body = '{"preg":1.0,"plas":85.0,"pres":66.0,"skin":29.0,"test":0.0,"mass":26.6,"pedi":0.351,"age":31.0}'
    Invoke-RestMethod -Method Post "http://localhost:8000/predict" `
        -ContentType "application/json" -Body $body | ConvertTo-Json -Compress | Write-Host

    Write-Host "==> 7/8  Metricas Prometheus (ML customizadas)"
    (Invoke-WebRequest "http://localhost:8000/metrics").Content -split "`n" |
        Select-String "diabetes_predictions_total|diabetes_prediction_confidence" |
        Select-Object -First 10
}
finally {
    Write-Host "==> 8/8  Encerrando API"
    if ($api -and -not $api.HasExited) { Stop-Process -Id $api.Id -Force }
}

Write-Host "Demo concluida com sucesso."
