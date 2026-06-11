# Relatório de Performance — Teste de Carga

Teste de carga da API de inferência usando **Locust**
(`tests/performance/locustfile.py`).

## Pré-requisitos

```bash
pip install -r requirements-dev.txt   # inclui locust
```

> ⚠️ **Rate limiting:** o `/predict` aplica `10/minute` por IP por padrão.
> Para um teste de carga representativo, suba a API com um limite alto:
>
> ```bash
> # Linux/macOS
> PREDICT_RATE_LIMIT="1000000/minute" uvicorn src.api:app --host 0.0.0.0 --port 8000
>
> # Windows (PowerShell)
> $env:PREDICT_RATE_LIMIT="1000000/minute"; uvicorn src.api:app --host 0.0.0.0 --port 8000
> ```

## Execução

```bash
# 1. Suba a API (com o modelo já treinado: rode o pipeline antes)
python src/pipeline_manager.py
uvicorn src.api:app --host 0.0.0.0 --port 8000

# 2. Em outro terminal, inicie o Locust
make loadtest
# ou:
locust -f tests/performance/locustfile.py --host http://localhost:8000
```

Abra `http://localhost:8089`, defina o número de usuários (ex.: 50) e a taxa de
spawn (ex.: 10/s). O cenário alterna `POST /predict` (peso 4) e
`GET /health/ready` (peso 1), com `wait_time` de 1–3 s por usuário.

### Modo headless (CI/relatório automatizado)

```bash
locust -f tests/performance/locustfile.py --host http://localhost:8000 \
  --headless -u 50 -r 10 -t 1m --csv reports/loadtest
```

## Resultados Esperados (referência)

Hardware modesto (4 vCPU), modelo Random Forest, 50 usuários simultâneos:

| Métrica | Valor de referência |
|---|---|
| Throughput | ~30–80 req/s (limitado por `wait_time` e CPU) |
| Latência mediana (p50) | < 50 ms |
| Latência p95 | < 200 ms |
| Taxa de erro (5xx) | ~0% (com modelo carregado) |

> Os números variam conforme hardware e modelo selecionado. O objetivo é
> demonstrar estabilidade sob carga e validar os SLOs documentados em
> `observability/README.md` (p95 < 200 ms, erro 5xx < 1%).

## Correlação com Observabilidade

Durante o teste, observe no Grafana/Prometheus:

- `http_request_duration_seconds` (latência p95).
- `diabetes_predictions_total` (volume e distribuição de classes).
- `diabetes_prediction_confidence` (distribuição de confiança sob carga).
