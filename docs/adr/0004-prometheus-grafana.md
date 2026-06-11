# ADR 0004 — Prometheus + Grafana para observabilidade

## Status

Aceito.

## Contexto

O pilar de Observabilidade exige métricas de desempenho em tempo real.
Alternativas: Datadog, New Relic (SaaS pagos).

## Decisão

Adotar **Prometheus** (coleta/TSDB) + **Grafana** (visualização), com
`prometheus-fastapi-instrumentator` expondo `/metrics`.

- Métricas HTTP automáticas (latência, throughput, status).
- **Métricas ML customizadas** no mesmo `/metrics`:
  `diabetes_predictions_total{resultado}` (Counter) e
  `diabetes_prediction_confidence` (Histogram) — para detectar drift de negócio.
- Stack isolada em `docker-compose.observability.yml`; healthcheck usa
  `/health/live` (coleta de métricas mesmo antes do modelo carregar).
- Análise batch complementar de drift via PSI (`generate_report.py`).

## Consequências

- **Positivas:** open-source, scrape-ready, sem servidor externo pago; cobre
  tanto SLOs de infra quanto sinais de ML.
- **Negativas / trade-offs:** credenciais do Grafana via `.env` (não versionado);
  retenção do Prometheus limitada a 15d na PoC.
