# 📊 Observability Stack — Santander ML Pipeline

Stack de monitoramento composta por **Prometheus** (coleta) + **Grafana** (visualização), consumindo métricas geradas pelo `prometheus-fastapi-instrumentator` no endpoint `/metrics` da API.

---

## 🚀 Quick Start

```bash
# Subir toda a stack (API + Prometheus + Grafana)
docker-compose -f docker-compose.observability.yml up -d

# Verificar se os serviços estão rodando
docker-compose -f docker-compose.observability.yml ps
```

| Serviço    | URL                          | Credenciais                          |
|------------|------------------------------|--------------------------------------|
| API        | http://localhost:8000        | —                                    |
| Prometheus | http://localhost:9090        | —                                    |
| Grafana    | http://localhost:3000        | `GF_SECURITY_ADMIN_*` (ver `.env`)   |

> **Credenciais do Grafana:** definidas via `GF_SECURITY_ADMIN_USER` /
> `GF_SECURITY_ADMIN_PASSWORD` no arquivo `.env` (copie de `.env.example`).
> Sem `.env`, o compose usa o fallback `admin` / `admin` — **troque antes de
> qualquer ambiente compartilhado.**

---

## 🔗 Configurar Datasource no Grafana

1. Acesse **Grafana** → ⚙️ **Configuration** → **Data Sources** → **Add data source**
2. Selecione **Prometheus**
3. Em **URL**, insira: `http://prometheus:9090`  
   _(nome do serviço na rede Docker, não `localhost`)_
4. Clique em **Save & Test**

---

## 📈 Queries PromQL para Dashboards

As métricas abaixo são geradas automaticamente pelo `prometheus-fastapi-instrumentator` e estão disponíveis em `/metrics`.

### 1. Taxa de Erro HTTP 5xx (Confiabilidade)

Mede a taxa de respostas com erro do servidor nos últimos 5 minutos. **SLO típico: < 1%.**

```promql
# Taxa de erro 5xx como percentual do total de requests (últimos 5 min)
sum(rate(http_requests_total{handler="/predict", status=~"5.."}[5m]))
/
sum(rate(http_requests_total{handler="/predict"}[5m]))
* 100
```

**Uso no Grafana:** Painel do tipo **Stat** ou **Gauge**, com thresholds:
- 🟢 `0 – 1%` (Saudável)
- 🟡 `1 – 5%` (Atenção)
- 🔴 `> 5%` (Crítico)

---

### 2. Latência P95 do Endpoint `/predict`

Mede o tempo de resposta do percentil 95 — "95% das requests são mais rápidas que este valor".

```promql
# Latência P95 em segundos (últimos 5 min)
histogram_quantile(
  0.95,
  sum(rate(http_request_duration_seconds_bucket{handler="/predict"}[5m])) by (le)
)
```

**Uso no Grafana:** Painel do tipo **Time Series** com unidade `seconds (s)`. Adicione linhas horizontais como referência:
- 🟢 `< 200ms` — Excelente
- 🟡 `200ms – 500ms` — Aceitável
- 🔴 `> 500ms` — Investigar (modelo pesado ou contenção de recursos)

---

### 3. Total de Requisições por Endpoint (Throughput)

Visão geral do volume de tráfego por endpoint e status code.

```promql
# Requests por segundo, agrupados por handler e status (últimos 5 min)
sum(rate(http_requests_total[5m])) by (handler, status)
```

**Uso no Grafana:** Painel do tipo **Time Series** com legenda `{{handler}} — {{status}}`. Permite identificar:
- Endpoints mais acessados
- Distribuição de status codes (2xx vs 4xx vs 5xx)
- Padrões de tráfego ao longo do dia

---

## 🏗️ Métricas Disponíveis (Referência)

Métricas expostas pelo `prometheus-fastapi-instrumentator`:

| Métrica | Tipo | Descrição |
|---|---|---|
| `http_requests_total` | Counter | Total de requests (labels: `method`, `handler`, `status`) |
| `http_request_duration_seconds` | Histogram | Duração das requests com buckets |
| `http_request_size_bytes` | Summary | Tamanho do corpo das requests |
| `http_response_size_bytes` | Summary | Tamanho do corpo das respostas |
| `http_requests_in_progress` | Gauge | Requests sendo processadas agora |

---

## 🧠 Métricas ML Customizadas

Além das métricas HTTP, a API expõe métricas de **negócio** no mesmo `/metrics`,
úteis para detectar drift de comportamento do modelo em tempo real:

| Métrica | Tipo | Descrição |
|---|---|---|
| `diabetes_predictions_total{resultado="positivo\|negativo"}` | Counter | Total de predições por classe |
| `diabetes_prediction_confidence` | Histogram | Distribuição da confiança (probabilidade máxima) |

### Taxa de predições positivas (PromQL)

```promql
sum(rate(diabetes_predictions_total{resultado="positivo"}[5m]))
/
sum(rate(diabetes_predictions_total[5m]))
```

> **Alerta de drift de conceito:** um salto sustentado na taxa de positivos
> (ex.: de ~35% para ~70%) sem mudança no perfil de entrada sugere que a
> população mudou ou que o modelo degradou — investigue com o Data Drift Report.

### Confiança média (PromQL)

```promql
rate(diabetes_prediction_confidence_sum[5m])
/
rate(diabetes_prediction_confidence_count[5m])
```

---

## 📉 Data Drift Report (PSI)

Complementa as métricas em tempo real com uma análise **offline/batch** de drift
de distribuição, baseada em **PSI (Population Stability Index)** — sem Evidently.

```bash
# Compara o dataset de treino (referência) com os logs de inferência
PYTHONPATH=. python src/generate_report.py
# Saída: reports/drift_report_YYYYMMDD_HHMMSS.json (+ .md)
```

Limiares de PSI por feature: `≤ 0.20` 🟢 estável · `0.20–0.25` 🟡 moderado ·
`> 0.25` 🔴 significativo. Detalhes em [`README.md` §3.5.3](../README.md).

---

## 📁 Estrutura de Arquivos

```
observability/
├── prometheus.yml              # Configuração de scrape do Prometheus
└── README.md                   # Esta documentação

docker-compose.observability.yml  # Orquestração da stack completa
```
