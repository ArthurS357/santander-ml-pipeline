# Roteiro de Apresentação — Santander ML Pipeline (1h30)

Guia cronometrado para a banca de Engenharia de Machine Learning, com blocos de
conteúdo e **comandos ao vivo**. Pré-requisito: ambiente preparado
(`make install`) e dataset disponível.

---

## ⏱️ Timeline

| Tempo | Bloco | Foco |
|---|---|---|
| 0–10 min | 1. Contexto & Arquitetura | Problema, 6 pilares, visão de DAG |
| 10–25 min | 2. Pipeline de ML ao vivo | Ingestão → preproc → treino → MLflow |
| 25–40 min | 3. Governança de Modelos | Signature, alias `@champion`, snapshots |
| 40–55 min | 4. API & Validação | `/predict`, Pydantic forte, rate limiting |
| 55–70 min | 5. Observabilidade & Drift | Prometheus, métricas ML, PSI |
| 70–80 min | 6. CI/CD & DevSecOps | Gates, K8s, escalabilidade |
| 80–90 min | 7. Q&A & Diferenciais | Model/Data Card, ADRs, performance |

---

## Bloco 1 — Contexto & Arquitetura (0–10 min)

- Abrir o `README.md` → seção **Objetivo** e tabela dos **6 pilares**.
- Mostrar os diagramas Mermaid (DAG + arquitetura técnica).
- Citar decisões em [`docs/adr/`](adr/).

## Bloco 2 — Pipeline ao vivo (10–25 min)

```bash
make pipeline        # ingestão → preproc → treino multi-modelo
mlflow ui            # abrir http://localhost:5000
```

- No MLflow UI: comparar accuracy/F1 entre RF, LR, SVM; mostrar o run vencedor.

## Bloco 3 — Governança de Modelos (25–40 min)

- MLflow Registry → `PimaDiabetesClassifier`, versão e **alias `@champion`**.
- Mostrar **signature** no `MLmodel`.
- Explicar [`docs/model_card.md`](model_card.md) e [`docs/data_card.md`](data_card.md).
- Snapshot com `dataset_sha256` (rastreabilidade dado→modelo).

## Bloco 4 — API & Validação (40–55 min)

```bash
make api             # sobe a API em :8000
```

```bash
# Swagger
# http://localhost:8000/docs

# Predição válida (note os floats: 1.0, não 1)
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"preg":1.0,"plas":85.0,"pres":66.0,"skin":29.0,"test":0.0,"mass":26.6,"pedi":0.351,"age":31.0}'

# Validação forte: payload fora de faixa → 422
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"preg":1.0,"plas":0.0,"pres":66.0,"skin":29.0,"test":0.0,"mass":26.6,"pedi":0.351,"age":31.0}'

# Rate limiting (10/min) → repetir o POST acima >10x mostra 429
```

## Bloco 5 — Observabilidade & Drift (55–70 min)

```bash
# Métricas ML customizadas
curl -s http://localhost:8000/metrics | grep diabetes_

# Stack completa (Prometheus + Grafana)
docker-compose -f docker-compose.observability.yml up -d

# Data Drift ao vivo (gera reports/drift_report_*.json e .md)
make drift
```

> `reports/` é gitignored — o relatório é gerado **ao vivo** na apresentação.

- Abrir o `.md` gerado e mostrar PSI por feature + alertas.

## Bloco 6 — CI/CD & DevSecOps (70–80 min)

- Abrir `.github/workflows/ci.yml`: Security Gate (black, flake8, bandit),
  **DevSecOps** (gitleaks, pip-audit, checkov, trivy), testes com
  `--cov-fail-under=85`, build + push GHCR.
- Mostrar `k8s/` (Deployment, HPA, probes separadas).
- (Opcional) teste de carga: ver [`docs/performance.md`](performance.md).

## Bloco 7 — Q&A & Diferenciais (80–90 min)

- ADRs ([`docs/adr/`](adr/)), Model/Data Card, mapa de evidências no README.
- Limitações conhecidas e roadmap (PostgreSQL, calibração, A/B testing).

---

## Plano B (sem internet / ambiente restrito)

Use `setup_enterprise.ps1` / `.sh` e o "Modo Caverna" do README (dataset local
em `data/raw/`). Todo o fluxo roda offline com MLflow em SQLite.
