# Relatório de Estado do Sistema — Santander ML Pipeline

**Data de referência:** 2026-06-11
**Responsável:** Pipeline DevSecOps — Academia Santander
**Branch:** `main` | **Commit HEAD:** `e1291b4`

---

## 1. Estado dos Gates de Qualidade e Segurança

| Gate | Ferramenta | Resultado | Observação |
|---|---|---|---|
| Vulnerabilidades de dependências | `pip-audit` | **PASS** | 8 CVEs corrigidos (ver seção 3) |
| Análise estática de segurança (SAST) | `bandit` | **PASS** | Severidade MEDIUM+, confiança MEDIUM+ |
| Formatação de código | `black` | **PASS** | 9 arquivos — sem alterações necessárias |
| Linting estático | `flake8` | **PASS** | 0 ocorrências |
| Segredo embarcado na imagem | `trivy` / `checkov` | **PASS** | `ENV ADMIN_RELOAD_TOKEN` removido do Dockerfile |
| IaC / Manifestos Kubernetes | `checkov` | **PASS** | 89 checks aprovados, 3 suprimidos (PoC), 0 falhas |
| Cobertura de testes | `pytest-cov` | **PASS** | 94.44% (gate: ≥ 85%) |
| Suíte de testes | `pytest` | **PASS** | 123 aprovados, 0 falhas |

---

## 2. Cobertura de Testes por Módulo

| Módulo | Linhas | Cobertura |
|---|---|---|
| `src/__init__.py` | 0 | 100% |
| `src/api.py` | 138 | 99% |
| `src/config.py` | 9 | 100% |
| `src/data_ingestion.py` | 42 | 81% |
| `src/generate_report.py` | 103 | 97% |
| `src/pipeline_manager.py` | 86 | 100% |
| `src/preprocessing.py` | 47 | 100% |
| `src/test_api.py` | 46 | 96% |
| `src/train.py` | 249 | 90% |
| **TOTAL** | **720** | **94.44%** |

---

## 3. CVEs Resolvidos

| Pacote | Versão vulnerável | Versão corrigida | CVE / Reason |
|---|---|---|---|
| `cryptography` | `46.0.6` | `>=46.0.7` | CVE publicado |
| `idna` | `3.11` | `>=3.15` | CVE publicado |
| `urllib3` | `2.6.3` | `>=2.7.0` | CVE publicado |
| `GitPython` | `3.1.46` | `>=3.1.50` | CVE publicado |
| `Mako` | `1.3.10` | `>=1.3.12` | CVE publicado |
| `mlflow` / `mlflow-skinny` / `mlflow-tracing` | `3.10.1` | `>=3.11.1` | CVE publicado (alinhados) |
| `starlette` | `0.52.1` | `>=1.0.1` | CVE publicado |
| `pytest` | `9.0.2` | `>=9.0.3` | CVE publicado |
| `prometheus-fastapi-instrumentator` | `7.1.0` | `>=8.0.0` | Conflito de resolução (desbloqueia `starlette>=1.0.0`) |

---

## 4. Hardening de Infraestrutura Kubernetes

| Item | Antes | Depois |
|---|---|---|
| Namespace em todos os manifestos | ausente | `namespace: santander-ml` |
| Manifesto de namespace | ausente | `k8s/namespace.yaml` criado |
| `automountServiceAccountToken` | ausente (default: true) | `false` |
| `imagePullPolicy` | `IfNotPresent` | `Always` |
| Supressões Checkov | comentários ignorados | annotations `checkov.io/skipN` |
| Segredo no Dockerfile | `ENV ADMIN_RELOAD_TOKEN=""` | removido — injetado via K8s Secret |

---

## 5. Histórico de Commits do Ciclo de Hardening

| Hash | Mensagem | Escopo |
|---|---|---|
| `b7a75bf` | `fix(security): resolve CVEs in python dependencies` | `requirements.txt` |
| `efae648` | `fix(docker): remove sensitive ENV variable from dockerfile` | `Dockerfile` |
| `9bd485c` | `fix(k8s): enforce security policies and suppress PoC checkov rules` | `k8s/` (5 arquivos + `namespace.yaml`) |
| `bb2380e` | `fix(deps): bump prometheus-fastapi-instrumentator for starlette 1.x` | `requirements.txt` |
| `f2650af` | `fix(k8s): use checkov annotations for skips and defuse secret regex` | `k8s/deployment.yaml`, `k8s/configmap-secret.yaml` |
| `e1291b4` | `test(ci): allow mlflow file store for ephemeral test isolation` | `tests/conftest.py` |

---

## 6. Débitos Técnicos Documentados

| Item | Risco | Mitigação recomendada |
|---|---|---|
| Pins `>=` em `requirements.txt` | Builds não-reproduzíveis entre ambientes | Gerar `requirements.lock` via `pip-compile` |
| `MLFLOW_ALLOW_FILE_STORE=true` nos testes | Dependência de comportamento legado do MLflow | Migrar fixtures de integração para `sqlite://` efêmero |
| Tags Kubernetes (`CKV_K8S_43/35/14`) suprimidas | Imagem mutável, segredos como env vars | Em produção: digest SHA, montagem de arquivos para segredos |
