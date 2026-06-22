# Relatório de Sessão — Santander ML Pipeline V1.0.5

**Data de referência:** 2026-06-21 (GMT-3) — commits e CI em 2026-06-22 (UTC)
**Responsável:** Pipeline DevSecOps — Academia Santander
**Versão:** `V1.0.5`
**Branch:** `main` | **Base:** `c257c44` (v1.0.5)

---

## 1. Resumo Executivo

A sessão cobre **duas frentes complementares**, executadas com a mesma disciplina de menor risco e confirmação interativa antes de cada edição:

- **Parte A — Sincronização didática "Talk & Show" (commit `39e5d9a`):** auditoria cruzada entre o roteiro de apresentação e o código-fonte, inserindo **27 comentários-âncora** `# [TALK & SHOW: Slide N]` acima das declarações que o apresentador exibe ao vivo, e harmonizando o roteiro (correção de nomes de script, contagem de passos, nota técnica `uv` e novo Apêndice G de Big Data). **Zero alteração de lógica, tipagem ou AST.**
- **Parte B — Hotfix de CI (commit `7e390e0`, PR #1, merge `9184353`):** o gate de pipeline do GitHub Actions passou a falhar no passo de treino com `Untrusted types found in the file: ['numpy.dtype']`. Causa raiz: **drift de dependência** — `mlflow>=3.11` passou a serializar o flavor sklearn via **skops** por padrão. Correção cirúrgica de 1 kwarg (`serialization_format="cloudpickle"`) na única chamada `log_model`.

Resultado consolidado: **CI verde novamente** (runs #24 e #25), **128 testes aprovados**, pipeline end-to-end reproduzido com sucesso, e o roteiro alinhado à realidade da v1.0.5 sem desestabilizar o código congelado.

---

## 2. Parte A — Âncoras Visuais "Talk & Show" no Código

### 2.1 Diagnóstico

O roteiro promete "mostrar ao vivo no VS Code" diversas declarações-chave por slide, mas elas não tinham um marcador visual que permitisse ao apresentador localizá-las em milissegundos. Decisão: inserir comentários padronizados `# [TALK & SHOW: Slide N] …`, **somente comentários** — sem tocar lógica, ordem de execução ou tipos.

### 2.2 Comentários inseridos (27 ao todo)

| Arquivo | Slide(s) | Âncoras |
|---|---|---|
| `src/pipeline_manager.py` | 3 | `MLPipelineOrchestrator` + as 5 etapas (`run_ingestion`, `run_preprocessing`, `run_training`, `run_reporting`, `run_pipeline`) — 6 marcações |
| `src/train.py` | 4 / 5 | `SUPPORTED_SELECTION_METRICS`, `DEFAULT_SELECTION_METRIC`, `REGISTRY_NAME`, `CHAMPION_ALIAS`, `stratify=y`, dict `models`, `DatasetSnapshotRecord`, `_calculate_sha256`, `_log_model_with_signature`, `_register_with_champion_alias` — 10 marcações |
| `src/api.py` | 6 / 7 | `PatientData`, `/predict`, `/health/live`, `/health/ready`, `/admin/reload_model`, `PREDICTION_COUNTER`, `log_prediction` — 7 marcações |
| `src/generate_report.py` | 7 | `PSI_MODERATE`/`PSI_HIGH`, `_MIN_CURRENT_ROWS`, `calculate_psi`, `generate_data_drift_report` — 4 marcações |

### 2.3 Garantia de qualidade (FASE py-code-health / caveman-lite)

- **Impacto de AST:** nulo — `py_compile` OK nos 4 módulos; comentários não entram na AST.
- **Limite de linha:** todas as âncoras ≤ 120 caracteres (máx. observado: **99**).
- **Lint:** `ruff check` → `All checks passed!` nos 4 arquivos.

### 2.4 Harmonização do Roteiro (documento externo ao repositório)

O roteiro (`Roteiro_AtualizadoV1.docx`) foi editado via `python-docx`, preservando formatação (substituições em nível de *run*). Correções factuais aplicadas:

| Item | Antes | Depois |
|---|---|---|
| Nome do script de setup | `setup_enterprise_uv.ps1` (inexistente) | `setup_enterprise.ps1` (real no disco) — 16 ocorrências |
| Contagem de passos do demo | `passos 0/9 a 9/9`, `passo 5/9` | `passos 0/8 a 8/8`, `passo 3/8` (alinhado ao `scripts/demo_end_to_end.ps1`) |
| Narrativa `uv` | mantida | **mantida** + nova *nota técnica* explicando `uv pip install` sobre o `.venv` padrão |
| Escalabilidade | ausente | **novo Apêndice G — Defesa de Escalabilidade para Big Data** |

> **Decisão (confirmada interativamente):** o roteiro mantém a narrativa `uv` na fala, mas os **nomes de arquivo** apontam para os `.ps1` reais (que usam `pip` + `venv`). Os scripts **não** foram renomeados nem alterados — apenas a documentação. Ver [memória de projeto `uv-narrative-is-doc-only`].

### 2.5 Apêndice G — Tese de Big Data (fundamentada no código real)

O argumento de engenharia entregue ancora-se em mecanismos já existentes: chaveamento automático `src/config.py::use_dask_mode` (USE_DASK ou > 500 MB); treino out-of-core `_train_incremental` via `pandas.read_csv(chunksize=…)` + `SGDClassifier.partial_fit` com holdout por buffer rolante; hash `_calculate_sha256` em blocos de 1 MB; logging de inferência assíncrono via `BackgroundTasks` (evoluível para streaming/Parquet); drift incremental (PyArrow/Polars); e escala horizontal stateless via alias `@champion` + HPA no Kubernetes. A postura de segurança exige Secret Manager, IAM de menor privilégio, TLS + criptografia em repouso e tratamento de logs clínicos como dados sensíveis.

---

## 3. Parte B — Hotfix de CI: `mlflow` skops → cloudpickle

### 3.1 Sintoma

O passo `Run Orchestrated Pipeline (CI)` (`python src/pipeline_manager.py`) falhou no treino do RandomForest:

```
ERROR - Erro no Treinamento: The saved sklearn model references untrusted types.
If you are sure loading these types is safe, set the 'skops_trusted_types' parameter
when calling 'log_model' or 'save_model' ... Root error: Untrusted types found in the
file: ['numpy.dtype'].
=== Pipeline finalizado com FALHA ===
```

### 3.2 Diagnóstico — drift de dependência

| | Ambiente local (passava) | CI / GitHub Actions (falhava) |
|---|---|---|
| `mlflow` | **3.10.1** → default `serialization_format=cloudpickle` | **≥ 3.11.1** (piso aberto em `requirements.txt`) → usa **skops** |
| Efeito no `log_model` | sem checagem de tipos | skops bloqueia `numpy.dtype` (presente em qualquer `Pipeline`/`SimpleImputer`) |

`requirements.txt` fixa quase tudo com `==`, mas mantém `mlflow>=3.11.1` (linhas 47–49) e `skops==0.13.0` (linha 83). O CI instala o mlflow mais recente, que adota skops por padrão no flavor sklearn — clássico cenário *verde-local / vermelho-CI*. **Sem relação com os comentários da Parte A.**

### 3.3 Prova do mecanismo (reproduzido na máquina local, mlflow 3.10.1)

Forçando cada formato sobre o mesmo `Pipeline(SimpleImputer + RandomForest)`:

```
RESULT [skops]       -> FAIL: MlflowException: The saved sklearn model references
                                untrusted types ... set 'skops_trusted_types' ...
RESULT [cloudpickle] -> OK (round-trip)
```

O erro do CI foi reproduzido deterministicamente via `serialization_format='skops'`, e `cloudpickle` faz o round-trip (log + load) sem erro.

### 3.4 Correção (`src/train.py` — `_log_model_with_signature`)

```python
# Antes:
    return mlflow.sklearn.log_model(
        model, "model", signature=signature, input_example=X_sample
    )

# Depois:
    # serialization_format fixo em cloudpickle (default histórico do MLflow): a
    # partir de mlflow>=3.11 o flavor sklearn passou a usar skops por padrão,
    # cuja checagem de tipos rejeita numpy.dtype e quebra o log_model no CI.
    return mlflow.sklearn.log_model(
        model,
        "model",
        signature=signature,
        input_example=X_sample,
        serialization_format="cloudpickle",
    )
```

Ponto único de log de modelo — a correção cobre tanto o fluxo padrão (`_train_standard`) quanto o incremental/Big Data (`_train_incremental`). Respeita o piso `mlflow>=3.11.1` (não rebaixa dependência) e restaura o comportamento sob o qual o projeto foi validado.

### 3.5 Alternativas avaliadas e descartadas

| Opção | Motivo da rejeição |
|---|---|
| Pinar `mlflow` em `requirements.txt` | A versão boa conhecida (3.10.1) está **abaixo** do piso `>=3.11.1`; geraria conflito de intenção. |
| Manter skops + `skops_trusted_types=[…]` | Frágil — exigiria enumerar múltiplos tipos numpy/sklearn e só vale nas versões que usam skops. |
| **Forçar `cloudpickle` (adotada)** | 1 linha, version-robusta, cobre ambos os fluxos, seguro para modelo próprio (carregado pelo mesmo pipeline via `mlflow.sklearn.load_model` em `src/api.py`). |

---

## 4. Validações Realizadas

| Verificação | Comando / Evidência | Resultado |
|---|---|---|
| Sintaxe / AST dos 4 módulos | `python -m py_compile …` | **PASS** — ALL COMPILE OK |
| Limite de linha das âncoras | medição programática | **PASS** — máx. 99 ≤ 120 |
| Lint | `ruff check src/…` | **PASS** — All checks passed |
| Suíte completa | `pytest -q` | **PASS** — 128 aprovados, 0 falhas |
| Pipeline end-to-end (igual ao CI) | `python src/pipeline_manager.py` (tracking limpo) | **PASS** — RandomForest registrado, alias `@champion` atualizado, "SUCESSO em 11.23s" |
| Repro skops × cloudpickle | script isolado | **PASS** — skops FAIL / cloudpickle OK |
| GitHub Actions — branch do fix | run **#24** (PR #1) | **VERDE** |
| GitHub Actions — merge na `main` | run **#25** (`9184353`) | **VERDE** |

---

## 5. Revisão de Segurança

| Item | Status |
|---|---|
| Âncoras `# [TALK & SHOW]` | Comentários puros — sem segredos, sem lógica |
| Tese de Big Data (Apêndice G) | Auditada — exige Secret Manager, IAM menor privilégio, TLS + criptografia em repouso, sem bypass de SSL, logs clínicos como dados sensíveis; **não** sugere prática insegura |
| Trade-off `cloudpickle` | Aceitável — formato é o default histórico do MLflow; o modelo é **autoproduzido** e recarregado pelo próprio pipeline (`MLFLOW_ALLOW_PICKLE_DESERIALIZATION` default `True`). Risco de desserialização arbitrária limitado a artefato confiável interno |
| Contrato `/predict`, fail-secure do `ADMIN_RELOAD_TOKEN`, validação `PatientData` | Inalterados |

**Conclusão:** nenhuma superfície de ataque alterada; nenhuma informação sensível exposta.

---

## 6. Arquivos Alterados

```
# Parte A — commit 39e5d9a (âncoras Talk & Show)
src/pipeline_manager.py |  6 +    (6 âncoras de slide)
src/train.py            | 10 +    (10 âncoras de slide)
src/api.py              |  7 +    (7 âncoras de slide)
src/generate_report.py  |  4 +    (4 âncoras de slide)
Roteiro_AtualizadoV1.docx (externo) | nomes de script, passos, nota uv, Apêndice G

# Parte B — commit 7e390e0 (hotfix de CI)
src/train.py            |  8 ++++++++-  (serialization_format="cloudpickle" + comentário)
```

Todas as edições em `src/` são comentários ou 1 kwarg de serialização. Zero mudança de contrato, zero mudança de comportamento de treino/serving.

---

## 7. Histórico de Commits desta Sessão

| Hash | Tipo | Escopo | Descrição | CI |
|---|---|---|---|---|
| `39e5d9a` | `docs` | `talk-and-show` | Âncoras visuais no código + Apêndice de Big Data | run #23 ❌ (skops) |
| `7e390e0` | `fix` | `train` | Forçar cloudpickle no `log_model` para mitigar bloqueio do skops | run #24 ✅ |
| `9184353` | `merge` | `PR #1` | Merge `fix/mlflow-skops-serialization` → `main` | run #25 ✅ |

---

## 8. Impacto e Riscos

| Dimensão | Avaliação |
|---|---|
| **Compatibilidade de API** | Zero — `PredictionResponse` inalterado |
| **Comportamento de treino** | Zero — apenas o formato de serialização do artefato muda (cloudpickle, default histórico) |
| **Serving** | Zero — `mlflow.sklearn.load_model` lê o formato gravado no MLmodel |
| **Cobertura de testes** | Mantida (gate 85% ✓); nenhum teste adicionado/removido |
| **Risco de regressão** | Nulo — 128 testes aprovados; CI verde em branch e em `main` |

---

## 9. Riscos Residuais e Próximos Passos

1. **`mlflow` com piso aberto (`>=3.11.1`):** o `serialization_format="cloudpickle"` neutraliza o risco do skops, mas futuras versões podem introduzir outros defaults. Considerar pinagem de teto (`mlflow>=3.11.1,<4`) numa próxima janela de manutenção para reprodutibilidade total.
2. **Ambiente local com `mlflow.db` de schema antigo:** rodar `python src/pipeline_manager.py` localmente sem `MLFLOW_TRACKING_URI` limpo dispara "out-of-date database schema". O CI começa limpo e não é afetado. Mitigação local: tracking store novo ou `mlflow db upgrade sqlite:///mlflow.db`.
3. **Avisos cosméticos do Pyrefly em `generate_report.py`** (`unnecessary-type-conversion` em casts `float()/int()`): benignos (`np.float64 ⊂ float`); `ruff` do CI já passa limpo. Decisão registrada: **manter o arquivo intacto**.
