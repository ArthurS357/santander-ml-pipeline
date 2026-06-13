# Relatório de Sessão — Santander ML Pipeline V1.0.4

**Data de referência:** 2026-06-13
**Responsável:** Pipeline DevSecOps — Academia Santander
**Versão:** `V1.0.4`
**Branch:** `main` | **Base:** `fe2614f` (v1.0.3)

---

## 1. Resumo Executivo

A sessão **v1.0.4** resulta de uma auditoria de aderência entre os documentos de especificação do projeto (apresentação PPTX + roteiro talk-and-show) e o código-fonte real do repositório. A auditoria identificou quatro divergências (GAP-01 a GAP-04). Todas as decisões de correção adotaram a opção de **menor risco**, confirmada interativamente antes de qualquer edição:

- **GAP-01 — Docstring do orquestrador** (`src/pipeline_manager.py`): a descrição da classe prometia "Ingestão → Pré-processamento → Treinamento → Monitoramento → Reporting", mas `run_pipeline()` executa apenas as três primeiras etapas. Docstring corrigida para refletir a realidade; comportamento inalterado.
- **GAP-02 — Nomenclatura "Dask + SGDClassifier"** (`src/config.py`, `src/train.py`): o nome `use_dask_mode` e comentários anteriores sugeriam que Dask atuava no treino. Docstring e comentário de chaveamento corrigidos para deixar explícito que Dask é exclusivo da camada de dados (ingestão/pré-processamento) e que o treino Big Data usa `pandas.read_csv(chunksize=...)` + `SGDClassifier.partial_fit`. Comportamento inalterado. Durante a mesma intervenção em `src/train.py`, foram removidos dois `cast(np.ndarray, …)` redundantes em `IncrementalDiabetesModel` (diagnóstico Pyrefly `redundant-cast`).
- **GAP-03 — Localização de `MODEL_URI`** (`k8s/deployment.yaml`): o roteiro instrui a mostrar `MODEL_URI` no `deployment.yaml`, mas ele está em `k8s/configmap.yaml` (injetado via `envFrom`) — o que é funcionalmente correto e evita fonte dupla de verdade. Comentário-âncora adicionado ao `envFrom` do deployment apontando para o ConfigMap; nenhum valor duplicado.
- **GAP-04 — Campos de resposta `/predict`**: o roteiro cita `prediction`/`probability`, mas o contrato real de `PredictionResponse` usa `predicao`/`confianca`/`modelo_versao`/`latencia_s`. Decisão: manter o código; a correção compete ao roteiro/PPTX (fora deste repositório). Nenhuma edição em `src/api.py`.

Resultado: **4 arquivos modificados, 21 inserções, 9 remoções**, todos exclusivamente em comentários, docstrings e remoção de casts estáticos. Zero mudança de comportamento, zero alteração de contrato, zero regressão.

---

## 2. GAP-01 — Docstring do `MLPipelineOrchestrator` (`src/pipeline_manager.py`)

### 2.1 Diagnóstico

A docstring da classe descrevia um fluxo que incluía "Monitoramento → Reporting", estágios que `run_pipeline()` nunca executa. O Reporting (`generate_data_drift_report`) é disparado exclusivamente pelo scheduler (`schedule_pipeline`) ou via `make drift`, pois depende de logs de inferência acumulados em produção.

### 2.2 Correção

```python
# Antes (src/pipeline_manager.py:30):
class MLPipelineOrchestrator:
    """Orquestrador de Pipeline de ML (Simulação de DAG).

    Fluxo: Ingestão → Pré-processamento → Treinamento → Monitoramento → Reporting.
    """

# Depois:
class MLPipelineOrchestrator:
    """Orquestrador de Pipeline de ML (Simulação de DAG).

    `run_pipeline` executa o DAG de treino: Ingestão → Pré-processamento →
    Treinamento. O Reporting (Data Drift) NÃO faz parte de `run_pipeline`: é
    disparado pelo scheduler (`schedule_pipeline`) ou sob demanda via
    `run_reporting` (`make drift`), pois depende de logs de inferência
    acumulados em produção.
    """
```

### 2.3 Impacto

Nenhum — pura correção documental. `run_pipeline()`, `run_reporting()` e o scheduler permanecem inalterados. Testes existentes não foram modificados.

---

## 3. GAP-02 — Nomenclatura "Dask + SGD" no Modo Big Data (`src/config.py`, `src/train.py`)

### 3.1 Diagnóstico

A função `use_dask_mode` (nome, docstring) e o log "Modo Big Data detectado — usando treinamento incremental (SGD)" sugeriam que Dask era usado no treino. Na realidade:

- **Dask** atua na camada de dados: `data_ingestion.py:74` e `preprocessing.py:68` usam Dask real quando `use_dask_mode()` retorna `True`.
- **Treino incremental** (`_train_incremental`): usa `pandas.read_csv(chunksize=CHUNK)` + `SGDClassifier.partial_fit` — **sem Dask**.

Adicionalmente, o Pyrefly reportava `redundant-cast` em dois métodos de `IncrementalDiabetesModel`, pois os stubs do scikit-learn já anotam `predict`/`predict_proba` com retorno `np.ndarray` — o `cast` era supérfluo.

### 3.2 Correção — `src/config.py`

```python
# Antes:
def use_dask_mode(file_path: str | None = None) -> bool:
    """
    Retorna True se o processamento deve usar Dask em vez de Pandas.

    Ativa o modo Dask quando:
    - A variável de ambiente USE_DASK="true" estiver definida, OU
    - O arquivo informado tiver tamanho superior a 500 MB.
    """

# Depois:
def use_dask_mode(file_path: str | None = None) -> bool:
    """Retorna True quando o caminho "Big Data" deve ser ativado.

    O nome refere-se à camada de dados: ingestão e pré-processamento usam Dask
    de fato quando este chaveamento retorna True. No treino, porém, o mesmo
    chaveamento seleciona o fluxo incremental (`_train_incremental` em
    `train.py`), que usa `pandas.read_csv(chunksize=...)` + `SGDClassifier.
    partial_fit` — não Dask. Em resumo: Dask processa os dados; o treino é por
    chunks pandas + SGD incremental.

    Ativa o modo quando:
    - A variável de ambiente USE_DASK="true" estiver definida, OU
    - O arquivo informado tiver tamanho superior a 500 MB.
    """
```

### 3.3 Correção — `src/train.py` (comentário de chaveamento)

```python
# Antes (src/train.py:555):
    if use_dask_mode(str(data_p)):
        logger.info("Modo Big Data detectado — usando treinamento incremental (SGD).")
        _train_incremental(data_p)

# Depois:
    if use_dask_mode(str(data_p)):
        # Big Data no treino = chunks pandas + SGDClassifier.partial_fit, NÃO Dask.
        # (Dask atua só na camada de dados: ingestão/pré-processamento.)
        logger.info("Modo Big Data detectado — usando treinamento incremental (SGD).")
        _train_incremental(data_p)
```

### 3.4 Correção — `src/train.py` (casts redundantes em `IncrementalDiabetesModel`)

```python
# Antes (src/train.py:261–265):
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return cast(np.ndarray, self.classifier.predict(self.imputer.transform(X)))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return cast(
            np.ndarray, self.classifier.predict_proba(self.imputer.transform(X))
        )

# Depois:
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classifier.predict(self.imputer.transform(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.classifier.predict_proba(self.imputer.transform(X))
```

O `cast` na linha 368 (`pipeline.predict` → `ndarray | tuple`) foi preservado: a anotação do sklearn Pipeline declara tipo mais amplo e o estreitamento é justificado.

### 3.5 Impacto

Nenhum em runtime — `cast` é transparente. Diagnóstico Pyrefly `redundant-cast` eliminado. `test_config.py` e testes de treino inalterados (comportamento = idêntico).

---

## 4. GAP-03 — Âncora de `MODEL_URI` no `k8s/deployment.yaml`

### 4.1 Diagnóstico

O roteiro instrui mostrar `MODEL_URI` no `deployment.yaml`, mas o valor reside em `k8s/configmap.yaml:28` e chega ao pod via `envFrom: configMapRef: santander-ml-config` — arquitetura correta que evita fonte dupla de verdade. O deployment não continha nenhuma referência textual ao `MODEL_URI`, tornando a relação opaca para quem lê apenas o manifesto de deployment.

### 4.2 Correção

```yaml
# Antes (k8s/deployment.yaml:96):
          # Bulk inject: todas as chaves do ConfigMap viram env vars
          envFrom:
            - configMapRef:
                name: santander-ml-config

# Depois:
          # Bulk inject: todas as chaves do ConfigMap viram env vars.
          # MODEL_URI (models:/PimaDiabetesClassifier@champion) chega por aqui —
          # definido em k8s/configmap.yaml, não duplicado neste manifesto.
          envFrom:
            - configMapRef:
                name: santander-ml-config
```

### 4.3 Impacto

Nenhum — comentário YAML puro. Manifesto permanece válido e sem duplicação de valor.

---

## 5. GAP-04 — Contrato de Resposta `/predict` (sem edição de código)

### 5.1 Diagnóstico

O roteiro cita resposta com campos `prediction` e `probability`. O contrato real (`PredictionResponse` em `src/api.py:121`) usa `predicao`, `confianca`, `modelo_versao`, `latencia_s`. Os nomes `prediction`/`probability` existem apenas no CSV de log (`LOG_FIELDNAMES` em `src/api.py:225`).

### 5.2 Decisão

**(A) Manter o código — corrigir o roteiro.** Zero risco para consumidores da API. O contrato `PredictionResponse` e os testes de `src/api.py` permanecem intactos.

Esta correção compete à documentação externa ao repositório (roteiro/PPTX) — ver Seção 8.

---

## 6. Validações Realizadas

| Verificação | Ferramenta / Comando | Resultado |
|---|---|---|
| Suíte completa de testes | `pytest -q --cov=src --cov-fail-under=85` | **PASS** — 128 aprovados, 0 falhas |
| Gate de cobertura | `--cov-fail-under=85` | **PASS** — cobertura 94,53% |
| Testes unitários de treino | `pytest tests/unit/test_train_helpers.py` | **PASS** — 20 aprovados |
| Regressão em pipeline_manager | `pytest tests/unit/test_pipeline_manager.py tests/integration/test_pipeline_manager.py` | **PASS** |
| Inspeção de segurança | Revisão manual das mudanças | **PASS** — ver Seção 7 |

---

## 7. Revisão de Segurança

Todas as alterações são comentários, docstrings e remoção de casts estáticos. Nenhuma lógica de execução foi modificada. Itens auditados:

| Item | Status |
|---|---|
| `MODEL_URI` no comentário do deployment | Alias de modelo já público no ConfigMap — não é segredo |
| Validação de input em `/predict` (`PatientData`, `strict=True`, `extra="forbid"`) | Inalterada |
| Fail-secure do `ADMIN_RELOAD_TOKEN` | Inalterado |
| Escrita em `reports/` e `data/logs/` (path traversal) | Caminhos de escrita inalterados |
| `ADMIN_RELOAD_TOKEN` em logs | Inalterado — token nunca logado |

**Conclusão:** zero surface de ataque alterada. Nenhuma informação sensível exposta.

---

## 8. Riscos Residuais e Próximos Passos

### No repositório

Nenhum risco residual introduzido por esta sessão.

### Na documentação externa (roteiro/PPTX — fora deste repositório)

> **Atualização 2026-06-13:** todos os itens abaixo foram concluídos em sessão separada — ver Seção 12 para detalhes.

1. ✅ **GAP-04 — campos `/predict`:** substituir `prediction`/`probability` por `predicao`/`confianca`/`modelo_versao`/`latencia_s` nos slides e no roteiro que demonstram a resposta do endpoint.
2. ✅ **GAP-03 — localização de `MODEL_URI`:** ao mostrar a variável ao vivo, abrir `k8s/configmap.yaml` (linha 28) e explicar que ela chega ao pod via `envFrom: configMapRef`. Não apontar o `deployment.yaml` como origem do valor.
3. ✅ **GAP-02 — descrição "Dask + SGDClassifier":** corrigir nos slides para "Dask na camada de dados (ingestão/pré-processamento) | treino incremental por chunks pandas + `SGDClassifier.partial_fit`".

---

## 9. Arquivos Alterados (diff resumido)

```
k8s/deployment.yaml     |  4 +++-   (comentário-âncora MODEL_URI)
src/config.py           | 12 +++++++++---   (docstring use_dask_mode)
src/pipeline_manager.py |  6 +++++-   (docstring MLPipelineOrchestrator)
src/train.py            |  8 ++++----   (comentário switch + remoção cast redundante)
4 files changed, 21 insertions(+), 9 deletions(-)
```

---

## 10. Histórico de Commits desta Sessão

| Hash | Tipo | Escopo | Descrição |
|---|---|---|---|
| `d291879` | `docs` | `audit` | Fix doc×code gaps GAP-01–03; remove redundant ndarray casts; add status_report5 |

---

## 11. Impacto e Riscos

| Dimensão | Avaliação |
|---|---|
| **Compatibilidade de API** | Zero — `PredictionResponse` inalterado |
| **Comportamento de treino** | Zero — lógica de `_train_incremental`, `use_dask_mode`, `IncrementalDiabetesModel` inalterada |
| **Manifesto Kubernetes** | Zero — YAML válido; `envFrom` inalterado |
| **Cobertura de testes** | 94,53% (gate 85% ✓) — nenhum teste adicionado ou removido |
| **Risco de regressão** | Nulo — 128 testes aprovados; zero falha |

---

## 12. Atualização — Itens de Documentação Concluídos (2026-06-13)

Os três itens listados na Seção 8 foram aplicados em sessão separada nos arquivos `santander_ml_case_v1_0_3.pptx` e `roteiro_talk_and_show_santander_ml_case_v1_0_3.docx` (externos a este repositório). Detalhes abaixo.

### 12.1 GAP-02 — PPTX, Slide 4 "Treinamento do Modelo"

Bullet "Modo Big Data" reescrita:

```
Antes:
  Modo Big Data: Dask + SGDClassifier com partial_fit para arquivos > 500 MB.

Depois:
  Modo Big Data (> 500 MB): Dask na ingestão/pré-processamento;
  treino incremental via chunks pandas + SGDClassifier.partial_fit.
```

### 12.2 GAP-02 — Roteiro, Slide 4

Adições na seção "Talk and show — o que mostrar junto com o slide":

- Nova bullet orientando esclarecer que `use_dask_mode` (`src/config.py`, limiar 500 MB) é utilizado em `src/data_ingestion.py` e `src/preprocessing.py` (Dask = camada de dados), enquanto `_train_incremental` (`src/train.py`) não usa Dask — processa em chunks via `pandas.read_csv(chunksize=...)` + `SGDClassifier.partial_fit`.
- Novo trecho no bloco "Comando ou trecho para mostrar":

```bash
grep -n "use_dask_mode|_train_incremental|partial_fit" src/config.py src/train.py
```

### 12.3 GAP-03 — Roteiro, Slide 9 "Deploy Cloud-Native"

Bullet de abertura do "Talk and show" alterada:

```
Antes:
  Abrir k8s/deployment.yaml e destacar MODEL_URI.

Depois:
  1. Abrir k8s/configmap.yaml → MODEL_URI: models:/PimaDiabetesClassifier@champion.
  2. Abrir k8s/deployment.yaml → envFrom: configMapRef: name: santander-ml-config.
  Deixar explícito que o valor não é duplicado no manifesto de deployment.
```

Bloco "Comando ou trecho para mostrar" atualizado para incluir `code k8s/configmap.yaml` antes de `code k8s/deployment.yaml`. Apêndice A (sequência de janelas/abas do VS Code) atualizado para incluir `k8s/configmap.yaml`.

### 12.4 GAP-04 — Roteiro, Slides 6 e 11

**Slide 6, "Evidência de sucesso para verbalizar":**

```
Antes:
  /predict deve retornar classe prevista, probabilidade e versão do modelo.

Depois:
  /predict deve retornar JSON com `predicao`, `confianca`, `modelo_versao`
  e `latencia_s`.
```

**Slide 11, "Talk and show":**

```
Antes:
  destaque `prediction`, `probability` e `modelo_versao`

Depois:
  destaque `predicao`, `confianca`, `modelo_versao` e `latencia_s`
```

Os nomes agora correspondem ao contrato real de `PredictionResponse` em `src/api.py:121`.

### 12.5 Validação nos Documentos

Edição realizada via unpack/edit XML/pack (python-docx / python-pptx). QA visual por render para imagem confirmou ausência de overflow, corte ou sobreposição nos trechos alterados em ambos os documentos.
