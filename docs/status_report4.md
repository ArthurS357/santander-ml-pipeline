# Relatório de Release — Santander ML Pipeline V1.0.3

**Data de referência:** 2026-06-12
**Responsável:** Pipeline DevSecOps — Academia Santander
**Versão:** `V1.0.3`
**Branch:** `main` | **Último Commit:** `4494db1`

---

## 1. Resumo Executivo

O patch **v1.0.3** consolida um conjunto de correções críticas de produção e hardening de infraestrutura identificados durante a revisão técnica pós-release do v1.0.2. As intervenções se distribuem em três eixos:

- **Correção crítica de roteamento no K8s:** o caminho de carregamento de modelo via `MODEL_URI=models:/PimaDiabetesClassifier@champion` — utilizado pelo ConfigMap de produção — provocava `IndexError` não capturado nos endpoints `/`, `/health/ready` e `/predict`. A função `_get_model_version_id` foi reescrita para tratar explicitamente os esquemas URI do MLflow Registry antes do fallback de fatiamento de caminho.
- **Tipagem segura e conformidade DevSecOps:** o registro do handler de rate limiting expunha uma incompatibilidade de tipos (`bad-argument-type`) entre a interface do `slowapi` e o `Starlette`, detectada pelos type checkers Pyrefly e pyright. A correção utiliza um adaptador com `typing.cast`, eliminando o alerta sem `# type: ignore` e sem `assert` (que acionaria a violação de segurança B101 no Bandit do CI).
- **Governança de infraestrutura e scripts:** o manifesto `configmap-secret.yaml` foi segregado, os scripts de demonstração receberam bootstrap automático de dataset, e o gate de cobertura de testes foi padronizado globalmente em 85%.

A versão mantém **retrocompatibilidade total** com a interface pública da API, os contratos MLflow e as validações Pydantic.

---

## 2. Correção Crítica — `_get_model_version_id` para URIs do MLflow Registry (`src/api.py`)

### 2.1 Diagnóstico

A função `_get_model_version_id` assume que `modelo_path` é um caminho de sistema de arquivos e aplica `Path(modelo_path).parts` para extrair o `run_id`. Para um valor no formato `models:/PimaDiabetesClassifier@champion`:

```
Path("models:/PimaDiabetesClassifier@champion").parts
# ('models:', 'PimaDiabetesClassifier@champion')  → len = 2
```

O bloco `except (ValueError, IndexError)` tentava o fallback `parts[-3]`, que em uma tupla com apenas 2 elementos lança um novo `IndexError` **não capturado** — derrubando os endpoints `/`, `/health/ready` e `/predict` exatamente no caminho de produção descrito no ConfigMap do Kubernetes.

### 2.2 Correção

```python
# Antes (bugado em produção com MODEL_URI=models:/...@champion):
def _get_model_version_id() -> str:
    if not modelo_path:
        return "desconhecido"
    parts = Path(modelo_path).parts
    try:
        mlruns_idx = [p.lower() for p in parts].index("mlruns")
        run_id = parts[mlruns_idx + 2]
        return f"run_{run_id}"
    except (ValueError, IndexError):
        return f"run_{Path(modelo_path).parts[-3]}"   # ← IndexError aqui

# Depois (todos os esquemas tratados antes do fallback):
def _get_model_version_id() -> str:
    if not modelo_path:
        return "desconhecido"
    if modelo_path.startswith("models:/"):
        return modelo_path.removeprefix("models:/")   # PimaDiabetesClassifier@champion
    if modelo_path.startswith("runs:/"):
        parts = modelo_path.split("/")
        if len(parts) > 1 and parts[1]:
            return f"run_{parts[1]}"
        return "run_desconhecido"
    parts = Path(modelo_path).parts
    try:
        mlruns_idx = [p.lower() for p in parts].index("mlruns")
        run_id = parts[mlruns_idx + 2]
        return f"run_{run_id}"
    except (ValueError, IndexError):
        if len(parts) >= 3:
            return f"run_{parts[-3]}"
        return "modelo_externo"                        # ← nunca mais IndexError
```

### 2.3 Cobertura de Testes

Adicionados 5 casos parametrizados em `tests/unit/test_api_logic.py`:

| ID | Entrada (`modelo_path`) | Saída esperada |
|---|---|---|
| `registry_alias_champion` | `models:/PimaDiabetesClassifier@champion` | `PimaDiabetesClassifier@champion` |
| `registry_version_pin` | `models:/PimaDiabetesClassifier/1` | `PimaDiabetesClassifier/1` |
| `runs_uri` | `runs:/abc123/model` | `run_abc123` |
| `runs_uri_without_id` | `runs:/` | `run_desconhecido` |
| `short_path_no_indexerror` | `modelo.pkl` | `modelo_externo` |

---

## 3. Tipagem Segura — Handler de Rate Limiting (`src/api.py`)

### 3.1 Diagnóstico

`add_exception_handler` do Starlette é assinado como:

```python
def add_exception_handler(
    exc_class: type[Exception],
    handler: Callable[[Request, Exception], Response]
) -> None: ...
```

O `_rate_limit_exceeded_handler` do slowapi aceita `(Request, RateLimitExceeded)` — um **subtipo** de `Exception`. Por **contravariância de parâmetro de função**, esse handler mais estreito não é atribuível onde se espera um que aceita `Exception`. O Pyrefly (Pylance/pyright-compatible) reportava `bad-argument-type` nessa linha. O comportamento em runtime é correto (o Starlette só roteia o tipo registrado), mas a anotação estática é inválida.

### 3.2 Correção

```python
# Adaptador com assinatura larga (Request, Exception) delegando ao handler oficial:
def _rate_limit_handler(request: Request, exc: Exception) -> Response:
    """Adaptador de tipo entre o slowapi e o Starlette.

    `add_exception_handler` tipa o handler como `(Request, Exception)`, mas
    `_rate_limit_exceeded_handler` exige `RateLimitExceeded` (tipo mais
    estreito) — incompatível por contravariância de parâmetro. Em runtime o
    Starlette só roteia `RateLimitExceeded` até aqui, então o cast é seguro.
    """
    return _rate_limit_exceeded_handler(request, cast(RateLimitExceeded, exc))

app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
```

### 3.3 Por que `cast` e não alternativas?

| Alternativa | Por que descartada |
|---|---|
| `# type: ignore` | Supressão genérica; não documenta a causa; varia de sintaxe entre mypy/pyright/pyrefly |
| `assert isinstance(exc, RateLimitExceeded)` | O Bandit (`-ll -ii`) flagueia `assert` via **B101** (assert used) no Security Gate do CI |
| Sobrescrever o tipo com `cast` | Explícito, portável, sem supressão — caminho correto |

---

## 4. Hardening de Infraestrutura Kubernetes

### 4.1 Segregação do ConfigMap e Secret

O manifesto unificado `k8s/configmap-secret.yaml` era aplicado com um único `kubectl apply`, incluindo involuntariamente o bloco Secret com placeholders (`ADMIN_RELOAD_TOKEN: CHANGE_ME`) — contradizendo a orientação do README de criar o Secret imperativamente fora do Git.

**Ação:** manifesto segregado em dois arquivos com papéis e controles de acesso distintos:

| Arquivo | Conteúdo | Comando de aplicação |
|---|---|---|
| `k8s/configmap.yaml` | Variáveis não sensíveis (ConfigMap) | `kubectl apply -f k8s/configmap.yaml` |
| `k8s/secret.example.yaml` | Placeholder documental (Secret Opaque) | **Não aplicar.** Usar `kubectl create secret` |

### 4.2 Bootstrap de Dataset nos Scripts de Demo

Os scripts `scripts/demo_end_to_end.sh` e `scripts/demo_end_to_end.ps1` falhavam na etapa de pipeline em máquinas limpas porque `data/raw/pima_diabetes.csv` não está versionado no Git. Adicionada uma etapa `0/8` que:

1. Verifica se o arquivo existe localmente.
2. Se ausente, baixa do mirror público UCI (mesma URL usada pelo workflow de CI).
3. Prefixa o cabeçalho de colunas (o arquivo público vem sem nomes).

---

## 5. Governança Documental e Testes

### 5.1 Strict Mode do Pydantic — Retificação Técnica

O comentário do `PatientData` e a tabela de trade-offs do README afirmavam incorretamente que inteiros JSON (ex.: `"plas": 85`) eram rejeitados com 422 em `strict=True`. A verificação empírica com Pydantic v2 demonstrou que:

- **Strings numéricas** (`"plas": "85.0"`) → 422 (sem coerção de tipo).
- **Inteiros JSON** (`"plas": 85`) → aceitos e convertidos para `float` via a **tabela de conversão interna do Pydantic v2**.

A documentação foi corrigida em ambos os locais (docstring e README).

### 5.2 Gate de Cobertura Padronizado em 85%

O CI já exigia `--cov-fail-under=85`, mas o `Makefile`, o `docs/roteiro_apresentacao.md` e o README ainda referenciavam 80%. Todos os pontos foram alinhados para 85%. Cobertura atual: **94.53%**.

### 5.3 Exemplos `docker run` Funcionais

O exemplo principal de `docker run` no README subia o container sem modelo (nenhum `MODEL_URI` ou volume `mlruns/`), produzindo um `503` silencioso — confuso para a banca. Substituídos por dois exemplos completos e funcionais:

- **Dev local:** volume `mlruns/` montado como somente leitura (`-v "$PWD/mlruns:/app/mlruns:ro"`).
- **Produção:** variável `MODEL_URI=models:/PimaDiabetesClassifier@champion` via env var.

---

## 6. Validações Realizadas

| Verificação | Ferramenta / Comando | Resultado |
|---|---|---|
| Formatação de código | `black --check src/api.py` | **PASS** |
| Linting estático (flags CI) | `flake8 src/api.py --max-line-length=120 --ignore=E501,W503,E402` | **PASS** — 0 erros |
| Análise de segurança | `bandit -r src/ -ll -ii` | **PASS** — 0 issues |
| Suíte completa de testes | `pytest --cov=src --cov-fail-under=85` | **PASS** — 128 aprovados, cobertura 94.53% |
| Parse de YAMLs k8s | `python -c "import yaml; yaml.safe_load_all(...)"` | **PASS** — todos válidos |
| Sintaxe dos scripts | `bash -n` (sh) / parser PS1 (PowerShell) | **PASS** |

---

## 7. Histórico de Commits (V1.0.2 → V1.0.3)

| Hash | Tipo | Escopo | Descrição |
|---|---|---|---|
| `b06e667` | `fix` | `api` | Support MLflow Registry URIs in `_get_model_version_id` |
| `b4e7142` | `refactor` | `k8s` | Split `configmap-secret.yaml` into configmap and secret example |
| `ddaed36` | `fix` | `scripts` | Bootstrap dataset automatically in end-to-end demo |
| `01780c0` | `chore` | `quality` | Align local coverage gate with CI at 85% |
| `3857106` | `docs` | `readme` | Fix strict-mode claim, k8s split refs, coverage 85 and docker run examples |
| `4494db1` | `fix` | `api` | Resolve exception handler type mismatch for slowapi |

---

## 8. Impacto e Riscos

| Dimensão | Avaliação |
|---|---|
| **Compatibilidade de API** | Zero — nenhuma assinatura pública alterada |
| **Contratos MLflow** | Zero — signature e input_example intactos |
| **Validação Pydantic** | Zero — `PatientData`, `strict=True` e `extra="forbid"` inalterados |
| **Comportamento do rate limiting** | Zero — adaptador delega ao handler oficial; 429 verificado em teste |
| **Risco de regressão** | Baixíssimo — 128 testes aprovados; bug anterior coberto por 5 casos parametrizados |

---

## 9. Próximos Passos

1. **Apresentação à banca:** demonstrar o fluxo completo com `MODEL_URI=models:/PimaDiabetesClassifier@champion`, validar que `/health/ready` retorna 200 e `/predict` retorna predição com `modelo_versao: "PimaDiabetesClassifier@champion"`.
2. **Validação em ambiente corporativo:** executar `./scripts/demo_end_to_end.sh` numa máquina sem dataset pré-existente para confirmar o bootstrap automático.
3. **Roadmap pós-banca:** avaliar migração de type checker de Pyrefly para pyright com `strict: true` no `pyrightconfig.json`, cobrindo toda a base de código.
