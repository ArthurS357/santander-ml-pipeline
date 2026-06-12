# Relatório de Release — Santander ML Pipeline V1.0.2

**Data de referência:** 2026-06-12
**Responsável:** Pipeline DevSecOps — Academia Santander
**Versão:** `V1.0.2`
**Branch:** `main` | **Último Commit:** `97509ef`

---

## 1. Resumo Executivo

O patch **v1.0.2** consolida os refinamentos de qualidade técnica exigidos pelo padrão sênior da banca avaliadora do case de certificação **Academia Santander — Engenharia de Machine Learning**. As alterações, estritamente cirúrgicas e não-funcionais do ponto de vista do negócio, abordam três eixos interdependentes:

- **Portabilidade de infraestrutura:** eliminação de dependência implícita de utilitários externos (`curl`) na imagem slim do contêiner.
- **Corretude de concorrência:** correção de race condition latente na rotina de logging de inferências.
- **Conformidade com o modelo de execução assíncrono do FastAPI:** conversão da rota de predição para o paradigma síncrono correto, garantindo que operações CPU-bound do Scikit-Learn sejam isoladas no threadpool gerenciado pelo framework.

A versão mantém **100% de retrocompatibilidade** com a interface pública da API, os contratos MLflow e as validações Pydantic.

---

## 2. Melhorias Implementadas

### 2.1 Infraestrutura — Healthcheck Nativo via Python (`docker-compose.observability.yml`)

**Problema:** O comando de healthcheck do serviço `ml-api` no Compose de observabilidade utilizava o utilitário `curl`, ausente nas imagens `python:*-slim` adotadas pelo projeto. Em ambientes onde apenas a imagem slim está disponível (ambientes corporativos com pull policy restritivo ou mirrors internos), o healthcheck falhava silenciosamente com "executable not found", impedindo a detecção do estado real do processo.

**Solução adotada:**

```yaml
# Antes
test: ["CMD", "curl", "-f", "http://localhost:8000/health/live"]

# Depois
test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live', timeout=5)"]
```

**Justificativa técnica:**
- `urllib.request` é módulo da biblioteca padrão do Python — zero dependências externas.
- `timeout=5` espelha o parâmetro `timeout: 5s` já definido no bloco `healthcheck`, garantindo comportamento consistente.
- Elimina uma superfície de ataque adicional: a imagem slim não precisa instalar `curl` apenas para o healthcheck.
- Compatível com `python:3.14-slim` e qualquer variante slim das versões anteriores.

---

### 2.2 Otimização de Performance na API — Rota `/predict` como `def` Síncrono (`src/api.py`)

**Problema:** O endpoint `POST /predict` estava declarado como `async def`. O FastAPI executa rotas `async def` diretamente no event loop do `asyncio`. O modelo Scikit-Learn (`sklearn.Pipeline`, `RandomForestClassifier`, etc.) é uma implementação síncrona e CPU-bound: durante a execução de `predict()` e `predict_proba()`, a GIL do CPython é mantida e o event loop fica bloqueado, degradando a latência de todas as demais requisições concorrentes.

**Solução adotada:**

```python
# Antes
async def predict(request: Request, data: PatientData, background_tasks: BackgroundTasks) -> PredictionResponse:

# Depois
def predict(request: Request, data: PatientData, background_tasks: BackgroundTasks) -> PredictionResponse:
```

**Justificativa técnica:**
- O FastAPI encaminha rotas `def` síncronas para um threadpool (executor `asyncio.to_thread` via `Starlette`), liberando o event loop durante a inferência.
- O decorator `@limiter.limit(PREDICT_RATE_LIMIT)` do `slowapi`, os `BackgroundTasks` e a injeção de dependência via `Request` funcionam identicamente em rotas síncronas.
- A mudança não altera o contrato de resposta (`PredictionResponse`), a validação Pydantic (`PatientData`) nem as métricas Prometheus.

---

### 2.3 Correção de Concorrência — Race Condition no Log de Inferências (`src/api.py`)

**Problema:** A verificação `write_header = not log_path.exists() or log_path.stat().st_size == 0` ocorria **antes** do bloco `with _log_lock:`. Em cenários de alta concorrência (múltiplas requisições simultâneas ao `/predict`, onde cada uma dispara `log_prediction` em background), duas ou mais threads podiam:

1. Avaliar `write_header = True` simultaneamente (arquivo ainda vazio).
2. Adquirir o lock sequencialmente.
3. Cada uma escrever o cabeçalho CSV individualmente, resultando em cabeçalhos duplicados que corrompem a leitura posterior do arquivo de logs.

**Solução adotada:**

```python
# Antes
write_header = not log_path.exists() or log_path.stat().st_size == 0
with _log_lock:
    with log_path.open("a", newline="", encoding="utf-8") as f:
        ...

# Depois
with _log_lock:
    # Verificação dentro do lock: evita que duas threads vejam arquivo vazio
    # e ambas escrevam o cabeçalho.
    write_header = not log_path.exists() or log_path.stat().st_size == 0
    with log_path.open("a", newline="", encoding="utf-8") as f:
        ...
```

**Justificativa técnica:**
- A verificação de existência/tamanho e a abertura do arquivo são agora uma operação atômica sob o mesmo lock.
- Garante que apenas a primeira thread que adquire o lock encontra o arquivo vazio e escreve o cabeçalho.
- O arquivo `data/logs/inference_logs.csv` permanece válido como dataset de referência para o relatório de Data Drift (PSI), que espera exatamente um cabeçalho.

---

### 2.4 Governança Documental — Revisão do `README.md`

As seguintes inconsistências históricas foram corrigidas para alinhar a documentação ao código efetivamente implementado:

| Item | Antes | Depois | Arquivo impactado |
|---|---|---|---|
| **Versão do sistema** | `v2.0` | `v1.0.1` / `v1.0.2` | `README.md` (heading + âncora TOC) |
| **Data Drift** | "Evidently" / "desabilitado" | "PSI próprio (Population Stability Index) — relatório JSON/Markdown, sem Evidently" | `README.md` (2 diagramas Mermaid, árvore de projeto, nota Python 3.14, seção offline) |
| **Detecção de formato** | `os.path.splitext()` | `pathlib.Path.suffix` | `README.md` (seção de formatos suportados) |
| **ConfigMap Kubernetes** | "DATABASE_URL base64" | "ConfigMap + Secret Opaque com placeholders via stringData" | `README.md` (árvore do projeto) |
| **Deploy — PoC vs Produção** | Ausente | Nota técnica: "Nesta PoC, o CD publica a imagem e simula o deploy. Em produção corporativa, a etapa final seria Kubernetes nativo via Helm ou Argo CD (GitOps)." | `README.md` (seção 3.10) |

---

## 3. Validações Realizadas

| Verificação | Ferramenta / Comando | Resultado |
|---|---|---|
| Formatação de código | `uv run python -m black --check src/api.py` | **PASS** — sem alterações necessárias |
| Linting estático (flags do CI) | `uv run python -m flake8 src/api.py --max-line-length=120 --ignore=E501,W503,E402` | **PASS** — 0 erros |
| Suíte de testes unitários | `uv run python -m pytest src/test_api.py tests/unit -q` | **PASS** — 116 aprovados, 0 falhas |

---

## 4. Histórico de Commits (V1.0.1 → V1.0.2)

| Hash | Tipo | Escopo | Descrição |
|---|---|---|---|
| `61def39` | `fix` | `docker` | Replace curl healthcheck with pure python urllib for slim image |
| `375aa91` | `fix` | `api` | Prevent header race in inference log and make predict sync |
| `97509ef` | `docs` | `readme` | Align docs with v1.0.1 and PSI drift implementation |

---

## 5. Impacto e Riscos

| Dimensão | Avaliação |
|---|---|
| **Compatibilidade de API** | Zero — nenhuma assinatura pública alterada |
| **Contratos MLflow** | Zero — signature e input_example intactos |
| **Validação Pydantic** | Zero — `PatientData`, `strict=True` e `extra="forbid"` inalterados |
| **Comportamento do rate limiting** | Zero — `@limiter.limit` funciona identicamente em rotas síncronas |
| **Risco de regressão** | Baixo — 116 testes aprovados sem falhas; alterações limitadas a plano de infraestrutura, sincronização e documentação |

---

## 6. Próximos Passos

1. **Validação em ambiente corporativo:** executar `./scripts/demo_end_to_end.sh` (ou `.ps1`) na infraestrutura restrita para verificar que o healthcheck Python funciona sem acesso a `curl`.
2. **Apresentação à banca:** demonstrar ao vivo o endpoint `/predict` (latência, rate limiting) e o relatório de drift (`make drift`) como evidência das seções de observabilidade e monitoramento.
3. **Roadmap pós-banca:** reintroduzir Evidently quando houver release compatível com Python 3.14; avaliar migração para PostgreSQL como backend do MLflow.
