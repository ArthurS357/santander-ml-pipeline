# ADR 0006 — DevSecOps Pipeline Hardening

## Status

Aceito.

## Contexto

O pipeline de CI/CD falhou em múltiplos gates de segurança após o incremento
de dependências e a adição dos manifestos Kubernetes. Os gates impactados
foram:

| Gate | Ferramenta | Falha observada |
|---|---|---|
| Vulnerabilidades de dependências | `pip-audit` | 8 pacotes com CVEs publicados: `cryptography`, `idna`, `urllib3`, `GitPython`, `Mako`, `mlflow`, `starlette`, `pytest` |
| Segredo embarcado na imagem | `trivy` / `checkov CKV_SECRET` | `ENV ADMIN_RELOAD_TOKEN=` presente no `Dockerfile` |
| Supressão de checks Kubernetes | `checkov` (runner K8s) | Comentários `# checkov:skip=` ignorados pelo runner; apenas annotations funcionam |
| Padrão basic-auth em placeholder | `checkov CKV_SECRET_4` | Regex `://USER:PASSWORD@` em `stringData` do Secret detectado como credencial real |
| Testes de integração falhando | `pytest` (gate de cobertura) | `mlflow>=3.11` coloca o file store em "maintenance mode", quebrando os fixtures que usam `file://` efêmero para isolamento; cobertura caiu para 79.58% < 85% |
| Conflito de resolução de dependências | `pip` | `prometheus-fastapi-instrumentator==7.1.0` restringe `starlette<1.0.0`, conflitando com o pin de CVE `starlette>=1.0.1` |

## Decisão

### 1. Atualização de dependências vulneráveis (`requirements.txt`)

Todos os pins com CVE publicado foram migrados de versão exata (`==`) para
limite mínimo seguro (`>=`), alinhando o resolver com as versões corrigidas:

| Pacote | Antes | Depois |
|---|---|---|
| `cryptography` | `==46.0.6` | `>=46.0.7` |
| `idna` | `==3.11` | `>=3.15` |
| `urllib3` | `==2.6.3` | `>=2.7.0` |
| `GitPython` | `==3.1.46` | `>=3.1.50` |
| `Mako` | `==1.3.10` | `>=1.3.12` |
| `mlflow` / `mlflow-skinny` / `mlflow-tracing` | `==3.10.1` | `>=3.11.1` (alinhados) |
| `starlette` | `==0.52.1` | `>=1.0.1` |
| `pytest` | `==9.0.2` | `>=9.0.3` |
| `prometheus-fastapi-instrumentator` | `==7.1.0` | `>=8.0.0` (desbloqueia `starlette>=1.0.0`) |

### 2. Remoção de segredo embarcado no Dockerfile

A linha `ENV ADMIN_RELOAD_TOKEN=""` foi removida da imagem. O token é injetado
exclusivamente em tempo de execução via Kubernetes Secret
(`santander-ml-secrets`). A ausência da variável aciona o comportamento
`fail-secure` implementado na API: o endpoint `/admin/reload_model` nega acesso
por padrão quando o token não está presente.

### 3. Supressão de checks Kubernetes via annotations

O runner Kubernetes do Checkov ignora comentários inline (`# checkov:skip=`).
A supressão correta é declarada como annotation no bloco `metadata` do recurso:

```yaml
metadata:
  annotations:
    checkov.io/skip1: 'CKV_K8S_43=PoC: using tag instead of digest'
    checkov.io/skip2: 'CKV_K8S_35=PoC: secrets as env vars is acceptable for this scope'
    checkov.io/skip3: 'CKV_K8S_14=PoC: acceptable to not fix tag rigorously'
```

As supressões cobrem três checks aceitos como trade-offs de PoC:
- **CKV_K8S_43** — imagem referenciada por tag (`:latest`) em vez de digest imutável.
- **CKV_K8S_35** — segredos injetados como variáveis de ambiente em vez de montagem de arquivo.
- **CKV_K8S_14** — tag não fixada em SHA imutável.

### 4. Neutralização do regex basic-auth nos placeholders

O placeholder `DATABASE_URL: "postgresql://USER:PASSWORD@HOST:5432/DBNAME"` no
manifesto `configmap-secret.yaml` acionava o detector de padrão basic-auth
(`CKV_SECRET_4`). A solução adota chaves de template (`{USER}:{PASSWORD}`) em
vez de dois-pontos literais, que ficam fora do padrão `://[^{}\s]+:[^{}\s]+@`
reconhecido pelo detect-secrets. O conteúdo documental do placeholder é
preservado.

### 5. Opt-in do file store do MLflow nos testes de integração

O MLflow 3.11 introduziu um "maintenance mode" para o backend de arquivo
(`./mlruns`), lançando `MlflowException` por padrão quando detecta URI
`file://`. Os fixtures de integração (`isolated_train_module`,
`isolated_orchestrator`) utilizam `file://` efêmero em `tmp_path` como
mecanismo de isolamento de testes — comportamento correto e intencional.

A solução é um fixture `autouse` em `tests/conftest.py` que exporta
`MLFLOW_ALLOW_FILE_STORE=true` (opt-out oficial documentado pelo MLflow) para
todos os testes. O escopo é restrito ao processo de teste via `monkeypatch.setenv`,
sem afetar nenhum outro componente.

A produção utiliza tracking remoto (`http://mlflow-service:5000`) e não é
afetada por esta mudança.

### 6. Namespace explícito em todos os manifestos Kubernetes

`namespace: santander-ml` adicionado ao bloco `metadata` de todos os cinco
manifestos (`deployment.yaml`, `configmap-secret.yaml`, `service.yaml`,
`hpa.yaml`, `networkpolicy.yaml`) e um manifesto `namespace.yaml`
(`kind: Namespace`) criado para declarar o namespace explicitamente, evitando
falha de `kubectl apply` em clusters sem o namespace pré-criado.

## Consequências

### Positivas

- **CI verde:** todos os gates (`pip-audit`, `checkov`, `bandit`, `pytest`) são
  satisfeitos localmente no Python 3.14.4 — ambiente idêntico ao runner.
- **Superfície de ataque reduzida:** 8 CVEs resolvidos nos pacotes Python;
  segredo removido da camada de imagem Docker.
- **Cobertura de testes restaurada:** 94.44% (ante 79.58% pós-falha), acima do
  gate mínimo de 85%.
- **Isolamento de testes preservado:** os fixtures de integração continuam
  usando backends efêmeros (`file://`, `sqlite://`) sem interferência entre
  execuções.
- **Infraestrutura Kubernetes namespaced:** todos os recursos são criados no
  namespace dedicado `santander-ml`, eliminando ambiguidade em clusters
  compartilhados.

### Negativas / trade-offs

- A migração de pins `==` para `>=` delega ao resolver pip a escolha da versão
  exata em cada ambiente de build. Em produção, recomenda-se gerar um
  `requirements.lock` (via `pip-compile`) para builds reproduzíveis.
- O opt-in `MLFLOW_ALLOW_FILE_STORE=true` é uma concessão ao comportamento
  legado do MLflow. O roadmap técnico inclui migrar os fixtures de teste para
  `sqlite://` efêmero, eliminando a dependência do file store por completo.
- Os três checks do Checkov suprimidos via annotation (`CKV_K8S_43/35/14`)
  representam débitos técnicos documentados, aceitáveis no escopo de PoC e
  endereçáveis na transição para produção.
