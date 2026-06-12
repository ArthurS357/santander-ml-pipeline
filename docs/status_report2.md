# Relatório de Release — Santander ML Pipeline V1.0.1

**Data de referência:** 2026-06-12
**Responsável:** Pipeline DevSecOps — Academia Santander
**Versão:** `V1.0.1`
**Branch:** `main` | **Último Commit:** `aaf1a9b`

---

## 1. Resumo da Versão 1.0.1

A versão **1.0.1** do Santander ML Pipeline foca na estabilização do ambiente de desenvolvimento, correção de conflitos de dependências secundárias que impactavam ferramentas de análise estática de infraestrutura (como o Checkov) ao usar resolutores modernos (como o `uv`), e na homologação definitiva do empacotamento local.

O código foi validado para rodar perfeitamente tanto em infraestruturas baseadas em contêineres quanto em execuções nativas restritas em ambientes corporativos.

---

## 2. Validações Locais Realizadas

Antes da liberação desta versão, a bateria de validação de ambiente (Clean Install) garantiu que o projeto está pronto para a transferência para a rede corporativa:

| Etapa | Ferramenta / Comando | Resultado |
|---|---|---|
| **Clean Install (Isolamento)** | `python -m venv` + `pip` (ou `uv pip`) | **PASS** - Dependências instaladas a partir do zero sem back-tracking eterno. |
| **Geração de Modelo (Treinamento)** | `python src/pipeline_manager.py` | **PASS** - Modelo do scikit-learn gerado, rastreado e armazenado no MLflow/SQLite local sem erros. |
| **Validação da API (Nativo)** | `uvicorn src.api:app` | **PASS** - API capaz de subir nativamente sem Docker e responder corretamente no endpoint `/predict`. |
| **Validação de Imagem (Empacotamento)** | `docker build -t santander-ml-api .` | **PASS** - Imagem gerada com sucesso contendo o código de inferência e a estrutura do servidor. |

---

## 3. Histórico de Commits (V1.0.0 → V1.0.1)

Apenas ajustes cirúrgicos foram necessários para destravar o Pipeline e evitar conflitos.

| Hash | Mensagem | Escopo / Motivo |
|---|---|---|
| `aaf1a9b` | `fix(deps): adjust subdependencies to resolve checkov conflicts with uv` | Correção na árvore de dependências para suportar instalação rápida via `uv` e resolução de conflitos silenciosos da integração do Checkov. |

---

## 4. Próximos Passos (Ações Pós-Release)

1. **Deploy Corporativo:** Extrair a base de código e mover para a infraestrutura interna do Santander (Windows corporativo).
2. **Setup Ambiente Restrito:** Executar treinamento nativo via `venv` no ambiente sem Docker para recriar os modelos do MLflow.
3. **Observabilidade:** Conectar e inicializar os containers do Grafana/Prometheus (em infraestruturas onde o Docker for permitido) para habilitar as métricas de Data Drift no `/predict`.
