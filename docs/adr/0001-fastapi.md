# ADR 0001 — FastAPI como framework da API de inferência

## Status

Aceito.

## Contexto

A solução precisa expor o modelo via API REST de baixa latência, com validação
de entrada robusta, documentação automática (para a banca) e suporte a
observabilidade. Alternativas consideradas: Flask e Django REST Framework.

## Decisão

Adotar **FastAPI** (servido por Uvicorn).

Motivos:

- Validação automática de payload via **Pydantic v2** (`PatientData` com faixas
  de domínio, `extra="forbid"`, `strict=True`).
- **OpenAPI/Swagger** embutido em `/docs` — demonstração imediata.
- Suporte nativo a `async` e a `BackgroundTasks` (usado no inference logging).
- Ecossistema de observabilidade (`prometheus-fastapi-instrumentator`) e rate
  limiting (`slowapi`).

## Consequências

- **Positivas:** menos código de validação manual; contrato de API
  autodocumentado; alta performance.
- **Negativas / trade-offs:** acopla a stack a Starlette/Uvicorn; `strict=True`
  exige que clientes enviem floats explícitos (ex.: `1.0`, não `1`).
