# ADR 0002 — MLflow para tracking e Model Registry

## Status

Aceito.

## Contexto

O case exige versionamento de modelos, métricas e artefatos, além de
rastreabilidade. Alternativas: Weights & Biases, Neptune, DVC.

## Decisão

Adotar **MLflow** (Tracking + Model Registry), auto-hospedado.

- Experimentos, métricas e artefatos versionados por run.
- Registro sob nome único `PimaDiabetesClassifier`.
- **Signature + input_example** em todo modelo (enforcement de schema no serving).
- Alias **`@champion`** separando "registrado" de "aprovado para produção".
- Backend de tracking via arquivo (`mlruns/`) na PoC; SQLite/PostgreSQL em produção.

## Consequências

- **Positivas:** open-source, sem custo; Registry integrado; promoção por alias
  sem redeploy (a API consome `models:/PimaDiabetesClassifier@champion`).
- **Negativas / trade-offs:** o file store (`mlruns/`) está marcado como
  *deprecated* pelo MLflow — migração para backend de banco é um próximo passo.
