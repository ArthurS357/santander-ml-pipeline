# ADR 0005 — SQLite na PoC, PostgreSQL em produção

## Status

Aceito.

## Contexto

É preciso persistir metadados de treino (`training_records`) e snapshots de
dataset (`training_dataset_snapshots`). A PoC deve ser reproduzível sem
provisionar infraestrutura, mas a produção precisa de concorrência.

## Decisão

Usar **SQLite** na PoC e **PostgreSQL** em produção, abstraídos por
**SQLAlchemy**.

- Conexão controlada por `DATABASE_URL` (env var).
- PoC: `sqlite:///./training_history.db` (zero infraestrutura).
- Produção: `postgresql://user:pass@host/db` — troca de uma variável, sem
  alterar código (o dialeto é abstraído pelo SQLAlchemy).
- Driver `psycopg2-binary` já em `requirements.txt`.

## Consequências

- **Positivas:** reprodutibilidade local imediata; caminho de migração trivial
  para produção; sem lock-in de dialeto.
- **Negativas / trade-offs:** SQLite não suporta múltiplos escritores
  concorrentes — aceitável no pipeline sequencial da PoC, inadequado para
  produção (daí o PostgreSQL).
