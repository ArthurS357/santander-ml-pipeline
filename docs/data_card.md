# Data Card — Pima Indians Diabetes

> Documentação do dataset usado no treino, sua qualidade, tratamento e rastreabilidade.

## Origem

| Campo | Valor |
|---|---|
| **Dataset** | Pima Indians Diabetes Database |
| **Fonte original** | National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) |
| **Mirror usado no CI** | `raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv` |
| **População** | Mulheres de herança Pima, ≥ 21 anos |
| **Volume** | 768 registros, 8 features + 1 alvo |
| **Formatos suportados** | CSV (padrão), Excel (`.xlsx/.xls`), Parquet — detecção automática em `data_ingestion.py` |

## Schema

| Coluna | Tipo | Descrição | Faixa válida (API) |
|---|---|---|---|
| `preg` | float | Nº de gestações | 0–20 |
| `plas` | float | Glicose plasmática (mg/dL) | >0–250 |
| `pres` | float | Pressão diastólica (mm Hg) | >0–150 |
| `skin` | float | Dobra cutânea tríceps (mm) | 0–100 |
| `test` | float | Insulina sérica 2h (mu U/ml) | 0–1000 |
| `mass` | float | IMC (kg/m²) | >0–100 |
| `pedi` | float | Diabetes pedigree function | 0–3 |
| `age` | float | Idade (anos) | 0–120 |
| `class` | int | Alvo: 1 = diabetes, 0 = não | — |

As faixas são validadas na API via Pydantic (`PatientData`, `extra="forbid"`, `strict=True`).

## Qualidade dos Dados

**Zeros clínicos inválidos:** em `plas`, `pres`, `skin`, `test`, `mass`, o valor `0` é fisiologicamente impossível e representa **dado ausente**, não medição real.

**Tratamento (`src/preprocessing.py`):**

1. Esses zeros são convertidos para `NaN` (marcação de ausência).
2. A imputação por **mediana** é delegada ao `Pipeline` de treino (`SimpleImputer`), ajustada **somente no conjunto de treino** — evitando *data leakage*.
3. `preg`, `age`, `class` **não** são tratados (zeros são válidos).

## Rastreabilidade (Data Lineage)

A cada execução de treino, `src/train.py` calcula e persiste um **snapshot lógico** do dataset:

- `dataset_sha256` — hash SHA-256 do arquivo físico (calculado em chunks de 1 MiB).
- `row_count`, `column_count`, `target_column`.
- `schema_json` — mapa coluna → dtype.
- `mlflow_run_id` — vincula o snapshot ao run específico.

Gravado na tabela `training_dataset_snapshots` (SQLite/PostgreSQL) **e** como tag `dataset_sha256` no MLflow. Isso permite auditar qualquer predição de produção até a versão exata do arquivo de treino que gerou o modelo.

## Monitoramento de Drift

A distribuição de treino (referência) é comparada com os logs de inferência (`data/logs/inference_logs.csv`) via **PSI** em `src/generate_report.py`. Ver [README §3.5.3](../README.md).

## Privacidade

O dataset é público e anonimizado (sem PII). Os logs de inferência contêm apenas as 8 features numéricas + predição — nenhum identificador pessoal.
