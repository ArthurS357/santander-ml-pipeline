# Model Card — PimaDiabetesClassifier

> Documento de governança do modelo, inspirado no padrão *Model Cards for Model Reporting* (Mitchell et al., 2019).

## Visão Geral

| Campo | Valor |
|---|---|
| **Nome no Registry** | `PimaDiabetesClassifier` |
| **Alias de produção** | `@champion` |
| **Tarefa** | Classificação binária (diabetes: positivo/negativo) |
| **Domínio** | Saúde pública / triagem clínica |
| **Framework** | scikit-learn (servido via FastAPI) |
| **Versionamento** | MLflow Tracking + Model Registry |

## Finalidade (Intended Use)

- **Uso pretendido:** triagem auxiliar / educacional — estimar risco de diabetes a partir de 8 features clínicas do dataset *Pima Indians Diabetes*.
- **Usuários:** equipe de dados/ML em contexto de PoC e avaliação acadêmica (Academia Santander).
- **Fora de escopo:** **não** é dispositivo médico nem substitui diagnóstico clínico. Não deve ser usado para decisão isolada sobre pacientes reais.

## Algoritmos Avaliados

Treinados e comparados a cada execução (`src/train.py`, modo padrão):

| Algoritmo | Hiperparâmetros principais |
|---|---|
| Random Forest | `n_estimators=100`, `max_depth=5`, `random_state=42` |
| Logistic Regression | `max_iter=1000`, `random_state=42` |
| SVM | `SVC(probability=True)`, `random_state=42` |

**Modo Big Data** (arquivos > 500 MB ou `USE_DASK=true`): `SGDClassifier(loss="log_loss")` com `partial_fit` incremental, empacotado em `IncrementalDiabetesModel` (imputer + classificador).

Todos usam `Pipeline` com `SimpleImputer(strategy="median")` — a imputação é ajustada **apenas no treino** (sem data leakage).

## Métrica de Seleção

O "melhor modelo" é escolhido por métrica configurável via `MODEL_SELECTION_METRIC` (default `f1_score`). Suportadas: `accuracy`, `balanced_accuracy`, `f1_score`, `precision`, `recall`, `roc_auc`. Em saúde, `recall` é frequentemente priorizado (falso negativo é caro).

Métricas registradas no MLflow por run: accuracy, balanced_accuracy, precision, recall, f1_score, roc_auc. Valores de referência (holdout 20% estratificado, dataset público): accuracy ≈ 0.77, f1 ≈ 0.67 (variam por execução/seed).

## Entrada / Saída (Signature)

Toda versão registrada carrega `signature` + `input_example` (`mlflow.models.infer_signature`). Entrada: 8 floats (`preg, plas, pres, skin, test, mass, pedi, age`). Saída: classe (0/1). A API expõe `predicao`, `confianca`, `modelo_versao`, `latencia_s`.

## Limitações e Riscos

- **Dataset pequeno e específico** (768 mulheres Pima ≥ 21 anos) — não generaliza para outras populações.
- **Possível desbalanceamento** de classes — mitigado com `stratify=y` no split.
- **Zeros clínicos** tratados como ausentes (ver [Data Card](data_card.md)) — imputação por mediana pode atenuar sinais.
- **Sem calibração de probabilidade** formal — `confianca` é a probabilidade bruta do modelo.

## Uso Responsável

- Monitoramento de drift via PSI (`src/generate_report.py`) e métricas Prometheus (`diabetes_predictions_total`, `diabetes_prediction_confidence`).
- Rastreabilidade ponta-a-ponta: cada modelo vincula-se ao `dataset_sha256` do arquivo de treino (ver [Data Card](data_card.md)).
- Promoção controlada por alias `@champion` (registrar ≠ aprovar para produção).

## Manutenção

- **Retreino:** orquestrado por `src/pipeline_manager.py` (agendável).
- **Reload sem downtime:** `POST /admin/reload_model` (protegido por token).
- **Decisões de arquitetura:** ver [ADRs](adr/).
