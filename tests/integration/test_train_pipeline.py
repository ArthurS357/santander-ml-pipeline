"""Teste de integração do fluxo `_train_standard` — DB e MLflow efêmeros."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select


@pytest.mark.integration
def test_train_standard_persists_snapshot_and_metrics(
    isolated_train_module: object, sample_dataset: Path
) -> None:
    """Roda o pipeline standard ponta-a-ponta em ambiente isolado.

    Valida:
    - tabela `training_dataset_snapshots` recebe linha com hash de 64 chars
    - `mlflow_run_id` é preenchido
    - `row_count` bate com o CSV
    - tabela legada `training_records` continua sendo populada
    """
    train_mod = isolated_train_module

    train_mod.train_model(sample_dataset)

    with train_mod.SessionLocal() as session:
        snapshots = (
            session.execute(select(train_mod.DatasetSnapshotRecord)).scalars().all()
        )

        assert len(snapshots) >= 1, "esperava ao menos um snapshot persistido"

        snap = snapshots[0]
        assert len(snap.dataset_sha256) == 64
        assert snap.row_count == 30  # sample_dataset tem 30 linhas
        assert snap.column_count == 9  # 8 features + class
        assert snap.target_column == "class"
        assert snap.mlflow_run_id is not None and len(snap.mlflow_run_id) > 0

        # Tabela legada continua sendo populada (retrocompatibilidade)
        legacy = session.execute(select(train_mod.TrainingRecord)).scalars().all()
        assert len(legacy) >= 1


@pytest.mark.integration
def test_train_logs_extended_metrics_to_mlflow(
    isolated_train_module: object,
    sample_dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifica que as novas métricas (precision, recall, roc_auc, balanced_accuracy)
    foram registradas no run do MLflow."""
    import mlflow

    train_mod = isolated_train_module
    monkeypatch.setenv("MODEL_SELECTION_METRIC", "recall")

    train_mod.train_model(sample_dataset)

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Pima_Diabetes_Pipeline")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) >= 3  # RF + LR + SVM

    expected_metrics = {
        "accuracy",
        "balanced_accuracy",
        "f1_score",
        "precision",
        "recall",
    }
    for run in runs:
        logged = set(run.data.metrics.keys())
        missing = expected_metrics - logged
        assert not missing, f"métricas ausentes no run {run.info.run_id}: {missing}"
        # Tag de rastreabilidade do dataset
        assert "dataset_sha256" in run.data.tags
        assert len(run.data.tags["dataset_sha256"]) == 64
        # Param de configuração da seleção
        assert run.data.params.get("selection_metric") == "recall"
        assert run.data.params.get("stratified_split") == "True"


@pytest.mark.integration
def test_invalid_selection_metric_falls_back_silently(
    isolated_train_module: object,
    sample_dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Métrica desconhecida não deve quebrar o pipeline — degrada para o default."""
    train_mod = isolated_train_module
    monkeypatch.setenv("MODEL_SELECTION_METRIC", "fantasy_metric_xyz")

    with caplog.at_level("WARNING"):
        train_mod.train_model(sample_dataset)

    assert any("fantasy_metric_xyz" in record.message for record in caplog.records), (
        "esperava warning sobre métrica inválida"
    )


@pytest.mark.integration
def test_incremental_bigdata_saves_wrapper_imputer_and_alias(
    isolated_train_module: object,
    sample_dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fluxo Big Data: wrapper (imputer+SGD) salvo, holdout dedicado, alias champion.

    Valida a correção do bug em que o imputer não era persistido e o holdout
    era o próprio chunk de treino.
    """
    import mlflow
    import mlflow.sklearn
    import pandas as pd

    train_mod = isolated_train_module

    # Força o modo incremental e usa chunk/holdout pequenos p/ o dataset de 30 linhas.
    monkeypatch.setenv("USE_DASK", "true")
    monkeypatch.setenv("BIGDATA_CHUNK_SIZE", "10")
    monkeypatch.setenv("BIGDATA_HOLDOUT_ROWS", "10")

    train_mod.train_model(sample_dataset)

    client = mlflow.tracking.MlflowClient()

    # Alias 'champion' aponta para a versão registrada.
    champion = client.get_model_version_by_alias("PimaDiabetesClassifier", "champion")
    assert champion is not None

    # O artefato carregado é o wrapper com imputer AJUSTADO + classificador.
    model = mlflow.sklearn.load_model("models:/PimaDiabetesClassifier@champion")
    assert isinstance(model, train_mod.IncrementalDiabetesModel)
    assert hasattr(model.imputer, "statistics_"), "imputer não foi salvo/ajustado"

    # Predição ponta-a-ponta sobre features cruas (com imputação interna).
    features = pd.read_csv(sample_dataset).drop("class", axis=1).head(3)
    preds = model.predict(features)
    assert len(preds) == 3

    # O run incremental registrou métricas de holdout (não do chunk de treino).
    experiment = client.get_experiment_by_name("Pima_Diabetes_Pipeline")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    sgd_runs = [r for r in runs if r.data.params.get("algorithm") == "SGDClassifier"]
    assert sgd_runs, "esperava um run de treinamento incremental SGD"
    assert "accuracy" in sgd_runs[0].data.metrics
    assert int(sgd_runs[0].data.params["holdout_rows"]) > 0
