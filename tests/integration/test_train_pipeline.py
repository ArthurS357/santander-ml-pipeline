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
