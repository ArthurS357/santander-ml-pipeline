"""Testes unitários dos helpers introduzidos em src/train.py (P1)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from src.train import (
    DEFAULT_SELECTION_METRIC,
    SUPPORTED_SELECTION_METRICS,
    DatasetSnapshot,
    _calculate_sha256,
    _collect_dataset_snapshot,
    _compute_metrics,
    _resolve_selection_metric,
)


@pytest.mark.unit
class TestCalculateSha256:
    def test_hash_matches_hashlib_reference(self, tmp_path: Path) -> None:
        payload = b"pima,diabetes,test\n" * 1000
        target = tmp_path / "sample.csv"
        target.write_bytes(payload)

        expected = hashlib.sha256(payload).hexdigest()
        assert _calculate_sha256(target) == expected

    def test_hash_is_deterministic_across_chunk_sizes(self, tmp_path: Path) -> None:
        target = tmp_path / "big.bin"
        target.write_bytes(b"x" * (3 * 1024 * 1024 + 17))  # 3 MiB + odd tail

        small = _calculate_sha256(target, chunk_size=1024)
        big = _calculate_sha256(target, chunk_size=1024 * 1024)
        assert small == big

    def test_empty_file_hashes_to_known_constant(self, tmp_path: Path) -> None:
        target = tmp_path / "empty.bin"
        target.touch()
        # SHA-256 do arquivo vazio
        assert _calculate_sha256(target) == (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )


@pytest.mark.unit
class TestCollectDatasetSnapshot:
    def test_snapshot_fields(self, sample_dataset: Path) -> None:
        snap = _collect_dataset_snapshot(sample_dataset, target_column="class")

        df = pd.read_csv(sample_dataset)
        assert isinstance(snap, DatasetSnapshot)
        assert snap.dataset_path == str(sample_dataset)
        assert snap.row_count == df.shape[0]
        assert snap.column_count == df.shape[1]
        assert snap.target_column == "class"
        assert "class" in snap.schema
        assert len(snap.dataset_sha256) == 64

    def test_snapshot_is_immutable(self, sample_dataset: Path) -> None:
        snap = _collect_dataset_snapshot(sample_dataset)
        with pytest.raises(Exception):
            snap.row_count = 999  # type: ignore[misc]

    def test_schema_json_is_valid_and_sorted(self, sample_dataset: Path) -> None:
        import json

        snap = _collect_dataset_snapshot(sample_dataset)
        parsed = json.loads(snap.schema_json)
        assert isinstance(parsed, dict)
        # sort_keys=True garante ordem lexicográfica das chaves
        assert list(parsed.keys()) == sorted(parsed.keys())


@pytest.mark.unit
class TestResolveSelectionMetric:
    def test_default_when_env_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MODEL_SELECTION_METRIC", raising=False)
        assert _resolve_selection_metric() == DEFAULT_SELECTION_METRIC

    def test_returns_env_value_when_supported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MODEL_SELECTION_METRIC", "recall")
        assert _resolve_selection_metric() == "recall"

    def test_falls_back_on_unsupported_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MODEL_SELECTION_METRIC", "made_up_metric")
        assert _resolve_selection_metric() == DEFAULT_SELECTION_METRIC

    def test_all_supported_metrics_are_recognized(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for metric in SUPPORTED_SELECTION_METRICS:
            monkeypatch.setenv("MODEL_SELECTION_METRIC", metric)
            assert _resolve_selection_metric() == metric


@pytest.mark.unit
class TestComputeMetrics:
    def test_perfect_predictions_score_one(self) -> None:
        y = pd.Series([0, 1, 1, 0, 1])
        proba = pd.Series([0.1, 0.9, 0.8, 0.2, 0.95])
        metrics = _compute_metrics(y, y, proba)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_score"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["roc_auc"] == 1.0

    def test_missing_proba_yields_nan_roc_auc(self) -> None:
        y_true = pd.Series([0, 1, 1, 0])
        y_pred = pd.Series([0, 1, 0, 0])
        metrics = _compute_metrics(y_true, y_pred, None)
        assert metrics["roc_auc"] != metrics["roc_auc"]  # NaN
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_single_class_in_holdout_degrades_gracefully(self) -> None:
        # roc_auc não é definido com uma única classe — função deve degradar para NaN
        y_true = pd.Series([1, 1, 1, 1])
        y_pred = pd.Series([1, 1, 1, 0])
        proba = pd.Series([0.9, 0.8, 0.7, 0.4])
        metrics = _compute_metrics(y_true, y_pred, proba)
        assert metrics["roc_auc"] != metrics["roc_auc"]  # NaN
