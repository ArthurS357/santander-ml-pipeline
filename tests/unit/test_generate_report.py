"""Testes unitários do Data Drift Report (PSI próprio, sem Evidently)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.generate_report import (
    FEATURE_COLUMNS,
    calculate_psi,
    generate_data_drift_report,
)


@pytest.mark.unit
class TestCalculatePsi:
    def test_identical_distributions_psi_near_zero(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)
        assert calculate_psi(data, data) < 0.01

    def test_shifted_distributions_psi_high(self) -> None:
        rng = np.random.default_rng(42)
        expected = rng.normal(0, 1, 2000)
        actual = rng.normal(5, 1, 2000)  # forte deslocamento de média
        assert calculate_psi(expected, actual) > 0.25

    def test_empty_input_returns_zero(self) -> None:
        assert calculate_psi([], [1.0, 2.0, 3.0]) == 0.0

    def test_constant_reference_returns_zero(self) -> None:
        assert calculate_psi([5.0] * 100, [5.0] * 100) == 0.0

    def test_handles_nan_without_error(self) -> None:
        expected = [1.0, 2.0, float("nan"), 4.0, 5.0]
        actual = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert calculate_psi(expected, actual) >= 0.0


def _make_reference(tmp_path: Path, rows: int = 200, seed: int = 1) -> Path:
    rng = np.random.default_rng(seed)
    data = {feat: rng.normal(50, 10, rows) for feat in FEATURE_COLUMNS}
    data["class"] = rng.integers(0, 2, rows)
    path = tmp_path / "reference.csv"
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def _make_current(
    tmp_path: Path, rows: int = 50, shift: float = 0.0, seed: int = 2
) -> Path:
    rng = np.random.default_rng(seed)
    data = {feat: rng.normal(50 + shift, 10, rows) for feat in FEATURE_COLUMNS}
    data["prediction"] = rng.integers(0, 2, rows)
    data["probability"] = rng.uniform(0.5, 1.0, rows)
    path = tmp_path / "current.csv"
    pd.DataFrame(data).to_csv(path, index=False)
    return path


@pytest.mark.unit
class TestGenerateDataDriftReport:
    def test_returns_none_when_reference_missing(self, tmp_path: Path) -> None:
        current = _make_current(tmp_path)
        result = generate_data_drift_report(
            reference_path=tmp_path / "missing.csv",
            current_path=current,
            output_dir=tmp_path / "reports",
        )
        assert result is None

    def test_returns_none_when_insufficient_rows(self, tmp_path: Path) -> None:
        reference = _make_reference(tmp_path)
        current = _make_current(tmp_path, rows=5)  # < _MIN_CURRENT_ROWS
        result = generate_data_drift_report(
            reference_path=reference,
            current_path=current,
            output_dir=tmp_path / "reports",
        )
        assert result is None

    def test_generates_json_and_md(self, tmp_path: Path) -> None:
        reference = _make_reference(tmp_path)
        current = _make_current(tmp_path, shift=0.0)
        out_dir = tmp_path / "reports"

        result = generate_data_drift_report(
            reference_path=reference, current_path=current, output_dir=out_dir
        )

        assert result is not None
        json_path = Path(result)
        assert json_path.exists() and json_path.suffix == ".json"
        assert json_path.with_suffix(".md").exists()

        report = json.loads(json_path.read_text(encoding="utf-8"))
        assert set(report["features"]) == set(FEATURE_COLUMNS)
        assert report["current_rows"] == 50
        assert report["prediction"]["confidence_mean"] is not None

    def test_detects_drift_on_shifted_features(self, tmp_path: Path) -> None:
        reference = _make_reference(tmp_path)
        current = _make_current(tmp_path, shift=40.0)  # drift forte
        out_dir = tmp_path / "reports"

        result = generate_data_drift_report(
            reference_path=reference, current_path=current, output_dir=out_dir
        )

        assert result is not None
        report = json.loads(Path(result).read_text(encoding="utf-8"))
        assert report["drift_detected"] is True
        assert len(report["alerts"]) > 0
