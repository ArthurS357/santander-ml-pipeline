"""Testes unitários de `src.data_ingestion`.

Cobre o despacho por extensão (CSV, Parquet) ponta-a-ponta usando datasets
pequenos via `tmp_path`. Excel é coberto condicionalmente — só roda se
`openpyxl` estiver instalado.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_ingestion import _EXPECTED_COLUMNS, load_and_save_data, load_data


def _make_csv(path: Path, n_rows: int = 5) -> Path:
    header = ",".join(_EXPECTED_COLUMNS)
    rows = [
        f"{i},{100 + i},{60 + i},{20 + i},{50 + i},{25.0 + i},{0.3:.3f},{30 + i},{i % 2}"
        for i in range(n_rows)
    ]
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


@pytest.mark.unit
class TestLoadData:
    def test_load_csv_returns_dataframe_with_expected_schema(
        self, tmp_path: Path
    ) -> None:
        csv = _make_csv(tmp_path / "data.csv", n_rows=10)
        df = load_data(csv)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (10, 9)
        assert list(df.columns) == _EXPECTED_COLUMNS

    def test_load_parquet_returns_dataframe(self, tmp_path: Path) -> None:
        pytest.importorskip("pyarrow")
        csv = _make_csv(tmp_path / "src.csv", n_rows=8)
        parquet = tmp_path / "data.parquet"
        pd.read_csv(csv).to_parquet(parquet)
        df = load_data(parquet)
        assert df.shape == (8, 9)
        assert list(df.columns) == _EXPECTED_COLUMNS

    def test_load_excel_returns_dataframe(self, tmp_path: Path) -> None:
        pytest.importorskip("openpyxl")
        csv = _make_csv(tmp_path / "src.csv", n_rows=6)
        excel = tmp_path / "data.xlsx"
        # Salva COM header — `_read_excel` usa `names=` que sobrescreve o header existente
        pd.read_csv(csv).to_excel(excel, index=False)
        df = load_data(excel)
        assert df.shape == (6, 9)
        assert list(df.columns) == _EXPECTED_COLUMNS

    def test_accepts_string_or_path(self, tmp_path: Path) -> None:
        csv = _make_csv(tmp_path / "data.csv", n_rows=3)
        # Aceita ambos os tipos sem TypeError
        assert load_data(csv).shape == load_data(str(csv)).shape


@pytest.mark.unit
class TestLoadAndSaveData:
    def test_persists_csv_with_expected_columns(self, tmp_path: Path) -> None:
        src = _make_csv(tmp_path / "input.csv", n_rows=4)
        dst = tmp_path / "output" / "saved.csv"

        load_and_save_data(src, dst)

        assert dst.exists()
        df = pd.read_csv(dst)
        assert df.shape == (4, 9)
        assert list(df.columns) == _EXPECTED_COLUMNS

    def test_creates_parent_directory_if_missing(self, tmp_path: Path) -> None:
        src = _make_csv(tmp_path / "input.csv", n_rows=2)
        dst = tmp_path / "deep" / "nested" / "out.csv"
        assert not dst.parent.exists()

        load_and_save_data(src, dst)

        assert dst.exists()

    def test_logs_shape_info(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        src = _make_csv(tmp_path / "input.csv", n_rows=7)
        dst = tmp_path / "out.csv"
        with caplog.at_level("INFO"):
            load_and_save_data(src, dst)
        assert any("shape=(7, 9)" in r.message for r in caplog.records)
