"""Testes unitários de `src.preprocessing`.

Cobre o caminho Pandas ponta-a-ponta: zeros clínicos viram NaN, colunas
não-clínicas são preservadas, schema parcial gera warning, e erros de I/O
sobem como exceção. O caminho Dask é coberto indiretamente — sua única
diferença é o sufixo `single_file=True` no `to_csv`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing import (
    _COLUNAS_ZEROS,
    _columns_to_clean,
    preprocess_data,
)


@pytest.fixture
def raw_dataset(tmp_path: Path) -> Path:
    """CSV bruto com zeros clínicos para testar conversão para NaN."""
    csv = tmp_path / "raw.csv"
    csv.write_text(
        "preg,plas,pres,skin,test,mass,pedi,age,class\n"
        "6,0,72,35,0,33.6,0.627,50,1\n"
        "1,85,0,29,168,26.6,0.351,31,0\n"
        "0,148,66,0,0,32.0,0.450,25,1\n",
        encoding="utf-8",
    )
    return csv


@pytest.mark.unit
class TestColumnsToClean:
    def test_returns_all_when_schema_complete(self) -> None:
        full_schema = list(_COLUNAS_ZEROS) + ["preg", "age", "class"]
        assert _columns_to_clean(full_schema) == list(_COLUNAS_ZEROS)

    def test_returns_subset_when_partial(self) -> None:
        partial = ["plas", "pres", "age", "class"]
        assert _columns_to_clean(partial) == ["plas", "pres"]

    def test_returns_empty_when_no_clinical_columns(self) -> None:
        assert _columns_to_clean(["preg", "age", "class"]) == []

    def test_logs_warning_when_missing(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            _columns_to_clean(["plas", "pres"])
        assert any("ausentes" in r.message for r in caplog.records)


@pytest.mark.unit
class TestPreprocessData:
    def test_zeros_in_clinical_columns_become_nan(
        self, raw_dataset: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "processed.csv"
        preprocess_data(raw_dataset, out)

        df = pd.read_csv(out)
        # Linha 0 tinha plas=0 e test=0; agora devem ser NaN
        assert pd.isna(df.loc[0, "plas"])
        assert pd.isna(df.loc[0, "test"])
        # `preg` na linha 2 era 0 mas NÃO é coluna clínica — deve permanecer 0
        assert df.loc[2, "preg"] == 0

    def test_non_clinical_columns_preserved(
        self, raw_dataset: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "processed.csv"
        preprocess_data(raw_dataset, out)
        df = pd.read_csv(out)
        # `age` e `class` nunca devem ser tocados
        assert list(df["class"]) == [1, 0, 1]
        assert list(df["age"]) == [50, 31, 25]

    def test_creates_output_directory_if_missing(
        self, raw_dataset: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "nested" / "deep" / "processed.csv"
        assert not out.parent.exists()
        preprocess_data(raw_dataset, out)
        assert out.exists()

    def test_raises_file_not_found_when_input_missing(self, tmp_path: Path) -> None:
        ghost = tmp_path / "nao_existe.csv"
        out = tmp_path / "out.csv"
        with pytest.raises(FileNotFoundError):
            preprocess_data(ghost, out)

    @pytest.mark.parametrize(
        "row_index,column,expected_nan",
        [
            (0, "plas", True),  # zero clínico → NaN
            (0, "test", True),  # zero clínico → NaN
            (1, "pres", True),  # zero clínico → NaN
            (1, "skin", False),  # 29.0 → preservado
            (2, "mass", False),  # 32.0 → preservado
        ],
        ids=[
            "row0_plas_zero_to_nan",
            "row0_test_zero_to_nan",
            "row1_pres_zero_to_nan",
            "row1_skin_29_preserved",
            "row2_mass_32_preserved",
        ],
    )
    def test_specific_cell_transformations(
        self,
        raw_dataset: Path,
        tmp_path: Path,
        row_index: int,
        column: str,
        expected_nan: bool,
    ) -> None:
        out = tmp_path / "processed.csv"
        preprocess_data(raw_dataset, out)
        df = pd.read_csv(out)
        assert pd.isna(df.loc[row_index, column]) is expected_nan
