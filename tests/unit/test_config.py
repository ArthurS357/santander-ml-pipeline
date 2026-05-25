"""Testes unitários de `src.config.use_dask_mode`.

Cobre todos os caminhos do gating: env var, threshold de tamanho de arquivo
e fallback. Nenhum arquivo > 500 MB é criado — o tamanho é mockado via
`monkeypatch` no `os.path.getsize`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.config import _DASK_SIZE_THRESHOLD, use_dask_mode


@pytest.mark.unit
class TestUseDaskMode:
    @pytest.mark.parametrize(
        ("env_value", "expected"),
        [
            ("true", True),
            ("TRUE", True),
            ("True", True),
            ("false", False),
            ("", False),
        ],
        ids=[
            "USE_DASK=true",
            "USE_DASK=TRUE (case-insensitive)",
            "USE_DASK=True (mixed case)",
            "USE_DASK=false",
            "USE_DASK vazio",
        ],
    )
    def test_env_var_drives_decision(
        self,
        monkeypatch: pytest.MonkeyPatch,
        env_value: str,
        expected: bool,
    ) -> None:
        monkeypatch.setenv("USE_DASK", env_value)
        assert use_dask_mode() is expected

    def test_returns_false_when_no_env_and_no_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("USE_DASK", raising=False)
        assert use_dask_mode() is False
        assert use_dask_mode(None) is False

    def test_returns_false_when_path_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("USE_DASK", raising=False)
        # Path inexistente — não deve ativar Dask mesmo que o filtro de tamanho seja burlado
        assert use_dask_mode(str(tmp_path / "nao_existe.csv")) is False

    def test_returns_true_when_file_above_threshold(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("USE_DASK", raising=False)
        target = tmp_path / "big.csv"
        target.write_text("dummy", encoding="utf-8")
        # Simula arquivo > 500 MB sem alocar 500 MB no disco do CI
        monkeypatch.setattr(os.path, "getsize", lambda _p: _DASK_SIZE_THRESHOLD + 1)
        assert use_dask_mode(str(target)) is True

    def test_returns_false_when_file_below_threshold(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("USE_DASK", raising=False)
        target = tmp_path / "small.csv"
        target.write_text("dummy", encoding="utf-8")
        assert use_dask_mode(str(target)) is False
