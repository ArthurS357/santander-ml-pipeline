"""Fixtures compartilhadas pelos testes (unit + integration).

Garante isolamento de DB SQLite e MLflow tracking store via `tmp_path`
e `monkeypatch`, conforme exigido pelo plano P1. Nenhum efeito colateral
escapa para fora do diretório efêmero do pytest.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Iterator

import pytest


@pytest.fixture
def isolated_train_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[object]:
    """Recarrega `src.train` apontando DB e MLflow para diretórios efêmeros.

    O módulo é recarregado pois o engine SQLAlchemy é construído no import.
    Após o teste, o módulo original é restaurado para não vazar estado.
    """
    db_path = tmp_path / "test_history.db"
    mlruns_path = tmp_path / "mlruns"
    mlruns_path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path.as_posix()}")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", mlruns_path.as_uri())
    monkeypatch.setenv("MODEL_SELECTION_METRIC", "f1_score")

    # Garante reload limpo de src.train (engine + Base reconstruídos)
    original = sys.modules.pop("src.train", None)
    train_mod = importlib.import_module("src.train")

    yield train_mod

    # Restaura o módulo original para não contaminar outros testes
    sys.modules.pop("src.train", None)
    if original is not None:
        sys.modules["src.train"] = original


@pytest.fixture
def sample_dataset(tmp_path: Path) -> Path:
    """Gera um CSV pequeno e balanceado para testes de treinamento."""
    csv_path = tmp_path / "tiny_pima.csv"
    # 30 linhas, classes balanceadas, sem NaN — suficiente para train_test_split + stratify
    lines = ["preg,plas,pres,skin,test,mass,pedi,age,class"]
    for i in range(30):
        cls = i % 2
        lines.append(
            f"{i % 10},{100 + i},{60 + i % 20},{20 + i % 10},"
            f"{50 + i},{25.0 + i * 0.1:.1f},{0.3 + i * 0.01:.3f},{20 + i % 40},{cls}"
        )
    csv_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path


@pytest.fixture(autouse=True)
def _no_user_admin_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Por segurança, remove ADMIN_RELOAD_TOKEN do ambiente antes de cada teste."""
    monkeypatch.delenv("ADMIN_RELOAD_TOKEN", raising=False)
    # Garante que o pytest enxergue o pacote src/ como raiz quando rodado da raiz
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    # Também limpa variáveis que poderiam vazar de uma sessão anterior
    os.environ.pop("MODEL_URI", None)
