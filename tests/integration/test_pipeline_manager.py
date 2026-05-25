"""Teste de integração do `MLPipelineOrchestrator`.

Roda o DAG completo (ingestão → pré-processamento → treino) em ambiente
totalmente isolado: todos os artefatos (raw, processed, mlruns, history.db)
vão para `tmp_path`. Nenhuma escrita escapa do diretório efêmero do pytest.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


@pytest.fixture
def isolated_orchestrator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Configura env vars + recarrega módulos para isolar o pipeline em tmp_path."""
    raw_src = tmp_path / "raw_input.csv"
    raw_target = tmp_path / "raw" / "pima_diabetes.csv"
    processed = tmp_path / "processed" / "pima_diabetes_processed.csv"
    db_path = tmp_path / "history.db"
    mlruns_path = tmp_path / "mlruns"
    mlruns_path.mkdir(parents=True, exist_ok=True)

    # Gera dataset bruto pequeno e balanceado (30 linhas, classes 0/1 alternadas)
    header = "preg,plas,pres,skin,test,mass,pedi,age,class"
    rows = [
        f"{i % 10},{100 + i},{60 + i % 20},{20 + i % 10},"
        f"{50 + i},{25.0 + i * 0.1:.1f},{0.3 + i * 0.01:.3f},{20 + i % 40},{i % 2}"
        for i in range(30)
    ]
    raw_src.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")

    monkeypatch.setenv("RAW_DATA_URL", str(raw_src))
    monkeypatch.setenv("RAW_DATA_FILE", str(raw_target))
    monkeypatch.setenv("PROCESSED_DATA_FILE", str(processed))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path.as_posix()}")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", mlruns_path.as_uri())
    monkeypatch.setenv("MODEL_SELECTION_METRIC", "f1_score")

    # Recarrega módulos que congelam env vars no import time
    for mod_name in ("src.train", "src.pipeline_manager"):
        sys.modules.pop(mod_name, None)

    pm = importlib.import_module("src.pipeline_manager")
    try:
        yield {
            "orchestrator_cls": pm.MLPipelineOrchestrator,
            "raw_target": raw_target,
            "processed": processed,
            "db_path": db_path,
        }
    finally:
        for mod_name in ("src.train", "src.pipeline_manager"):
            sys.modules.pop(mod_name, None)


@pytest.mark.integration
def test_full_pipeline_runs_successfully(isolated_orchestrator: dict) -> None:
    """Executa o DAG completo e valida que os 3 artefatos foram criados."""
    orchestrator = isolated_orchestrator["orchestrator_cls"]()
    success = orchestrator.run_pipeline()

    assert success is True
    assert isolated_orchestrator["raw_target"].exists(), "raw CSV não foi gerado"
    assert isolated_orchestrator["processed"].exists(), "processed CSV não foi gerado"
    assert isolated_orchestrator["db_path"].exists(), "SQLite não foi criado"


@pytest.mark.integration
def test_ingestion_fails_gracefully_when_input_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`run_ingestion` retorna False em vez de levantar quando o input some."""
    missing = tmp_path / "nao_existe.csv"
    raw_target = tmp_path / "out_raw.csv"
    monkeypatch.setenv("RAW_DATA_URL", str(missing))
    monkeypatch.setenv("RAW_DATA_FILE", str(raw_target))

    sys.modules.pop("src.pipeline_manager", None)
    pm = importlib.import_module("src.pipeline_manager")
    try:
        orchestrator = pm.MLPipelineOrchestrator()
        assert orchestrator.run_ingestion() is False
    finally:
        sys.modules.pop("src.pipeline_manager", None)


@pytest.mark.integration
def test_run_pipeline_short_circuits_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Se a ingestão falha, as etapas seguintes não são executadas."""
    missing = tmp_path / "nao_existe.csv"
    raw_target = tmp_path / "raw.csv"
    processed = tmp_path / "processed.csv"
    monkeypatch.setenv("RAW_DATA_URL", str(missing))
    monkeypatch.setenv("RAW_DATA_FILE", str(raw_target))
    monkeypatch.setenv("PROCESSED_DATA_FILE", str(processed))

    sys.modules.pop("src.pipeline_manager", None)
    pm = importlib.import_module("src.pipeline_manager")
    try:
        orchestrator = pm.MLPipelineOrchestrator()
        success = orchestrator.run_pipeline()
        assert success is False
        # Como ingestão falhou, o pré-processamento NÃO deve ter rodado
        assert not processed.exists()
    finally:
        sys.modules.pop("src.pipeline_manager", None)
