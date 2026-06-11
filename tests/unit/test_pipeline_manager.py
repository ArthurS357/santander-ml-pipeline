"""Testes unitários do orquestrador `src.pipeline_manager`.

Mocka as funções de etapa (ingestão/preproc/treino/report) importadas no
módulo para exercitar os ramos de sucesso e falha do DAG sem rodar o
pipeline real — complementa o teste de integração ponta-a-ponta.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src import pipeline_manager as pm
from src.pipeline_manager import MLPipelineOrchestrator


@pytest.fixture
def orchestrator() -> MLPipelineOrchestrator:
    return MLPipelineOrchestrator()


@pytest.mark.unit
class TestStageSuccessAndFailure:
    def test_run_ingestion_success(
        self, orchestrator: MLPipelineOrchestrator, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pm, "load_and_save_data", lambda *a, **k: None)
        assert orchestrator.run_ingestion() is True

    def test_run_ingestion_failure(
        self, orchestrator: MLPipelineOrchestrator, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(*a: object, **k: object) -> None:
            raise RuntimeError("download falhou")

        monkeypatch.setattr(pm, "load_and_save_data", _boom)
        assert orchestrator.run_ingestion() is False

    def test_run_preprocessing_failure(
        self, orchestrator: MLPipelineOrchestrator, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(*a: object, **k: object) -> None:
            raise ValueError("schema inválido")

        monkeypatch.setattr(pm, "preprocess_data", _boom)
        assert orchestrator.run_preprocessing() is False

    def test_run_training_failure(
        self, orchestrator: MLPipelineOrchestrator, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(*a: object, **k: object) -> None:
            raise RuntimeError("treino falhou")

        monkeypatch.setattr(pm, "train_model", _boom)
        assert orchestrator.run_training() is False


@pytest.mark.unit
class TestRunReporting:
    def test_returns_path_on_success(
        self, orchestrator: MLPipelineOrchestrator, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            pm, "generate_data_drift_report", lambda: "reports/drift.json"
        )
        assert orchestrator.run_reporting() == "reports/drift.json"

    def test_returns_none_when_report_skipped(
        self, orchestrator: MLPipelineOrchestrator, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pm, "generate_data_drift_report", lambda: None)
        assert orchestrator.run_reporting() is None

    def test_returns_none_on_exception(
        self, orchestrator: MLPipelineOrchestrator, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom() -> None:
            raise RuntimeError("erro no report")

        monkeypatch.setattr(pm, "generate_data_drift_report", _boom)
        assert orchestrator.run_reporting() is None


@pytest.mark.unit
class TestRunPipeline:
    def test_full_success(
        self, orchestrator: MLPipelineOrchestrator, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pm, "load_and_save_data", lambda *a, **k: None)
        monkeypatch.setattr(pm, "preprocess_data", lambda *a, **k: None)
        monkeypatch.setattr(pm, "train_model", lambda *a, **k: None)
        assert orchestrator.run_pipeline() is True

    def test_stops_on_ingestion_failure(
        self, orchestrator: MLPipelineOrchestrator, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(*a: object, **k: object) -> None:
            raise RuntimeError("falha ingestão")

        monkeypatch.setattr(pm, "load_and_save_data", _boom)
        # preproc/treino não devem ser chamados — se forem, falham o teste
        monkeypatch.setattr(
            pm, "preprocess_data", lambda *a, **k: pytest.fail("não deveria chamar")
        )
        assert orchestrator.run_pipeline() is False


@pytest.mark.unit
class TestSchedulePipeline:
    @pytest.mark.parametrize("demo", [True, False], ids=["demo", "prod"])
    def test_scheduler_setup_and_loop_entry(
        self, monkeypatch: pytest.MonkeyPatch, demo: bool
    ) -> None:
        """Cobre a configuração do agendador e a entrada no loop (sem travar)."""
        fake_schedule = MagicMock()
        monkeypatch.setattr(pm, "schedule", fake_schedule)
        # run_pipeline inicial vira no-op
        monkeypatch.setattr(MLPipelineOrchestrator, "run_pipeline", lambda self: True)

        # Quebra o `while True` na primeira iteração.
        def _stop(*_: object) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(pm.time, "sleep", _stop)

        with pytest.raises(KeyboardInterrupt):
            pm.schedule_pipeline(demo_mode=demo)

        fake_schedule.every.assert_called()
