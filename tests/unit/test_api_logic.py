"""Cobertura unitária da lógica interna de `src.api`.

Todos os testes injetam um `DummyModel` via `monkeypatch.setattr` no módulo —
nenhum modelo MLflow real é exigido. Logs de inferência vão para `tmp_path`.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_VALID_PAYLOAD = {
    "preg": 1.0,
    "plas": 85.0,
    "pres": 66.0,
    "skin": 29.0,
    "test": 0.0,
    "mass": 26.6,
    "pedi": 0.351,
    "age": 31.0,
}


# ---------------------------------------------------------------------------
# /predict — caminho feliz
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestPredictHappyPath:
    @pytest.mark.parametrize(
        ("prediction", "expected_label"),
        [
            (1, "Positivo para Diabetes"),
            (0, "Negativo para Diabetes"),
        ],
        ids=["classe_1_positivo", "classe_0_negativo"],
    )
    def test_predict_returns_mapped_label(
        self,
        monkeypatch: pytest.MonkeyPatch,
        api_client: TestClient,
        prediction: int,
        expected_label: str,
    ) -> None:
        from src import api
        from tests.conftest import DummyModel

        monkeypatch.setattr(api, "modelo", DummyModel(prediction=prediction))
        monkeypatch.setattr(api, "modelo_path", "mlruns/1/RUNID/artifacts/model")

        response = api_client.post("/predict", json=_VALID_PAYLOAD)

        assert response.status_code == 200
        body = response.json()
        assert body["predicao"] == expected_label
        assert 0.0 <= body["confianca"] <= 1.0
        assert body["modelo_versao"].startswith("run_")
        assert isinstance(body["latencia_s"], float)

    def test_predict_handles_model_without_predict_proba(
        self,
        monkeypatch: pytest.MonkeyPatch,
        api_client: TestClient,
    ) -> None:
        """Modelo sem `predict_proba` deve retornar confiança 1.0 (fallback)."""
        import numpy as np

        from src import api

        class NoProbaModel:
            def predict(self, X):  # noqa: ANN001
                return np.array([1])

        monkeypatch.setattr(api, "modelo", NoProbaModel())
        monkeypatch.setattr(api, "modelo_path", "")

        response = api_client.post("/predict", json=_VALID_PAYLOAD)

        assert response.status_code == 200
        assert response.json()["confianca"] == 1.0


# ---------------------------------------------------------------------------
# /health/ready — caminho feliz
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestReadinessHappyPath:
    def test_readiness_returns_200_when_model_loaded(
        self, dummy_model_loaded, api_client: TestClient
    ) -> None:
        response = api_client.get("/health/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ready"
        assert body["modelo_carregado"] is True
        assert body["modelo_versao"] == "run_abc123def456"


# ---------------------------------------------------------------------------
# _get_model_version_id — extração do run_id
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestGetModelVersionId:
    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("mlruns/1/abc123def456/artifacts/model", "run_abc123def456"),
            ("mlruns/0/RUN_XYZ/artifacts/model", "run_RUN_XYZ"),
            ("MLRUNS/1/lower_test/artifacts/model", "run_lower_test"),
        ],
        ids=["mlruns_path", "experiment_0", "case_insensitive_mlruns"],
    )
    def test_extracts_run_id_from_mlruns_path(
        self, monkeypatch: pytest.MonkeyPatch, path: str, expected: str
    ) -> None:
        from src import api

        monkeypatch.setattr(api, "modelo_path", path)
        assert api._get_model_version_id() == expected

    def test_returns_desconhecido_when_path_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src import api

        monkeypatch.setattr(api, "modelo_path", "")
        assert api._get_model_version_id() == "desconhecido"

    def test_fallback_when_path_not_mlruns_shaped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src import api

        # Estrutura não-mlruns: fallback usa `parts[-3]`
        monkeypatch.setattr(api, "modelo_path", "/var/models/v1/artifacts/model")
        # /var/models/v1/artifacts/model → parts[-3] = "v1"
        assert api._get_model_version_id() == "run_v1"


# ---------------------------------------------------------------------------
# log_prediction — escrita CSV thread-safe
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestLogPrediction:
    def test_writes_header_on_first_call(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from src import api

        log_file = tmp_path / "inference_logs.csv"
        monkeypatch.setattr(api, "INFERENCE_LOG_FILE", str(log_file))

        api.log_prediction(_VALID_PAYLOAD, prediction=1, probability=0.91)

        assert log_file.exists()
        with log_file.open() as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == api.LOG_FIELDNAMES
            rows = list(reader)
        assert len(rows) == 1
        assert int(rows[0]["prediction"]) == 1
        assert float(rows[0]["probability"]) == 0.91

    def test_does_not_duplicate_header_on_subsequent_calls(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from src import api

        log_file = tmp_path / "inference_logs.csv"
        monkeypatch.setattr(api, "INFERENCE_LOG_FILE", str(log_file))

        api.log_prediction(_VALID_PAYLOAD, prediction=1, probability=0.91)
        api.log_prediction(_VALID_PAYLOAD, prediction=0, probability=0.12)

        with log_file.open() as f:
            lines = f.readlines()
        # 1 header + 2 rows = 3 linhas
        assert len(lines) == 3

    def test_creates_parent_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from src import api

        log_file = tmp_path / "deep" / "nested" / "inference_logs.csv"
        monkeypatch.setattr(api, "INFERENCE_LOG_FILE", str(log_file))
        assert not log_file.parent.exists()

        api.log_prediction(_VALID_PAYLOAD, prediction=1, probability=0.5)

        assert log_file.exists()


# ---------------------------------------------------------------------------
# load_latest_model — ambas as estratégias
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestLoadLatestModel:
    def test_strategy_1_model_uri_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Quando MODEL_URI está setado, usa mlflow.sklearn.load_model."""
        from src import api
        from tests.conftest import DummyModel

        dummy = DummyModel()
        monkeypatch.setenv("MODEL_URI", "models:/PimaDiabetesClassifier/1")
        monkeypatch.setattr("mlflow.sklearn.load_model", lambda uri: dummy)

        result = api.load_latest_model()

        assert result is dummy
        assert api.modelo is dummy
        assert api.modelo_path == "models:/PimaDiabetesClassifier/1"

    def test_strategy_1_model_uri_failure_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exceção na Estratégia 1 deve retornar None sem propagar."""
        from src import api

        monkeypatch.setenv("MODEL_URI", "models:/Inexistente/99")

        def _fail(uri: str) -> object:
            raise RuntimeError("modelo não existe no registry")

        monkeypatch.setattr("mlflow.sklearn.load_model", _fail)

        assert api.load_latest_model() is None

    def test_strategy_2_local_fallback_success(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Sem MODEL_URI, busca em mlruns/ e carrega o mais recente."""
        from src import api
        from tests.conftest import DummyModel

        monkeypatch.delenv("MODEL_URI", raising=False)
        # Cria estrutura mlruns/exp/run/artifacts/model/MLmodel
        model_dir = tmp_path / "mlruns" / "1" / "RUN_LOCAL" / "artifacts" / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "MLmodel").write_text("artifact_path: model\n", encoding="utf-8")

        monkeypatch.chdir(tmp_path)
        dummy = DummyModel()
        monkeypatch.setattr("mlflow.sklearn.load_model", lambda p: dummy)

        result = api.load_latest_model()

        assert result is dummy
        assert "RUN_LOCAL" in api.modelo_path

    def test_strategy_2_no_mlruns_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Sem MODEL_URI e sem arquivos MLmodel → retorna None com warning."""
        from src import api

        monkeypatch.delenv("MODEL_URI", raising=False)
        # tmp_path está vazio, então mlruns/**/MLmodel não casa
        (tmp_path / "mlruns").mkdir()
        monkeypatch.chdir(tmp_path)

        assert api.load_latest_model() is None

    def test_strategy_2_exception_during_load_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Exceção dentro da Estratégia 2 deve retornar None."""
        from src import api

        monkeypatch.delenv("MODEL_URI", raising=False)
        model_dir = tmp_path / "mlruns" / "1" / "RUN_BAD" / "artifacts" / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "MLmodel").write_text("invalid", encoding="utf-8")

        monkeypatch.chdir(tmp_path)

        def _fail(p: str) -> object:
            raise RuntimeError("artefato corrompido")

        monkeypatch.setattr("mlflow.sklearn.load_model", _fail)

        assert api.load_latest_model() is None


# ---------------------------------------------------------------------------
# /admin/reload_model — com token válido (caminho feliz)
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestAdminReloadWithValidToken:
    def test_returns_200_when_token_matches_and_model_reloads(
        self,
        monkeypatch: pytest.MonkeyPatch,
        api_client: TestClient,
    ) -> None:
        from tests.conftest import DummyModel

        monkeypatch.setenv("ADMIN_RELOAD_TOKEN", "expected-secret-42")
        # Garante que `load_latest_model` chamado pelo endpoint encontre algo
        monkeypatch.setenv("MODEL_URI", "models:/PimaDiabetesClassifier/1")
        monkeypatch.setattr("mlflow.sklearn.load_model", lambda uri: DummyModel())

        response = api_client.post(
            "/admin/reload_model",
            headers={"X-Admin-Token": "expected-secret-42"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "Recarregamento solicitado"
        assert body["sucesso"] is True

    def test_returns_200_with_sucesso_false_when_reload_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
        api_client: TestClient,
    ) -> None:
        """Token correto + falha no load → endpoint ainda responde 200, sucesso=False."""

        monkeypatch.setenv("ADMIN_RELOAD_TOKEN", "secret-99")
        monkeypatch.setenv("MODEL_URI", "models:/Quebrado/1")
        monkeypatch.setattr(
            "mlflow.sklearn.load_model",
            lambda uri: (_ for _ in ()).throw(RuntimeError("404")),
        )

        response = api_client.post(
            "/admin/reload_model", headers={"X-Admin-Token": "secret-99"}
        )

        assert response.status_code == 200
        assert response.json()["sucesso"] is False
