"""Testes de schema/validação da API — caminhos não-felizes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import app

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


@pytest.mark.unit
class TestPredictSchema:
    def test_missing_required_field_returns_422(self) -> None:
        client = TestClient(app)
        bad = dict(_VALID_PAYLOAD)
        del bad["age"]
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_non_numeric_field_returns_422(self) -> None:
        client = TestClient(app)
        bad = dict(_VALID_PAYLOAD)
        bad["plas"] = "not-a-number"
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_empty_body_returns_422(self) -> None:
        client = TestClient(app)
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_get_on_predict_returns_405(self) -> None:
        client = TestClient(app)
        response = client.get("/predict")
        assert response.status_code == 405


@pytest.mark.unit
class TestAdminEndpointSafety:
    def test_admin_reload_constant_time_comparison_used(self) -> None:
        """Garante que `secrets.compare_digest` (timing-safe) é o método usado."""
        import inspect

        from src import api

        source = inspect.getsource(api.require_admin_token)
        assert "compare_digest" in source, (
            "Esperado uso de secrets.compare_digest para evitar timing attacks"
        )
