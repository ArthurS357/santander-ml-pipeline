"""Testes de schema/validação da API — caminhos não-felizes.

Usa a fixture `api_client` definida em `conftest.py` e consolida os 3
testes de payload inválido em um único parametrize semântico.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

_VALID_PAYLOAD: dict[str, float] = {
    "preg": 1.0,
    "plas": 85.0,
    "pres": 66.0,
    "skin": 29.0,
    "test": 0.0,
    "mass": 26.6,
    "pedi": 0.351,
    "age": 31.0,
}


def _payload_without(field: str) -> dict[str, float]:
    return {k: v for k, v in _VALID_PAYLOAD.items() if k != field}


def _payload_with_invalid(field: str, value: Any) -> dict[str, Any]:
    payload = dict(_VALID_PAYLOAD)
    payload[field] = value
    return payload


@pytest.mark.unit
class TestPredictSchema:
    @pytest.mark.parametrize(
        ("payload", "expected_status"),
        [
            (_payload_without("age"), 422),
            (_payload_without("preg"), 422),
            (_payload_with_invalid("plas", "not-a-number"), 422),
            (_payload_with_invalid("age", None), 422),
            ({}, 422),
        ],
        ids=[
            "missing_age_field",
            "missing_preg_field",
            "non_numeric_plas",
            "null_age",
            "empty_body",
        ],
    )
    def test_invalid_payload_returns_422(
        self, api_client: TestClient, payload: dict, expected_status: int
    ) -> None:
        response = api_client.post("/predict", json=payload)
        assert response.status_code == expected_status

    def test_get_on_predict_returns_405(self, api_client: TestClient) -> None:
        response = api_client.get("/predict")
        assert response.status_code == 405


@pytest.mark.unit
class TestAdminEndpointSafety:
    def test_admin_reload_uses_constant_time_comparison(self) -> None:
        """`secrets.compare_digest` é o método correto contra timing attacks."""
        import inspect

        from src import api

        source = inspect.getsource(api.require_admin_token)
        assert "compare_digest" in source, (
            "Esperado uso de secrets.compare_digest para evitar timing attacks"
        )
