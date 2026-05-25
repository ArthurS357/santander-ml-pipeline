import os

from fastapi.testclient import TestClient

from src import api
from src.api import app


def test_predict_endpoint():
    """O endpoint /predict deve responder 200 com modelo carregado ou 503 sem modelo."""
    client = TestClient(app)
    payload = {
        "preg": 1.0,
        "plas": 85.0,
        "pres": 66.0,
        "skin": 29.0,
        "test": 0.0,
        "mass": 26.6,
        "pedi": 0.351,
        "age": 31.0,
    }

    response = client.post("/predict", json=payload)

    if api.modelo is None:
        assert response.status_code == 503
    else:
        assert response.status_code == 200
        assert "predicao" in response.json()


def test_health_check():
    """Endpoint legado / sempre retorna 200 e informa estado do modelo."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "API ativa"


def test_liveness_always_alive():
    """Liveness deve responder 200 enquanto o processo estiver vivo."""
    client = TestClient(app)
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_readiness_reflects_model_state():
    """Readiness deve retornar 503 sem modelo carregado e 200 quando carregado."""
    client = TestClient(app)
    response = client.get("/health/ready")
    if api.modelo is None:
        assert response.status_code == 503
    else:
        assert response.status_code == 200
        assert response.json()["modelo_carregado"] is True


def test_reload_model_requires_token(monkeypatch):
    """Sem ADMIN_RELOAD_TOKEN configurado, o endpoint nega acesso (fail-secure)."""
    monkeypatch.delenv("ADMIN_RELOAD_TOKEN", raising=False)
    client = TestClient(app)
    response = client.post("/admin/reload_model")
    assert response.status_code == 503


def test_reload_model_rejects_invalid_token(monkeypatch):
    """Token configurado mas ausente/incorreto no header → 403."""
    monkeypatch.setenv("ADMIN_RELOAD_TOKEN", "expected-secret")
    client = TestClient(app)

    response = client.post("/admin/reload_model")
    assert response.status_code == 403

    response = client.post(
        "/admin/reload_model", headers={"X-Admin-Token": "wrong-secret"}
    )
    assert response.status_code == 403


def test_reload_model_legacy_path_is_gone():
    """A rota antiga /reload_model não existe mais (deve retornar 404)."""
    # Garante estado limpo do ambiente para não confundir com fail-secure
    os.environ.pop("ADMIN_RELOAD_TOKEN", None)
    client = TestClient(app)
    response = client.post("/reload_model")
    assert response.status_code == 404
