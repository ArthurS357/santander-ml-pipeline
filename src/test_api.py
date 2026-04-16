from fastapi.testclient import TestClient
from src.api import app


def test_predict_endpoint():
    """
    Testa se o endpoint /predict responde corretamente com dados simulados.
    """
    # Passo 1: Inicializar o cliente de teste
    client = TestClient(app)

    # Passo 2: Definir um payload de teste (dados de um paciente fictício)
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

    # Passo 3: Enviar requisição POST para a API
    response = client.post("/predict", json=payload)

    # Passo 4: Validar os resultados
    assert response.status_code == 200
    assert "predicao" in response.json()


def test_health_check():
    """
    Testa se o endpoint de saúde responde corretamente.
    """
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "API ativa"
