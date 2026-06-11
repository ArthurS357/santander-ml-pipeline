"""Teste de carga da API de inferência — Locust.

Execução (ver docs/performance.md):
    locust -f tests/performance/locustfile.py --host http://localhost:8000

Atenção: a API aplica rate limiting de 10/min por IP no /predict
(PREDICT_RATE_LIMIT). Para um teste de carga real, suba a API com um
limite alto, ex.: `PREDICT_RATE_LIMIT=100000/minute uvicorn src.api:app`.
"""

from __future__ import annotations

from locust import HttpUser, between, task

_PAYLOAD = {
    "preg": 1.0,
    "plas": 85.0,
    "pres": 66.0,
    "skin": 29.0,
    "test": 0.0,
    "mass": 26.6,
    "pedi": 0.351,
    "age": 31.0,
}


class DiabetesApiUser(HttpUser):
    """Usuário virtual que alterna entre predição e health check."""

    wait_time = between(1, 3)

    @task(4)
    def predict(self) -> None:
        self.client.post("/predict", json=_PAYLOAD, name="POST /predict")

    @task(1)
    def readiness(self) -> None:
        self.client.get("/health/ready", name="GET /health/ready")
