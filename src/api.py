from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    status,
)
from pydantic import BaseModel, ConfigDict, Field
import numpy as np
import pandas as pd
import mlflow.sklearn
import os
import csv
import logging
import secrets
import threading
import time
from pathlib import Path
from typing import Protocol
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Configuração de Logging para Observabilidade
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DiabetesAPI")

# Passo 1: Inicializar o aplicativo FastAPI
app = FastAPI(
    title="Diabetes Prediction API",
    description="API para previsão de diabetes com monitoramento em tempo real (Observabilidade)",
)

# Rate limiting por IP de origem (proteção contra abuso/DoS leve).
# Limite configurável via env PREDICT_RATE_LIMIT (default "10/minute").
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
PREDICT_RATE_LIMIT = os.getenv("PREDICT_RATE_LIMIT", "10/minute")

# Instrumentação para Prometheus (Métricas de monitoramento em tempo real)
Instrumentator().instrument(app).expose(app)

# Métricas ML customizadas (registry default → expostas no mesmo /metrics).
# Permitem monitorar drift de negócio: taxa de positivos e distribuição de
# confiança das predições, complementando as métricas HTTP do instrumentator.
PREDICTION_COUNTER = Counter(
    "diabetes_predictions_total",
    "Total de predições emitidas, rotuladas por classe.",
    ["resultado"],
)
CONFIDENCE_HISTOGRAM = Histogram(
    "diabetes_prediction_confidence",
    "Distribuição da confiança (probabilidade máxima) das predições.",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)


# Passo 2: Definir a estrutura dos dados de entrada
class PatientData(BaseModel):
    """Features clínicas do paciente (dataset Pima Indians Diabetes).

    Validação forte: rejeita campos extras (`extra="forbid"`), exige tipos
    estritos (`strict=True`) e impõe faixas plausíveis por feature — payloads
    fora de domínio retornam 422 antes de tocar o modelo.

    NOTA (`strict=True`): envie os valores como **float** no JSON (ex.: `1.0`,
    não `1`). Inteiros são rejeitados com 422 — comportamento intencional de
    validação forte.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    preg: float = Field(..., ge=0, le=20, description="Número de gestações")
    plas: float = Field(..., gt=0, le=250, description="Glicose plasmática (mg/dL)")
    pres: float = Field(..., gt=0, le=150, description="Pressão diastólica (mm Hg)")
    skin: float = Field(
        ..., ge=0, le=100, description="Espessura da dobra cutânea (mm)"
    )
    test: float = Field(..., ge=0, le=1000, description="Insulina sérica (mu U/ml)")
    mass: float = Field(..., gt=0, le=100, description="IMC (kg/m²)")
    pedi: float = Field(..., ge=0, le=3, description="Pedigree de diabetes (função)")
    age: float = Field(..., ge=0, le=120, description="Idade (anos)")


class PredictorModel(Protocol):
    """Contrato mínimo (duck typing) que a API exige de um modelo servível.

    Qualquer artefato carregado do MLflow (Pipeline sklearn, wrapper
    incremental, etc.) deve expor esta interface. `predict_proba` é
    verificado em runtime via `hasattr` — modelos sem ela degradam
    para confiança 1.0 no endpoint.
    """

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...  # noqa: E704

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...  # noqa: E704


class PredictionResponse(BaseModel):
    """Contrato de resposta do POST /predict (documentado no OpenAPI)."""

    predicao: str
    confianca: float
    modelo_versao: str
    latencia_s: float


# Passo 3: Carregar o modelo treinado (Lógica Robusta)
modelo: PredictorModel | None = None
modelo_path = ""


def _get_model_version_id() -> str:
    """Retorna um identificador seguro do modelo sem expor caminhos internos do servidor.
    Extrai o run_id do MLflow a partir da estrutura de diretórios (mlruns/<exp_id>/<run_id>/...).
    """
    if not modelo_path:
        return "desconhecido"
    parts = Path(modelo_path).parts
    # Estrutura esperada: mlruns / <experiment_id> / <run_id> / artifacts / model
    try:
        mlruns_idx = [p.lower() for p in parts].index("mlruns")
        run_id = parts[mlruns_idx + 2]
        return f"run_{run_id}"
    except (ValueError, IndexError):
        # Fallback genérico caso a estrutura de diretórios seja diferente
        return f"run_{Path(modelo_path).parts[-3]}"


def load_latest_model() -> PredictorModel | None:
    global modelo, modelo_path

    # --- Estratégia 1: MODEL_URI via variável de ambiente (Produção / Desacoplado) ---
    model_uri_env = os.getenv("MODEL_URI")
    if model_uri_env:
        try:
            logger.info(
                f"MODEL_URI detectado. Carregando modelo remoto de: {model_uri_env}"
            )
            modelo = mlflow.sklearn.load_model(model_uri_env)
            modelo_path = model_uri_env
            logger.info("Modelo remoto carregado com sucesso!")
            return modelo
        except Exception as e:
            logger.error(
                f"Falha ao carregar modelo via MODEL_URI '{model_uri_env}': {e}"
            )
            return None

    # --- Estratégia 2: Fallback local (Desenvolvimento / CI) ---
    try:
        base_path = Path("mlruns")
        # Busca todas as pastas que contenham o arquivo 'MLmodel'
        mlmodel_files = list(base_path.glob("**/MLmodel"))

        if not mlmodel_files:
            logger.warning("Nenhum modelo encontrado no diretório mlruns.")
            return None

        # Ordena pelos mais recentes com base no tempo de modificação do arquivo 'MLmodel'
        mlmodel_files.sort(key=lambda x: x.stat().st_mtime)

        # O diretório do modelo é o pai do arquivo 'MLmodel'
        latest_model_dir = mlmodel_files[-1].parent
        modelo_path = str(latest_model_dir)

        logger.info(f"Carregando modelo local de: {modelo_path}")
        modelo = mlflow.sklearn.load_model(modelo_path)
        logger.info("Modelo local carregado com sucesso!")
        return modelo
    except Exception as e:
        logger.error(f"Erro ao carregar o modelo local: {e}")
        return None


# Carga inicial
load_latest_model()

# ---------------------------------------------------------------------------
# Inference Logging (Big Data)
# ---------------------------------------------------------------------------
INFERENCE_LOG_FILE = os.getenv("INFERENCE_LOG_FILE", "data/logs/inference_logs.csv")
_log_lock = threading.Lock()

LOG_FIELDNAMES = [
    "timestamp",
    "preg",
    "plas",
    "pres",
    "skin",
    "test",
    "mass",
    "pedi",
    "age",
    "prediction",
    "probability",
]


def log_prediction(
    input_data: dict[str, float], prediction: int, probability: float
) -> None:
    """Anexa um registro de inferência ao CSV de logs de forma thread-safe.
    Executada em background para não impactar a latência da resposta.
    """
    log_path = Path(INFERENCE_LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **input_data,
        "prediction": prediction,
        "probability": round(probability, 6),
    }

    write_header = not log_path.exists() or log_path.stat().st_size == 0

    with _log_lock:
        with log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    logger.debug(
        f"Inferência registrada: prediction={prediction}, prob={probability:.4f}"
    )


# Passo 4: Criar o endpoint de predição
@app.post("/predict", response_model=PredictionResponse)
@limiter.limit(PREDICT_RATE_LIMIT)
async def predict(
    request: Request, data: PatientData, background_tasks: BackgroundTasks
) -> PredictionResponse:
    start_time = time.time()

    if modelo is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Execute o pipeline de treinamento primeiro.",
        )

    # Log de Entrada (Observabilidade: Rastreabilidade)
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"Requisição de predição recebida de {client_ip}")

    # Passo 5: Converter dados e prever
    input_dict = data.model_dump()

    df_entrada = pd.DataFrame([input_dict])

    # Realizando predição
    predicao = modelo.predict(df_entrada)
    probabilidade = (
        modelo.predict_proba(df_entrada).max()
        if hasattr(modelo, "predict_proba")
        else 1.0
    )

    resultado = (
        "Positivo para Diabetes" if predicao[0] == 1 else "Negativo para Diabetes"
    )

    # Métricas ML customizadas (Prometheus)
    PREDICTION_COUNTER.labels(
        resultado="positivo" if predicao[0] == 1 else "negativo"
    ).inc()
    CONFIDENCE_HISTOGRAM.observe(float(probabilidade))

    # Log de Saída e Performance (Observabilidade)
    latency = time.time() - start_time
    logger.info(
        f"Predição: {resultado} | Confiança: {probabilidade:.2f} | Latência: {latency:.4f}s"
    )

    # Big Data Logging: registra a inferência em background (sem impactar latência)
    background_tasks.add_task(
        log_prediction,
        input_data=input_dict,
        prediction=int(predicao[0]),
        probability=probabilidade,
    )

    return PredictionResponse(
        predicao=resultado,
        confianca=round(float(probabilidade), 4),
        modelo_versao=_get_model_version_id(),
        latencia_s=round(latency, 4),
    )


@app.get("/")
def health_check() -> dict[str, object]:
    """Endpoint informativo legado. Sempre 200 — não usar como readiness probe."""
    return {
        "status": "API ativa",
        "modelo_carregado": modelo is not None,
        "modelo_versao": _get_model_version_id(),
        "metrics_endpoint": "/metrics",
    }


@app.get("/health/live")
def liveness() -> dict[str, str]:
    """Liveness probe: processo respondendo. Sempre 200 enquanto o app estiver vivo."""
    return {"status": "alive"}


@app.get("/health/ready")
def readiness() -> dict[str, object]:
    """Readiness probe: 503 se o modelo não estiver carregado — bloqueia tráfego."""
    if modelo is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo não carregado",
        )
    return {
        "status": "ready",
        "modelo_carregado": True,
        "modelo_versao": _get_model_version_id(),
    }


def require_admin_token(x_admin_token: str | None = Header(default=None)) -> None:
    """Dependência fail-secure: nega acesso se ADMIN_RELOAD_TOKEN não estiver configurado."""
    expected = os.getenv("ADMIN_RELOAD_TOKEN")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ADMIN_RELOAD_TOKEN não configurado",
        )
    if not x_admin_token or not secrets.compare_digest(x_admin_token, expected):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Acesso negado",
        )


@app.post("/admin/reload_model")
def reload_model(_: None = Depends(require_admin_token)) -> dict[str, object]:
    """Recarrega o modelo após um novo deploy. Protegido por token administrativo."""
    new_model = load_latest_model()
    return {"status": "Recarregamento solicitado", "sucesso": new_model is not None}
