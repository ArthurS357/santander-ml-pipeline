from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn
import os
import csv
import logging
import threading
import time
from pathlib import Path
from prometheus_fastapi_instrumentator import Instrumentator

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

# Instrumentação para Prometheus (Métricas de monitoramento em tempo real)
Instrumentator().instrument(app).expose(app)


# Passo 2: Definir a estrutura dos dados de entrada
class PatientData(BaseModel):
    preg: float
    plas: float
    pres: float
    skin: float
    test: float
    mass: float
    pedi: float
    age: float


# Passo 3: Carregar o modelo treinado (Lógica Robusta)
modelo = None
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


def load_latest_model():
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


def log_prediction(input_data: dict, prediction: int, probability: float) -> None:
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
@app.post("/predict")
async def predict(
    data: PatientData, request: Request, background_tasks: BackgroundTasks
):
    start_time = time.time()

    if modelo is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Execute o pipeline de treinamento primeiro.",
        )

    # Log de Entrada (Observabilidade: Rastreabilidade)
    client_ip = request.client.host
    logger.info(f"Requisição de predição recebida de {client_ip}")

    # Passo 5: Converter dados e prever
    try:
        input_dict = data.dict()
    except AttributeError:
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
        probability=float(probabilidade),
    )

    return {
        "predicao": resultado,
        "confianca": round(float(probabilidade), 4),
        "modelo_versao": _get_model_version_id(),
        "latencia_s": round(latency, 4),
    }


@app.get("/")
def health_check():
    return {
        "status": "API ativa",
        "modelo_carregado": modelo is not None,
        "modelo_versao": _get_model_version_id(),
        "metrics_endpoint": "/metrics",
    }


@app.post("/reload_model")
def reload_model():
    """Endpoint para forçar o recarregamento do modelo após um novo deploy (CD/Orquestração)"""
    new_model = load_latest_model()
    return {"status": "Recarregamento solicitado", "sucesso": new_model is not None}
