from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn
import os
import logging
import time
from pathlib import Path
from prometheus_fastapi_instrumentator import Instrumentator

# Configuração de Logging para Observabilidade
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DiabetesAPI")

# Passo 1: Inicializar o aplicativo FastAPI
app = FastAPI(
    title="Diabetes Prediction API", 
    description="API para previsão de diabetes com monitoramento em tempo real (Observabilidade)"
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

def load_latest_model():
    global modelo, modelo_path
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
        
        logger.info(f"Carregando o modelo mais recente de: {modelo_path}")
        modelo = mlflow.sklearn.load_model(modelo_path)
        logger.info("Modelo carregado com sucesso!")
        return modelo
    except Exception as e:
        logger.error(f"Erro ao carregar o modelo: {e}")
        return None

# Carga inicial
load_latest_model()

# Passo 4: Criar o endpoint de predição
@app.post("/predict")
async def predict(data: PatientData, request: Request):
    start_time = time.time()
    
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado. Execute o pipeline de treinamento primeiro.")
    
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
    probabilidade = modelo.predict_proba(df_entrada).max() if hasattr(modelo, "predict_proba") else 1.0
    
    resultado = "Positivo para Diabetes" if predicao[0] == 1 else "Negativo para Diabetes"
    
    # Log de Saída e Performance (Observabilidade)
    latency = time.time() - start_time
    logger.info(f"Predição: {resultado} | Confiança: {probabilidade:.2f} | Latência: {latency:.4f}s")
    
    return {
        "predicao": resultado,
        "confianca": round(float(probabilidade), 4),
        "modelo_versao": modelo_path,
        "latencia_s": round(latency, 4)
    }

@app.get("/")
def health_check():
    return {
        "status": "API ativa",
        "modelo_carregado": modelo is not None,
        "modelo_path": modelo_path,
        "metrics_endpoint": "/metrics"
    }

@app.post("/reload_model")
def reload_model():
    """Endpoint para forçar o recarregamento do modelo após um novo deploy (CD/Orquestração)"""
    new_model = load_latest_model()
    return {"status": "Recarregamento solicitado", "sucesso": new_model is not None}
