# Usa uma imagem oficial e leve do Python
FROM python:3.14-slim

# Define a pasta de trabalho dentro do contêiner
WORKDIR /app

# Copia a lista de dependências e instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o restante do código do projeto
# Isso inclui a pasta mlruns necessária para carregar o modelo
COPY . .

# Expõe a porta que a aplicação vai rodar
EXPOSE 8000

# ---------------------------------------------------------------
# Variáveis de ambiente (sobrescreva em produção via docker run -e ou K8s)
# Nomes consistentes com: api.py, train.py, generate_report.py, k8s/
# ---------------------------------------------------------------
ENV MLFLOW_TRACKING_URI="http://servidor-mlflow:5000"
ENV MODEL_URI=""
ENV DATABASE_URL="sqlite:///./training_history.db"
ENV PROCESSED_DATA_FILE="data/processed/pima_diabetes_processed.csv"
ENV INFERENCE_LOG_FILE="data/logs/inference_logs.csv"
ENV DRIFT_THRESHOLD="0.5"
# Offline mode: aponte para espelho interno ou deixe o arquivo em data/raw/
ENV RAW_DATA_URL=""

# Comando para iniciar o servidor web da API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

