# Usa uma imagem oficial e leve do Python
FROM python:3.14-slim

# Define a pasta de trabalho dentro do contêiner
WORKDIR /app

# Copia a lista de dependências e instala (como root, para usar /usr/local/lib)
COPY requirements.txt .
# PIP_INDEX_URL pode ser sobrescrito via --build-arg para ambientes com mirror interno
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN pip install --no-cache-dir --index-url ${PIP_INDEX_URL} -r requirements.txt

# Copia o código do projeto. Atenção: .dockerignore exclui mlruns/, data/, *.db.
# Em produção, o modelo deve vir via MODEL_URI (MLflow Registry remoto).
# Em desenvolvimento local, monte mlruns/ como volume (-v $PWD/mlruns:/app/mlruns).
COPY . .

# ---------------------------------------------------------------
# Hardening: usuário não-root e diretórios graváveis explícitos.
# UID/GID 10001 alinhados com securityContext do Kubernetes.
# /app/data/logs precisa de volume emptyDir quando readOnlyRootFilesystem=true.
# ---------------------------------------------------------------
RUN groupadd --system --gid 10001 appuser \
    && useradd --system --uid 10001 --gid 10001 --home-dir /app --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/data/logs \
    && chown -R appuser:appuser /app

# Expõe a porta que a aplicação vai rodar
EXPOSE 8000

# ---------------------------------------------------------------
# Variáveis de ambiente (sobrescreva em produção via docker run -e ou K8s)
# Nomes consistentes com: api.py, train.py, generate_report.py, k8s/
# ---------------------------------------------------------------
ENV MLFLOW_TRACKING_URI="http://servidor-mlflow:5000"
# MODEL_URI deve ser sobrescrito em produção, ex:
#   models:/PimaDiabetesClassifier/1  ou  models:/PimaDiabetesClassifier@production
ENV MODEL_URI=""
ENV DATABASE_URL="sqlite:///./training_history.db"
ENV PROCESSED_DATA_FILE="data/processed/pima_diabetes_processed.csv"
ENV INFERENCE_LOG_FILE="data/logs/inference_logs.csv"
ENV DRIFT_THRESHOLD="0.5"
# Offline mode: aponte para espelho interno ou deixe o arquivo em data/raw/
ENV RAW_DATA_URL=""
# Token administrativo para /admin/reload_model — DEVE ser sobrescrito em produção.
# Vazio = endpoint nega acesso por padrão (fail-secure).
ENV ADMIN_RELOAD_TOKEN=""

# Troca para usuário não-root antes de iniciar o servidor
USER appuser

# Comando para iniciar o servidor web da API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
