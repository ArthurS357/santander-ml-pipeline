# Usa uma imagem oficial e leve do Python
FROM python:3.11-slim

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

# Comando para iniciar o servidor web da API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
