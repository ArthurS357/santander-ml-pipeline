import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import os

# Configuração do Banco de Dados SQLite (Persistência de Metadados)
DATABASE_URL = "sqlite:///./training_history.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class TrainingRecord(Base):
    __tablename__ = "training_records"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    algorithm = Column(String)
    accuracy = Column(Float)
    f1_score = Column(Float)
    data_path = Column(String)
    model_uri = Column(String)

# Criar as tabelas no banco de dados
Base.metadata.create_all(bind=engine)

def save_training_metadata(algo_name, acc, f1, data_path, model_uri):
    db = SessionLocal()
    try:
        record = TrainingRecord(
            algorithm=algo_name,
            accuracy=acc,
            f1_score=f1,
            data_path=data_path,
            model_uri=model_uri
        )
        db.add(record)
        db.commit()
    finally:
        db.close()

def train_model(data_path: str):
    """
    Treina múltiplos modelos de classificação, compara o desempenho e 
    registra o melhor no MLflow e no SQLite.
    """
    print(f"Carregando dados processados de: {data_path}")
    if not os.path.exists(data_path):
        print(f"Erro: Arquivo {data_path} não encontrado.")
        return

    df = pd.read_csv(data_path)
    
    # Separar features (X) e target (y)
    X = df.drop('class', axis=1)
    y = df['class']
    
    # Dividir os dados em treino e teste (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Configurar o experimento no MLflow
    mlflow.set_experiment("Pima_Diabetes_Pipeline")
    
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(probability=True, random_state=42)
    }

    best_acc = 0
    best_model_name = ""
    best_run_id = ""

    print("Iniciando treinamento de múltiplos algoritmos...")

    for name, model in models.items():
        with mlflow.start_run(run_name=f"Training_{name}"):
            print(f"Treinando {name}...")
            model.fit(X_train, y_train)
            
            predictions = model.predict(X_test)
            acc = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            
            # Registrar no MLflow
            mlflow.log_param("algorithm", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            
            model_info = mlflow.sklearn.log_model(model, "model")
            model_uri = model_info.model_uri
            
            # Persistir no SQLite
            save_training_metadata(name, acc, f1, data_path, model_uri)
            
            print(f"Resultados {name} -> Acc: {acc:.4f}, F1: {f1:.4f}")
            
            if acc > best_acc:
                best_acc = acc
                best_model_name = name
                best_run_id = mlflow.active_run().info.run_id

    print(f"\nMelhor modelo: {best_model_name} com Accuracy: {best_acc:.4f}")

    # Registrar o melhor modelo no MLflow Model Registry com stage Production
    if best_run_id:
        model_uri = f"runs:/{best_run_id}/model"
        registered = mlflow.register_model(model_uri=model_uri, name="DiabetesClassifier")
        print(f"Modelo '{best_model_name}' registrado no MLflow Registry.")
        print(f"  -> Nome: DiabetesClassifier | Versão: {registered.version} | Run ID: {best_run_id}")

if __name__ == "__main__":
    PROCESSED_DATA_FILE = "data/processed/pima_diabetes_processed.csv"
    train_model(PROCESSED_DATA_FILE)
