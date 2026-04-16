import os
import pandas as pd
import logging
from pathlib import Path
from typing import Union
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import mlflow
import mlflow.sklearn
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone

# Instancia o logger encapsulado para o módulo
logger = logging.getLogger(__name__)

# Carrega a URL do banco de dados da variável de ambiente.
# Dev/CI: usa SQLite como fallback seguro (sem alteração de comportamento).
# Produção: injete DATABASE_URL=postgresql://user:pass@host/db via Docker/Kubernetes.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./training_history.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class TrainingRecord(Base):
    __tablename__ = "training_records"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    algorithm = Column(String)
    accuracy = Column(Float)
    f1_score = Column(Float)
    data_path = Column(String)
    model_uri = Column(String)


Base.metadata.create_all(bind=engine)


def save_training_metadata(
    algo_name: str, acc: float, f1: float, data_path: str, model_uri: str
):
    """Salva os metadados do experimento garantindo atomicidade na transação."""
    with SessionLocal() as db:
        try:
            record = TrainingRecord(
                algorithm=algo_name,
                accuracy=acc,
                f1_score=f1,
                data_path=str(data_path),
                model_uri=model_uri,
            )
            db.add(record)
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(
                f"Erro ao salvar metadados no banco para o algoritmo {algo_name}: {e}"
            )
            raise


def train_model(data_path: Union[str, Path]):
    data_p = Path(data_path)
    logger.info(f"Carregando dados processados de: {data_p}")

    if not data_p.exists():
        logger.error(f"Arquivo {data_p} não encontrado.")
        return

    try:
        df = pd.read_csv(data_p)
    except Exception as e:
        logger.error(f"Falha ao ler o arquivo CSV: {e}")
        return

    X = df.drop("class", axis=1)
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_experiment("Pima_Diabetes_Pipeline")

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42
        ),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
    }

    best_acc = 0
    best_model_name = ""
    best_run_id = ""

    logger.info(
        "Iniciando treinamento de múltiplos algoritmos com Pipeline de Imputação..."
    )

    for name, classifier in models.items():
        with mlflow.start_run(run_name=f"Training_{name}") as run:
            logger.info(f"Treinando {name}...")

            # Construção Estrita do Pipeline para evitar data leakage
            pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("classifier", classifier),
                ]
            )

            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            acc = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)

            mlflow.log_param("algorithm", name)
            mlflow.log_param("imputation_strategy", "median")
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)

            model_info = mlflow.sklearn.log_model(pipeline, "model")

            save_training_metadata(name, acc, f1, str(data_p), model_info.model_uri)
            logger.info(f"Resultados {name} -> Acc: {acc:.4f}, F1: {f1:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_model_name = name
                best_run_id = run.info.run_id

    logger.info(f"\nMelhor pipeline: {best_model_name} com Accuracy: {best_acc:.4f}")

    if best_run_id:
        registry_name = f"PimaDiabetes_{best_model_name}_MedianImputer"
        registered = mlflow.register_model(
            model_uri=f"runs:/{best_run_id}/model", name=registry_name
        )
        logger.info(
            f"Artefato '{registry_name}' registrado no MLflow Registry (Versão: {registered.version})."
        )


if __name__ == "__main__":
    # Configuração de log restrita à execução do script como main
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Lê o caminho do arquivo processado da variável de ambiente.
    # Permite apontar para volumes partilhados ou buckets S3 no futuro sem alterar código.
    _default_processed = "data/processed/pima_diabetes_processed.csv"
    PROCESSED_DATA_FILE = Path(os.getenv("PROCESSED_DATA_FILE", _default_processed))
    train_model(PROCESSED_DATA_FILE)
