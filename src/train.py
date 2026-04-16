import os
import pandas as pd
import logging
from pathlib import Path
from typing import Union
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import mlflow
import mlflow.sklearn
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone
from src.config import use_dask_mode

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


def _train_standard(data_p: Path):
    """Fluxo padrão: RF + LR + SVM com train_test_split em memória."""
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


def _train_incremental(data_p: Path):
    """Fluxo Big Data: SGDClassifier com partial_fit em chunks de 50 000 linhas."""
    CHUNK = 50_000
    CLASSES = [0, 1]

    mlflow.set_experiment("Pima_Diabetes_Pipeline")

    with mlflow.start_run(run_name="Training_SGD_Incremental") as run:
        logger.info(
            "Modo Big Data: treinamento incremental com SGDClassifier (log_loss)."
        )

        clf = SGDClassifier(loss="log_loss", random_state=42)
        imputer = SimpleImputer(strategy="median")

        # Primeira passagem: ajusta o imputer no primeiro chunk e inicializa o modelo
        first_chunk = True
        n_total = 0

        try:
            for chunk in pd.read_csv(data_p, chunksize=CHUNK):
                X_chunk = chunk.drop("class", axis=1)
                y_chunk = chunk["class"]

                if first_chunk:
                    imputer.fit(X_chunk)
                    first_chunk = False

                X_imp = imputer.transform(X_chunk)
                clf.partial_fit(X_imp, y_chunk, classes=CLASSES)
                n_total += len(chunk)
                logger.info(f"  chunk processado — linhas acumuladas: {n_total}")
        except Exception as e:
            logger.error(f"Erro durante treinamento incremental: {e}")
            raise

        # Avaliação final no último chunk (proxy rápido — sem holdout dedicado)
        try:
            last_chunk = pd.read_csv(data_p, chunksize=CHUNK)
            eval_chunk = next(iter(last_chunk))
            X_eval = imputer.transform(eval_chunk.drop("class", axis=1))
            y_eval = eval_chunk["class"]
            preds = clf.predict(X_eval)
            acc = accuracy_score(y_eval, preds)
            f1 = f1_score(y_eval, preds, zero_division=0)
        except Exception:
            acc, f1 = 0.0, 0.0

        mlflow.log_param("algorithm", "SGDClassifier")
        mlflow.log_param("loss", "log_loss")
        mlflow.log_param("chunk_size", CHUNK)
        mlflow.log_param("total_rows", n_total)
        mlflow.log_metric("accuracy_last_chunk", acc)
        mlflow.log_metric("f1_last_chunk", f1)

        model_info = mlflow.sklearn.log_model(clf, "model")

        save_training_metadata(
            "SGD_Incremental", acc, f1, str(data_p), model_info.model_uri
        )
        logger.info(
            f"SGD Incremental — linhas: {n_total} | Acc(último chunk): {acc:.4f} | F1: {f1:.4f}"
        )

        registered = mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model",
            name="PimaDiabetes_SGD_Incremental",
        )
        logger.info(
            f"Artefato 'PimaDiabetes_SGD_Incremental' registrado (Versão: {registered.version})."
        )


def train_model(data_path: Union[str, Path]):
    data_p = Path(data_path)
    logger.info(f"Carregando dados processados de: {data_p}")

    if not data_p.exists():
        logger.error(f"Arquivo {data_p} não encontrado.")
        return

    if use_dask_mode(str(data_p)):
        logger.info("Modo Big Data detectado — usando treinamento incremental (SGD).")
        _train_incremental(data_p)
    else:
        logger.info("Modo padrão — usando RF, LR e SVM.")
        _train_standard(data_p)


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
