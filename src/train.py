import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, cast

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.models.model import ModelInfo
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

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

# Métricas suportadas como critério de seleção do melhor modelo.
# A escolha é configurável via env var MODEL_SELECTION_METRIC (default: f1_score),
# permitindo priorizar métricas de negócio (recall em saúde, precision em fraude, etc.).
SUPPORTED_SELECTION_METRICS = frozenset(
    {
        "accuracy",
        "balanced_accuracy",
        "f1_score",
        "precision",
        "recall",
        "roc_auc",
    }
)
DEFAULT_SELECTION_METRIC = "f1_score"

# Nome único e estável no MLflow Model Registry.
REGISTRY_NAME = "PimaDiabetesClassifier"
# Alias que aponta para o modelo aprovado para produção (governança).
CHAMPION_ALIAS = "champion"


class TrainingRecord(Base):
    __tablename__ = "training_records"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    algorithm = Column(String)
    accuracy = Column(Float)
    f1_score = Column(Float)
    data_path = Column(String)
    model_uri = Column(String)


class DatasetSnapshotRecord(Base):
    """Snapshot lógico do dataset usado em um run de treinamento.

    Tabela criada lado a lado com `training_records` para preservar
    retrocompatibilidade. Vincula o arquivo físico (hash SHA-256) ao
    run do MLflow, permitindo auditar qualquer predição até a versão
    exata do dataset que originou o modelo.
    """

    __tablename__ = "training_dataset_snapshots"
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    dataset_path = Column(String, nullable=False)
    dataset_sha256 = Column(String(64), nullable=False, index=True)
    row_count = Column(Integer, nullable=False)
    column_count = Column(Integer, nullable=False)
    target_column = Column(String, nullable=False)
    schema_json = Column(Text, nullable=False)
    mlflow_run_id = Column(String(64), nullable=True, index=True)


Base.metadata.create_all(bind=engine)


@dataclass(frozen=True)
class DatasetSnapshot:
    """Snapshot imutável de um dataset persistido em disco."""

    dataset_path: str
    dataset_sha256: str
    row_count: int
    column_count: int
    target_column: str
    schema: Mapping[str, str]

    @property
    def schema_json(self) -> str:
        return json.dumps(dict(self.schema), sort_keys=True)


def _calculate_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calcula o hash SHA-256 de um arquivo em chunks (memória constante)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_dataset_snapshot(
    data_p: Path, target_column: str = "class"
) -> DatasetSnapshot:
    """Lê o CSV uma única vez (somente cabeçalho + dtypes) e calcula o snapshot.

    O hash é calculado sobre o arquivo bruto em disco — não sobre o DataFrame —
    garantindo que a auditoria reflita o artefato físico usado no treino.
    """
    df = pd.read_csv(data_p)
    # `df.dtypes.items()` retorna chaves Hashable; convertemos para str
    # para casar com o contrato `Mapping[str, str]` do DatasetSnapshot.
    schema: dict[str, str] = {str(col): str(dtype) for col, dtype in df.dtypes.items()}
    rows, cols = df.shape
    return DatasetSnapshot(
        dataset_path=str(data_p),
        dataset_sha256=_calculate_sha256(data_p),
        row_count=rows,
        column_count=cols,
        target_column=target_column,
        schema=schema,
    )


def save_dataset_snapshot(snapshot: DatasetSnapshot, mlflow_run_id: str | None) -> None:
    """Persiste o snapshot em `training_dataset_snapshots`."""
    with SessionLocal() as db:
        try:
            record = DatasetSnapshotRecord(
                dataset_path=snapshot.dataset_path,
                dataset_sha256=snapshot.dataset_sha256,
                row_count=snapshot.row_count,
                column_count=snapshot.column_count,
                target_column=snapshot.target_column,
                schema_json=snapshot.schema_json,
                mlflow_run_id=mlflow_run_id,
            )
            db.add(record)
            db.commit()
        except Exception as exc:
            db.rollback()
            logger.error(f"Erro ao persistir snapshot do dataset: {exc}")
            raise


def save_training_metadata(
    algo_name: str, acc: float, f1: float, data_path: str, model_uri: str
) -> None:
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


def _resolve_selection_metric() -> str:
    """Lê e valida a métrica de seleção via env var (fail-safe para o default)."""
    metric = os.getenv("MODEL_SELECTION_METRIC", DEFAULT_SELECTION_METRIC)
    if metric not in SUPPORTED_SELECTION_METRICS:
        logger.warning(
            f"Métrica '{metric}' não suportada. Usando default '{DEFAULT_SELECTION_METRIC}'. "
            f"Opções válidas: {sorted(SUPPORTED_SELECTION_METRICS)}"
        )
        return DEFAULT_SELECTION_METRIC
    return metric


# Alias para qualquer tipo aceito pelas funções do sklearn.metrics.
# Cobre tanto saídas de `pipeline.predict()` (np.ndarray) quanto holdouts
# em DataFrames (pd.Series).
ArrayLikeMetric = pd.Series | np.ndarray


def _compute_metrics(
    y_true: ArrayLikeMetric,
    y_pred: ArrayLikeMetric,
    y_proba: ArrayLikeMetric | None,
) -> dict[str, float]:
    """Calcula o conjunto completo de métricas suportadas.

    `y_proba` é a probabilidade da classe positiva — necessária só para `roc_auc`.
    Falhas no `roc_auc` (modelo sem `predict_proba`, ou classe única no holdout)
    degradam silenciosamente para `nan`, sem quebrar o run inteiro.
    """
    # Notas de tipo (Pyrefly):
    # - balanced_accuracy_score retorna `float` puro — sem `float()` extra.
    # - accuracy_score retorna `float | int` (depende de `normalize`).
    # - f1/precision/recall/roc_auc retornam `float | ndarray` (multi-classe).
    # Como usamos `average='binary'` (default), o retorno é sempre escalar;
    # `float(...)` faz o estreitamento explícito para o type checker.
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError as exc:
            logger.warning(f"roc_auc indisponível neste run: {exc}")
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


@dataclass
class IncrementalDiabetesModel:
    """Wrapper que une imputer + classificador incremental num artefato único.

    Resolve o bug em que o modo Big Data salvava apenas o `SGDClassifier`,
    descartando o `SimpleImputer` — a inferência então recebia NaN crus.
    Expõe a mesma interface (`predict`/`predict_proba`) que a API consome.
    """

    imputer: SimpleImputer
    classifier: SGDClassifier

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return cast(np.ndarray, self.classifier.predict(self.imputer.transform(X)))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return cast(
            np.ndarray, self.classifier.predict_proba(self.imputer.transform(X))
        )


def _log_model_with_signature(
    model: object, X_sample: pd.DataFrame, y_pred_sample: np.ndarray
) -> ModelInfo:
    """Loga o modelo no MLflow com signature inferida + input_example.

    A signature documenta o schema de entrada/saída no MLmodel, habilitando
    enforcement de tipos no serving e auto-documentação no Registry.
    """
    # float64 evita o aviso de "integer columns" e casa com o payload da API
    # (PatientData usa floats), garantindo enforcement de schema consistente.
    X_sample = X_sample.astype("float64")
    signature = infer_signature(X_sample, y_pred_sample)
    return mlflow.sklearn.log_model(
        model, "model", signature=signature, input_example=X_sample
    )


def _register_with_champion_alias(run_id: str, context: str) -> None:
    """Registra o modelo do run e aponta o alias `champion` para a nova versão.

    O alias separa "modelo registrado" de "modelo aprovado para produção":
    a API consome `models:/PimaDiabetesClassifier@champion`, então promover
    um modelo é só mover o alias — sem redeploy.
    """
    registered = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model", name=REGISTRY_NAME
    )
    try:
        MlflowClient().set_registered_model_alias(
            name=REGISTRY_NAME, alias=CHAMPION_ALIAS, version=registered.version
        )
        logger.info(
            f"Artefato '{REGISTRY_NAME}' ({context}) registrado "
            f"(Versão: {registered.version}) e alias '@{CHAMPION_ALIAS}' atualizado."
        )
    except Exception as exc:  # backend de registry sem suporte a alias
        logger.warning(
            f"Modelo registrado (Versão: {registered.version}), mas falhou ao "
            f"definir alias '@{CHAMPION_ALIAS}': {exc}"
        )


def _train_standard(data_p: Path) -> None:
    """Fluxo padrão: RF + LR + SVM com train_test_split estratificado."""
    try:
        df = pd.read_csv(data_p)
    except Exception as e:
        logger.error(f"Falha ao ler o arquivo CSV: {e}")
        return

    X = df.drop("class", axis=1)
    y = df["class"]

    # stratify=y preserva a proporção 0/1 no holdout — crítico em dataset desbalanceado.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    selection_metric = _resolve_selection_metric()
    logger.info(f"Métrica de seleção do melhor modelo: {selection_metric}")

    mlflow.set_experiment("Pima_Diabetes_Pipeline")

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42
        ),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        # probability=True habilita predict_proba — necessário para roc_auc no SVM.
        "SVM": SVC(probability=True, random_state=42),
    }

    # Snapshot do dataset — uma única leitura/hash compartilhada por todos os runs.
    snapshot = _collect_dataset_snapshot(data_p, target_column="class")
    logger.info(
        f"Dataset snapshot: sha256={snapshot.dataset_sha256[:12]}… "
        f"rows={snapshot.row_count} cols={snapshot.column_count}"
    )

    best_score = float("-inf")
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
            # `predict` declara união `ndarray | tuple` (regressor GP com return_std);
            # para classificadores é sempre `ndarray` — estreitamos para o type checker.
            predictions = cast(np.ndarray, pipeline.predict(X_test))
            proba: np.ndarray | None = (
                pipeline.predict_proba(X_test)[:, 1]
                if hasattr(pipeline.named_steps["classifier"], "predict_proba")
                else None
            )

            metrics = _compute_metrics(y_test, predictions, proba)

            mlflow.log_param("algorithm", name)
            mlflow.log_param("imputation_strategy", "median")
            mlflow.log_param("selection_metric", selection_metric)
            mlflow.log_param("stratified_split", True)
            for metric_name, metric_value in metrics.items():
                # MLflow rejeita NaN; converte para None (omite a métrica)
                if metric_value == metric_value:  # NaN check
                    mlflow.log_metric(metric_name, metric_value)

            # Tags MLflow para rastreabilidade do snapshot
            mlflow.set_tag("dataset_sha256", snapshot.dataset_sha256)
            mlflow.set_tag("dataset_rows", snapshot.row_count)
            mlflow.set_tag("dataset_path", snapshot.dataset_path)

            # Persiste snapshot vinculado a este run
            save_dataset_snapshot(snapshot, mlflow_run_id=run.info.run_id)

            model_info = _log_model_with_signature(
                pipeline, X_test.head(5), predictions[:5]
            )

            save_training_metadata(
                name,
                metrics["accuracy"],
                metrics["f1_score"],
                str(data_p),
                model_info.model_uri,
            )
            logger.info(
                f"Resultados {name} -> "
                f"acc={metrics['accuracy']:.4f} f1={metrics['f1_score']:.4f} "
                f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
                f"roc_auc={metrics['roc_auc']:.4f}"
            )

            current_score = metrics[selection_metric]
            if current_score == current_score and current_score > best_score:
                best_score = current_score
                best_model_name = name
                best_run_id = run.info.run_id

    logger.info(
        f"\nMelhor pipeline: {best_model_name} com {selection_metric}={best_score:.4f}"
    )

    if best_run_id:
        _register_with_champion_alias(
            best_run_id,
            context=(
                f"algoritmo vencedor: {best_model_name}, "
                f"{selection_metric}={best_score:.4f}"
            ),
        )


def _train_incremental(data_p: Path) -> None:
    """Fluxo Big Data: SGDClassifier com partial_fit em chunks.

    Reserva os últimos `BIGDATA_HOLDOUT_ROWS` registros como holdout dedicado
    via buffer rolante (memória limitada, sem carregar o arquivo inteiro) e
    salva imputer + classificador juntos no `IncrementalDiabetesModel`.
    """
    CHUNK = int(os.getenv("BIGDATA_CHUNK_SIZE", "50000"))
    HOLDOUT_ROWS = int(os.getenv("BIGDATA_HOLDOUT_ROWS", "1000"))
    CLASSES = [0, 1]

    mlflow.set_experiment("Pima_Diabetes_Pipeline")

    with mlflow.start_run(run_name="Training_SGD_Incremental") as run:
        logger.info(
            "Modo Big Data: treinamento incremental com SGDClassifier (log_loss)."
        )

        clf = SGDClassifier(loss="log_loss", random_state=42)
        imputer = SimpleImputer(strategy="median")

        # Snapshot do dataset (hash do arquivo físico)
        snapshot = _collect_dataset_snapshot(data_p, target_column="class")
        save_dataset_snapshot(snapshot, mlflow_run_id=run.info.run_id)
        mlflow.set_tag("dataset_sha256", snapshot.dataset_sha256)
        mlflow.set_tag("dataset_rows", snapshot.row_count)

        state = {"imputer_fitted": False, "n_trained": 0}

        def _fit_on(part: pd.DataFrame) -> None:
            """Ajusta o imputer no primeiro lote e faz partial_fit no SGD."""
            if part.empty:
                return
            X_part = part.drop("class", axis=1)
            y_part = part["class"]
            if not state["imputer_fitted"]:
                imputer.fit(X_part)
                state["imputer_fitted"] = True
            clf.partial_fit(imputer.transform(X_part), y_part, classes=CLASSES)
            state["n_trained"] += len(part)

        buffer = pd.DataFrame()
        try:
            for chunk in pd.read_csv(data_p, chunksize=CHUNK):
                buffer = pd.concat([buffer, chunk], ignore_index=True)
                # Mantém só os últimos HOLDOUT_ROWS no buffer; treina o excedente.
                if len(buffer) > HOLDOUT_ROWS:
                    overflow = buffer.iloc[:-HOLDOUT_ROWS]
                    buffer = buffer.iloc[-HOLDOUT_ROWS:].reset_index(drop=True)
                    _fit_on(overflow)
                    logger.info(
                        f"  chunk processado — linhas treinadas: {state['n_trained']}"
                    )
        except Exception as e:
            logger.error(f"Erro durante treinamento incremental: {e}")
            raise

        # Holdout dedicado = buffer final (nunca visto no fit).
        holdout = buffer
        if state["n_trained"] == 0:
            # Dataset menor que o holdout: split simples 80/20 do buffer.
            split = max(1, int(len(buffer) * 0.8))
            _fit_on(buffer.iloc[:split])
            holdout = buffer.iloc[split:].reset_index(drop=True)

        model = IncrementalDiabetesModel(imputer=imputer, classifier=clf)

        # Avaliação no holdout dedicado (sem vazamento de dados de treino).
        metrics = {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "f1_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "roc_auc": float("nan"),
        }
        if not holdout.empty:
            X_hold = holdout.drop("class", axis=1)
            y_hold = holdout["class"]
            preds = model.predict(X_hold)
            proba = model.predict_proba(X_hold)[:, 1]
            metrics = _compute_metrics(y_hold, preds, proba)

        mlflow.log_param("algorithm", "SGDClassifier")
        mlflow.log_param("loss", "log_loss")
        mlflow.log_param("chunk_size", CHUNK)
        mlflow.log_param("holdout_rows", len(holdout))
        mlflow.log_param("total_rows", state["n_trained"])
        for metric_name, metric_value in metrics.items():
            if metric_value == metric_value:  # NaN check
                mlflow.log_metric(metric_name, metric_value)

        # Sample para signature/input_example (holdout, com fallback no buffer).
        sig_source = holdout if not holdout.empty else buffer
        X_sig = sig_source.drop("class", axis=1).head(5)
        model_info = _log_model_with_signature(model, X_sig, model.predict(X_sig))

        save_training_metadata(
            "SGD_Incremental",
            metrics["accuracy"],
            metrics["f1_score"],
            str(data_p),
            model_info.model_uri,
        )
        logger.info(
            f"SGD Incremental — treinadas: {state['n_trained']} | "
            f"holdout: {len(holdout)} | "
            f"acc={metrics['accuracy']:.4f} f1={metrics['f1_score']:.4f}"
        )

        _register_with_champion_alias(run.info.run_id, context="modo incremental SGD")


def train_model(data_path: str | Path) -> None:
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


# Exportações usadas pelos testes unitários
__all__ = [
    "CHAMPION_ALIAS",
    "DEFAULT_SELECTION_METRIC",
    "DatasetSnapshot",
    "IncrementalDiabetesModel",
    "REGISTRY_NAME",
    "SUPPORTED_SELECTION_METRICS",
    "_calculate_sha256",
    "_collect_dataset_snapshot",
    "_compute_metrics",
    "_resolve_selection_metric",
    "save_dataset_snapshot",
    "train_model",
]


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
