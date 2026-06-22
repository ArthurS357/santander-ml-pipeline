"""Orquestrador do pipeline de ML — DAG sequencial com scheduler opcional."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import schedule

from src.data_ingestion import load_and_save_data
from src.generate_report import generate_data_drift_report
from src.preprocessing import preprocess_data
from src.train import train_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Defaults alinhados com `train.py` (env vars `RAW_DATA_FILE`/`PROCESSED_DATA_FILE`).
_DEFAULT_RAW = Path("data/raw/pima_diabetes.csv")
_DEFAULT_PROCESSED = Path("data/processed/pima_diabetes_processed.csv")


# [TALK & SHOW: Slide 3] Orquestrador do DAG — coração da arquitetura end-to-end
class MLPipelineOrchestrator:
    """Orquestrador de Pipeline de ML (Simulação de DAG).

    `run_pipeline` executa o DAG de treino: Ingestão → Pré-processamento →
    Treinamento. O Reporting (Data Drift) NÃO faz parte de `run_pipeline`: é
    disparado pelo scheduler (`schedule_pipeline`) ou sob demanda via
    `run_reporting` (`make drift`), pois depende de logs de inferência
    acumulados em produção.
    """

    def __init__(self) -> None:
        # Permite apontar para storage compartilhado em produção sem alterar código.
        self.raw_data_url: str = os.getenv("RAW_DATA_URL", str(_DEFAULT_RAW))
        self.raw_data_path: Path = Path(os.getenv("RAW_DATA_FILE", str(_DEFAULT_RAW)))
        self.processed_data_path: Path = Path(
            os.getenv("PROCESSED_DATA_FILE", str(_DEFAULT_PROCESSED))
        )

    # [TALK & SHOW: Slide 3] Etapa 1 do fluxo: ingestão de dados
    def run_ingestion(self) -> bool:
        logger.info("Etapa 1: Iniciando Ingestão de Dados...")
        # `load_and_save_data` já trata local-vs-remoto via extensão/URL — não há
        # razão para dois branches separados; apenas logamos o modo.
        if Path(self.raw_data_url).exists():
            logger.info(
                f"Arquivo raw encontrado localmente ({self.raw_data_url}). "
                "Modo offline ativo."
            )
        try:
            load_and_save_data(self.raw_data_url, self.raw_data_path)
            logger.info("Ingestão concluída com sucesso.")
            return True
        except Exception as e:
            logger.error(f"Erro na Ingestão: {e}")
            return False

    # [TALK & SHOW: Slide 3] Etapa 2 do fluxo: pré-processamento / imputação
    def run_preprocessing(self) -> bool:
        logger.info("Etapa 2: Iniciando Pré-processamento...")
        try:
            preprocess_data(self.raw_data_path, self.processed_data_path)
            logger.info("Pré-processamento concluído com sucesso.")
            return True
        except Exception as e:
            logger.error(f"Erro no Pré-processamento: {e}")
            return False

    # [TALK & SHOW: Slide 3] Etapa 3 do fluxo: treino multi-modelo + registro
    def run_training(self) -> bool:
        logger.info("Etapa 3: Iniciando Treinamento e Comparação de Modelos...")
        try:
            train_model(self.processed_data_path)
            logger.info("Treinamento e Versionamento concluídos com sucesso.")
            return True
        except Exception as e:
            logger.error(f"Erro no Treinamento: {e}")
            return False

    # [TALK & SHOW: Slide 3] Etapa 4 (sob demanda): Data Drift Report
    def run_reporting(self) -> str | None:
        """Etapa 4: Gera o relatório de Data Drift (Big Data Report).

        Compara a distribuição dos dados de treino (referência) contra
        os logs de inferência em produção (current).
        """
        logger.info("Etapa 4: Iniciando geração do Big Data Report (Data Drift)...")
        try:
            report_path = generate_data_drift_report()
            if report_path:
                logger.info(f"Big Data Report gerado com sucesso: {report_path}")
            else:
                logger.warning(
                    "Big Data Report não gerado. "
                    "Verifique se há dados de inferência suficientes em data/logs/."
                )
            return report_path
        except Exception as e:
            logger.error(f"Erro ao gerar Big Data Report: {e}")
            return None

    # [TALK & SHOW: Slide 3] run_pipeline: dispara o DAG sequencial completo
    def run_pipeline(self) -> bool:
        """Executa o pipeline completo (DAG sequencial)."""
        logger.info("=== Iniciando execução do Pipeline de ML ===")
        start_time = time.time()

        success = self.run_ingestion()
        if success:
            success = self.run_preprocessing()
        if success:
            success = self.run_training()

        if success:
            logger.info(
                f"=== Pipeline finalizado com SUCESSO em {time.time() - start_time:.2f}s ==="
            )
            return True
        logger.error("=== Pipeline finalizado com FALHA ===")
        return False


def schedule_pipeline(demo_mode: bool = False) -> None:
    orchestrator = MLPipelineOrchestrator()

    if demo_mode:
        logger.info("Modo DEMO ativo: pipeline a cada 1 min | report a cada 2 min.")
        schedule.every(1).minutes.do(orchestrator.run_pipeline)
        schedule.every(2).minutes.do(orchestrator.run_reporting)
    else:
        logger.info(
            "Modo PRODUÇÃO: pipeline a cada 24h | report diariamente à meia-noite."
        )
        schedule.every(24).hours.do(orchestrator.run_pipeline)
        schedule.every().day.at("00:00").do(orchestrator.run_reporting)

    # Execução inicial imediata antes de entrar no loop de agendamento
    orchestrator.run_pipeline()

    logger.info("Agendador iniciado. Aguardando próxima execução...")
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]

    if "--demo" in args or "--schedule" in args:
        # Modos interativos: entram no loop de agendamento (uso local/produção)
        demo = "--demo" in args
        schedule_pipeline(demo_mode=demo)
    else:
        # Modo CI/CD: executa o pipeline uma única vez e encerra
        logger.info("Modo CI: execução única do pipeline.")
        orchestrator = MLPipelineOrchestrator()
        success = orchestrator.run_pipeline()
        if not success:
            sys.exit(1)
