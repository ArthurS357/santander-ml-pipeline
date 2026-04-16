import os
import time
import schedule
import logging
from src.data_ingestion import load_and_save_data
from src.preprocessing import preprocess_data
from src.train import train_model
from src.generate_report import generate_data_drift_report

# Configuração de Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MLPipelineOrchestrator:
    """
    Orquestrador de Pipeline de ML (Simulação de DAG).
    Fluxo: Ingestão -> Pré-processamento -> Treinamento -> Monitoramento -> Reporting.
    """

    def __init__(self):
        # RAW_DATA_URL: sobrescreva com caminho local (file://...) ou URL interna
        # em ambientes sem acesso à internet. Deixe vazio para usar arquivo local.
        self.raw_data_url = os.getenv(
            "RAW_DATA_URL",
            "data/raw/pima_diabetes.csv",
        )
        self.raw_data_path = "data/raw/pima_diabetes.csv"
        self.processed_data_path = "data/processed/pima_diabetes_processed.csv"

    def run_ingestion(self):
        logger.info("Etapa 1: Iniciando Ingestão de Dados...")

        # Modo offline: se o arquivo raw já existir localmente, pula o download.
        if os.path.exists(self.raw_data_path):
            logger.info(
                f"Arquivo raw encontrado localmente ({self.raw_data_path}). "
                "Ingestão de rede ignorada (modo offline)."
            )
            return True

        try:
            load_and_save_data(self.raw_data_url, self.raw_data_path)
            logger.info("Ingestão concluída com sucesso.")
            return True
        except Exception as e:
            logger.error(f"Erro na Ingestão: {e}")
            return False

    def run_preprocessing(self):
        logger.info("Etapa 2: Iniciando Pré-processamento...")
        try:
            preprocess_data(self.raw_data_path, self.processed_data_path)
            logger.info("Pré-processamento concluído com sucesso.")
            return True
        except Exception as e:
            logger.error(f"Erro no Pré-processamento: {e}")
            return False

    def run_training(self):
        logger.info("Etapa 3: Iniciando Treinamento e Comparação de Modelos...")
        try:
            train_model(self.processed_data_path)
            logger.info("Treinamento e Versionamento concluídos com sucesso.")
            return True
        except Exception as e:
            logger.error(f"Erro no Treinamento: {e}")
            return False

    def run_reporting(self):
        """Etapa 4: Gera o relatório de Data Drift (Big Data Report).
        Compara a distribuição dos dados de treino (referência)
        contra os logs de inferência em produção (current).
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

    def run_pipeline(self):
        """
        Executa o pipeline completo (DAG sequencial).
        """
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
        else:
            logger.error("=== Pipeline finalizado com FALHA ===")


def schedule_pipeline(demo_mode: bool = False):
    orchestrator = MLPipelineOrchestrator()

    if demo_mode:
        # Modo demonstração: pipeline a cada 1 minuto, report a cada 2 minutos
        logger.info("Modo DEMO ativo: pipeline a cada 1 min | report a cada 2 min.")
        schedule.every(1).minutes.do(orchestrator.run_pipeline)
        schedule.every(2).minutes.do(orchestrator.run_reporting)
    else:
        # Modo produção: pipeline a cada 24h, report todo dia à meia-noite
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
        orchestrator.run_pipeline()
