import os
import time
import schedule
import logging
from src.data_ingestion import load_and_save_data
from src.preprocessing import preprocess_data
from src.train import train_model

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLPipelineOrchestrator:
    """
    Orquestrador de Pipeline de ML (Simulação de DAG).
    Fluxo: Ingestão -> Pré-processamento -> Treinamento -> Monitoramento.
    """
    
    def __init__(self):
        self.raw_data_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        self.raw_data_path = "data/raw/pima_diabetes.csv"
        self.processed_data_path = "data/processed/pima_diabetes_processed.csv"

    def run_ingestion(self):
        logger.info("Etapa 1: Iniciando Ingestão de Dados...")
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
            logger.info(f"=== Pipeline finalizado com SUCESSO em {time.time() - start_time:.2f}s ===")
        else:
            logger.error("=== Pipeline finalizado com FALHA ===")

def schedule_pipeline(demo_mode: bool = False):
    orchestrator = MLPipelineOrchestrator()

    if demo_mode:
        # Modo demonstração: executa imediatamente e agenda repetição a cada 1 minuto
        logger.info("Modo DEMO ativo: pipeline agendada para cada 1 minuto.")
        schedule.every(1).minutes.do(orchestrator.run_pipeline)
    else:
        # Modo produção: executa a cada 24 horas
        logger.info("Modo PRODUÇÃO: pipeline agendada para cada 24 horas.")
        schedule.every(24).hours.do(orchestrator.run_pipeline)

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
