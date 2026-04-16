import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Union

# Criação do logger encapsulado ao módulo (sem basicConfig global)
logger = logging.getLogger(__name__)


def preprocess_data(
    input_path: Union[str, Path], output_path: Union[str, Path]
) -> None:
    """
    Lê os dados brutos, marca valores nulos clinicamente inválidos como NaN e salva.
    A imputação estatística foi delegada ao pipeline de treino para evitar data leakage.
    """
    input_p = Path(input_path)
    output_p = Path(output_path)

    logger.info(f"Iniciando pré-processamento. Lendo dados brutos de: {input_p}")

    try:
        df = pd.read_csv(input_p)
    except FileNotFoundError:
        logger.error(f"Arquivo de entrada não encontrado: {input_p}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"O arquivo de entrada está vazio: {input_p}")
        raise

    # Tratamento de valores ausentes mascarados como zero
    colunas_com_zeros = ["plas", "pres", "skin", "test", "mass"]

    # Verifica se as colunas esperadas realmente existem no DataFrame para evitar KeyError
    colunas_presentes = [col for col in colunas_com_zeros if col in df.columns]
    if len(colunas_presentes) != len(colunas_com_zeros):
        logger.warning(
            "Algumas colunas esperadas para tratamento de zeros não foram encontradas no dataset."
        )

    df[colunas_presentes] = df[colunas_presentes].replace(0, np.nan)

    logger.info(
        "Valores ausentes mapeados para NaN. Imputação delegada ao ML Pipeline."
    )

    # Garantir que a pasta de destino exista e salvar o arquivo de forma segura
    try:
        output_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_p, index=False)
        logger.info(f"Dados processados salvos com sucesso em: {output_p}")
    except PermissionError:
        logger.error(f"Permissão negada ao tentar salvar o arquivo em: {output_p}")
        raise


if __name__ == "__main__":
    # Configuração de log restrita à execução do script como main
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Lê os caminhos de entrada e saída de variáveis de ambiente.
    # Em produção, permitem apontar para volumes partilhados ou cloud storage.
    _default_input = "data/raw/pima_diabetes.csv"
    _default_output = "data/processed/pima_diabetes_processed.csv"
    INPUT_FILE = Path(os.getenv("RAW_DATA_FILE", _default_input))
    OUTPUT_FILE = Path(os.getenv("PROCESSED_DATA_FILE", _default_output))

    preprocess_data(INPUT_FILE, OUTPUT_FILE)
