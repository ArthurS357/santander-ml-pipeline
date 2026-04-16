import os
import logging
from pathlib import Path
from typing import Union
from src.config import use_dask_mode

logger = logging.getLogger(__name__)

_COLUNAS_ZEROS = ["plas", "pres", "skin", "test", "mass"]


def preprocess_data(
    input_path: Union[str, Path], output_path: Union[str, Path]
) -> None:
    """
    Lê os dados brutos, marca valores nulos clinicamente inválidos como NaN e salva.
    Usa Dask para arquivos > 500 MB ou quando USE_DASK=true; Pandas caso contrário.
    A imputação estatística é delegada ao pipeline de treino para evitar data leakage.
    """
    input_p = Path(input_path)
    output_p = Path(output_path)
    big = use_dask_mode(str(input_p))

    logger.info(
        f"Iniciando pré-processamento ({'Dask' if big else 'Pandas'}). "
        f"Lendo de: {input_p}"
    )

    try:
        if big:
            import dask.dataframe as dd

            df = dd.read_csv(input_p, assume_missing=True)
        else:
            import pandas as pd

            df = pd.read_csv(input_p)
    except FileNotFoundError:
        logger.error(f"Arquivo de entrada não encontrado: {input_p}")
        raise
    except Exception as e:
        logger.error(f"Falha ao ler o arquivo: {e}")
        raise

    colunas_presentes = [col for col in _COLUNAS_ZEROS if col in df.columns]
    if len(colunas_presentes) != len(_COLUNAS_ZEROS):
        logger.warning(
            "Algumas colunas esperadas para tratamento de zeros não foram encontradas no dataset."
        )

    df[colunas_presentes] = df[colunas_presentes].replace(0, float("nan"))

    logger.info(
        "Valores ausentes mapeados para NaN. Imputação delegada ao ML Pipeline."
    )

    try:
        output_p.parent.mkdir(parents=True, exist_ok=True)
        if big:
            # Dask: materializa em CSV (single_file evita sufixo de partição)
            df.to_csv(str(output_p), single_file=True, index=False)
        else:
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
