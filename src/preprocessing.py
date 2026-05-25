import logging
import os
from pathlib import Path
from typing import Union

from src.config import use_dask_mode

logger = logging.getLogger(__name__)

_COLUNAS_ZEROS = ["plas", "pres", "skin", "test", "mass"]


def _columns_to_clean(present_columns: list[str]) -> list[str]:
    """Retorna o subconjunto de `_COLUNAS_ZEROS` presentes no DataFrame.

    Loga um warning quando alguma coluna esperada está ausente — comum em
    schemas reduzidos ou após renomeação upstream.
    """
    found = [col for col in _COLUNAS_ZEROS if col in present_columns]
    if len(found) != len(_COLUNAS_ZEROS):
        missing = sorted(set(_COLUNAS_ZEROS) - set(found))
        logger.warning(
            f"Colunas esperadas para tratamento de zeros ausentes: {missing}"
        )
    return found


def _preprocess_pandas(input_p: Path, output_p: Path) -> None:
    """Caminho Pandas — datasets in-memory."""
    import pandas as pd

    df = pd.read_csv(input_p)
    cols = _columns_to_clean(list(df.columns))
    df[cols] = df[cols].replace(0, float("nan"))
    logger.info(
        "Valores ausentes mapeados para NaN. Imputação delegada ao ML Pipeline."
    )
    df.to_csv(output_p, index=False)


def _preprocess_dask(input_p: Path, output_p: Path) -> None:
    """Caminho Dask — datasets que não cabem em memória.

    `single_file=True` é específico do `dask.dataframe.to_csv` e evita o
    sufixo de partição (`*.part`) gerado pelo modo distribuído padrão.
    """
    import dask.dataframe as dd

    df = dd.read_csv(input_p, assume_missing=True)
    cols = _columns_to_clean(list(df.columns))
    df[cols] = df[cols].replace(0, float("nan"))
    logger.info(
        "Valores ausentes mapeados para NaN. Imputação delegada ao ML Pipeline."
    )
    df.to_csv(str(output_p), single_file=True, index=False)


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
        output_p.parent.mkdir(parents=True, exist_ok=True)
        if big:
            _preprocess_dask(input_p, output_p)
        else:
            _preprocess_pandas(input_p, output_p)
        logger.info(f"Dados processados salvos com sucesso em: {output_p}")
    except FileNotFoundError:
        logger.error(f"Arquivo de entrada não encontrado: {input_p}")
        raise
    except PermissionError:
        logger.error(f"Permissão negada ao tentar salvar o arquivo em: {output_p}")
        raise
    except Exception as e:
        logger.error(f"Falha no pré-processamento: {e}")
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
