"""Ingestão de dados brutos — detecta formato pela extensão e padroniza para CSV.

Suporta CSV, Excel (.xlsx/.xls) e Parquet. Em arquivos > 500 MB ou quando
`USE_DASK=true`, usa Dask para leitura distribuída e materializa em memória
no fim (PoC — produção real usaria storage distribuído).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from src.config import use_dask_mode

logger = logging.getLogger(__name__)

# Schema esperado para o dataset Pima Indians Diabetes.
# Centralizado aqui para evitar duplicação nos branches de cada formato.
_EXPECTED_COLUMNS: list[str] = [
    "preg",
    "plas",
    "pres",
    "skin",
    "test",
    "mass",
    "pedi",
    "age",
    "class",
]


def _read_csv(path: Path, big: bool) -> pd.DataFrame:
    """Lê CSV com Dask (>500 MB) ou Pandas, garantindo o schema padronizado."""
    if big:
        import dask.dataframe as dd

        ddf = dd.read_csv(
            str(path), names=_EXPECTED_COLUMNS, header=0, assume_missing=True
        )
        return ddf.compute()
    return pd.read_csv(path, names=_EXPECTED_COLUMNS, header=0)


def _read_parquet(path: Path, big: bool) -> pd.DataFrame:
    """Lê Parquet via Dask ou Pandas; renomeia colunas se o arity bater."""
    if big:
        import dask.dataframe as dd

        ddf = dd.read_parquet(str(path))
        if len(ddf.columns) == len(_EXPECTED_COLUMNS):
            ddf.columns = _EXPECTED_COLUMNS
        return ddf.compute()
    df = pd.read_parquet(path)
    if len(df.columns) == len(_EXPECTED_COLUMNS):
        df.columns = _EXPECTED_COLUMNS
    return df


def _read_excel(path: Path) -> pd.DataFrame:
    """Excel não tem suporte distribuído — sempre Pandas."""
    return pd.read_excel(path, names=_EXPECTED_COLUMNS)


def load_data(file_path: str | Path) -> pd.DataFrame:
    """Carrega dados detectando o formato pela extensão (.csv, .xlsx, .xls, .parquet).

    Retorna sempre um `pd.DataFrame` (Dask é materializado via `.compute()`).
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    big = use_dask_mode(str(path))

    if ext in {".xlsx", ".xls"}:
        return _read_excel(path)
    if ext == ".parquet":
        return _read_parquet(path, big)
    # CSV é o formato padrão / fallback
    return _read_csv(path, big)


def load_and_save_data(url: str | Path, output_path: str | Path) -> None:
    """Carrega dados em qualquer formato suportado e salva como CSV padronizado.

    Mantém compatibilidade com as etapas seguintes do pipeline (que esperam CSV).
    """
    output = Path(output_path)
    df = load_data(url)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    logger.info(f"Dados salvos com sucesso em: {output} — shape={df.shape}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    DATA_URL = os.getenv("RAW_DATA_URL", "data/raw/pima_diabetes.csv")
    OUTPUT_FILE = Path("data/raw/pima_diabetes.csv")
    load_and_save_data(DATA_URL, OUTPUT_FILE)
