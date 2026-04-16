import os
import pandas as pd
from src.config import use_dask_mode


def load_data(file_path: str) -> pd.DataFrame:
    """
    Carrega dados detectando o formato pela extensão (.csv, .xlsx, .xls, .parquet).
    Usa Dask para arquivos > 500 MB ou quando USE_DASK=true; Pandas caso contrário.
    Retorna sempre um pd.DataFrame (Dask é materializado via .compute()).
    """
    ext = os.path.splitext(file_path)[1].lower()
    colunas = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age", "class"]
    big = use_dask_mode(file_path)

    if ext in [".xlsx", ".xls"]:
        # Dask não suporta read_excel — Pandas é sempre usado para Excel
        return pd.read_excel(file_path, names=colunas)

    if ext == ".parquet":
        if big:
            import dask.dataframe as dd

            ddf = dd.read_parquet(file_path)
            if len(ddf.columns) == len(colunas):
                ddf.columns = colunas
            return ddf.compute()
        df = pd.read_parquet(file_path)
        if len(df.columns) == len(colunas):
            df.columns = colunas
        return df

    # CSV (padrão)
    if big:
        import dask.dataframe as dd

        ddf = dd.read_csv(file_path, names=colunas, header=0, assume_missing=True)
        return ddf.compute()

    return pd.read_csv(file_path, names=colunas)


def load_and_save_data(url: str, output_path: str):
    """
    Carrega dados em qualquer formato suportado e salva como CSV padronizado,
    mantendo compatibilidade com as etapas seguintes do pipeline.
    """
    df = load_data(url)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Dados salvos com sucesso em: {output_path}")
    print(f"Dimensões do dataset: {df.shape}")


if __name__ == "__main__":
    DATA_URL = os.getenv("RAW_DATA_URL", "data/raw/pima_diabetes.csv")
    OUTPUT_FILE = "data/raw/pima_diabetes.csv"
    load_and_save_data(DATA_URL, OUTPUT_FILE)
