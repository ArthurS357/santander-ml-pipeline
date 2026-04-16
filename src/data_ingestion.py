import pandas as pd
import os


def load_and_save_data(url: str, output_path: str):
    """
    Carrega os dados da URL e salva na pasta local do projeto.
    """
    # Nomes das colunas conforme o dataset Pima Indians
    colunas = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age", "class"]

    # Lendo os dados diretamente da internet
    df = pd.read_csv(url, names=colunas)

    # Garantindo que a pasta de destino exista
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Salvando os dados brutos localmente
    df.to_csv(output_path, index=False)
    print(f"Dados salvos com sucesso em: {output_path}")
    print(f"Dimensões do dataset: {df.shape}")


if __name__ == "__main__":
    DATA_URL = os.getenv("RAW_DATA_URL", "data/raw/pima_diabetes.csv")
    OUTPUT_FILE = "data/raw/pima_diabetes.csv"
    load_and_save_data(DATA_URL, OUTPUT_FILE)
