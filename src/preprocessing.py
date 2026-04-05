import pandas as pd
import numpy as np
import os

def preprocess_data(input_path: str, output_path: str):
    """
    Lê os dados brutos, trata valores ausentes (zeros) e salva os dados processados.
    """
    print(f"Lendo dados brutos de: {input_path}")
    df = pd.read_csv(input_path)
    
    # Colunas onde o valor 0 é na verdade um dado ausente (ex: Pressão Arterial 0 não existe)
    colunas_com_zeros = ['plas', 'pres', 'skin', 'test', 'mass']
    
    # Passo 1: Substituir 0 por NaN para facilitar o tratamento
    df[colunas_com_zeros] = df[colunas_com_zeros].replace(0, np.nan)
    
    # Passo 2: Preencher os valores NaN com a mediana de cada coluna
    for col in colunas_com_zeros:
        mediana = df[col].median()
        df[col] = df[col].fillna(mediana)
    
    print("Valores ausentes tratados com a mediana.")
    
    # Passo 3: Garantir que a pasta de destino exista e salvar o arquivo
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dados processados salvos em: {output_path}")

if __name__ == "__main__":
    INPUT_FILE = "data/raw/pima_diabetes.csv"
    OUTPUT_FILE = "data/processed/pima_diabetes_processed.csv"
    preprocess_data(INPUT_FILE, OUTPUT_FILE)
