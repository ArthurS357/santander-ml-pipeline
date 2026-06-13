import os

# Limiar em bytes a partir do qual o modo Dask é ativado automaticamente (500 MB)
_DASK_SIZE_THRESHOLD = 500 * 1024 * 1024


def use_dask_mode(file_path: str | None = None) -> bool:
    """Retorna True quando o caminho "Big Data" deve ser ativado.

    O nome refere-se à camada de dados: ingestão e pré-processamento usam Dask
    de fato quando este chaveamento retorna True. No treino, porém, o mesmo
    chaveamento seleciona o fluxo incremental (`_train_incremental` em
    `train.py`), que usa `pandas.read_csv(chunksize=...)` + `SGDClassifier.
    partial_fit` — não Dask. Em resumo: Dask processa os dados; o treino é por
    chunks pandas + SGD incremental.

    Ativa o modo quando:
    - A variável de ambiente USE_DASK="true" estiver definida, OU
    - O arquivo informado tiver tamanho superior a 500 MB.
    """
    if os.getenv("USE_DASK", "").lower() == "true":
        return True

    if file_path and os.path.exists(file_path):
        if os.path.getsize(file_path) > _DASK_SIZE_THRESHOLD:
            return True

    return False
