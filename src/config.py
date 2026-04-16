import os

# Limiar em bytes a partir do qual o modo Dask é ativado automaticamente (500 MB)
_DASK_SIZE_THRESHOLD = 500 * 1024 * 1024


def use_dask_mode(file_path: str = None) -> bool:
    """
    Retorna True se o processamento deve usar Dask em vez de Pandas.

    Ativa o modo Dask quando:
    - A variável de ambiente USE_DASK="true" estiver definida, OU
    - O arquivo informado tiver tamanho superior a 500 MB.
    """
    if os.getenv("USE_DASK", "").lower() == "true":
        return True

    if file_path and os.path.exists(file_path):
        if os.path.getsize(file_path) > _DASK_SIZE_THRESHOLD:
            return True

    return False
