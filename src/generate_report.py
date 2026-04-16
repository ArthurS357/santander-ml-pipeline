"""
generate_report.py — Motor de Relatório de Big Data (Data Drift).

Geração de relatório de data drift desabilitada temporariamente nesta branch
por conta da incompatibilidade da biblioteca Evidently com ambiente
estrito de Python 3.14 offline.
"""

import logging

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("BigDataReport")


def generate_data_drift_report() -> str | None:
    """
    Geração de relatório de data drift desabilitada (incompatível com Python 3.14).
    """
    logger.warning("Data drift report desabilitado — Evidently incompatível com Python 3.14 offline.")
    return None


# ---------------------------------------------------------------------------
# Execução direta (CLI)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = generate_data_drift_report()
    if result:
        print(f"\n✅ Relatório gerado com sucesso: {result}")
    else:
        print("\n❌ Falha ao gerar relatório / Report desabilitado.")
