"""
generate_report.py — Motor de Relatório de Big Data (Data Drift).

Compara a distribuição estatística dos dados de Referência (treino)
contra os dados de Produção (logs de inferência) utilizando a biblioteca
Evidently. Gera um relatório HTML interativo salvo em reports/.

Inclui sistema de alertas: se o drift_share ultrapassar o threshold
configurável, um WARNING é emitido simulando o disparo de um Webhook.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Evidently — importação com guarda de compatibilidade
#
# evidently<=0.7.x usa pydantic.v1 internamente, que quebra no Python >=3.14.
# A guarda abaixo impede que o módulo inteiro falhe em ambientes incompatíveis,
# retornando None com log de erro em vez de crash.
# Solução definitiva: Python 3.11/3.12 em produção (ver Dockerfile).
# ---------------------------------------------------------------------------
_EVIDENTLY_AVAILABLE = False
try:
    from evidently.report import Report  # type: ignore[import-untyped]
    from evidently.metric_preset import DataDriftPreset  # type: ignore[import-untyped]

    _EVIDENTLY_AVAILABLE = True
except Exception as _evidently_import_error:
    # Falha em Python >=3.14: evidently usa pydantic.v1 que quebra nessa versão.
    # Em produção (Dockerfile python:3.11-slim) o import funciona normalmente.
    Report = None  # type: ignore[assignment,misc]
    DataDriftPreset = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("BigDataReport")

# ---------------------------------------------------------------------------
# Constantes / Defaults
# ---------------------------------------------------------------------------
DEFAULT_PROCESSED_DATA = "data/processed/pima_diabetes_processed.csv"
DEFAULT_INFERENCE_LOG = "data/logs/inference_logs.csv"
REPORTS_DIR = "reports"

# Threshold de drift: fração de colunas com drift detectado para disparar alerta
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.5"))

# Colunas de feature esperadas no dataset de treino (Pima Diabetes)
FEATURE_COLUMNS = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age"]


def _ensure_dir(path: str) -> None:
    """Cria o diretório (e pais) caso não exista."""
    Path(path).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Sistema de Alertas
# ---------------------------------------------------------------------------
def send_alert(drift_share: float, report_path: str) -> None:
    """Dispara um alerta (simulação de Webhook) quando o drift excede o threshold.

    Em produção, esta função seria substituída por uma chamada real a um
    Webhook (Slack, PagerDuty, Teams, etc.). Aqui simulamos via log WARNING
    com payload JSON estruturado para facilitar integração futura.

    Args:
        drift_share: Fração de colunas que apresentaram drift (0.0 – 1.0).
        report_path: Caminho do relatório HTML gerado.
    """
    payload = {
        "alert_type": "DATA_DRIFT",
        "severity": "HIGH" if drift_share >= 0.75 else "MEDIUM",
        "drift_share": round(drift_share, 4),
        "threshold": DRIFT_THRESHOLD,
        "report_path": report_path,
        "timestamp": datetime.now().isoformat(),
        "message": (
            f"⚠️ ALERTA DE DATA DRIFT: {drift_share:.0%} das features apresentam "
            f"drift significativo (threshold: {DRIFT_THRESHOLD:.0%}). "
            f"Relatório disponível em: {report_path}"
        ),
    }

    logger.warning(
        f"[WEBHOOK SIMULADO] Data Drift detectado acima do threshold! "
        f"drift_share={drift_share:.2%} > threshold={DRIFT_THRESHOLD:.2%}"
    )
    logger.warning(f"[WEBHOOK PAYLOAD] {json.dumps(payload, ensure_ascii=False)}")


def _extract_drift_share(report: Report) -> float | None:
    """Extrai o drift_share do resultado do relatório Evidently.

    O drift_share representa a fração de colunas que tiveram drift detectado
    (valor entre 0.0 e 1.0).

    Returns:
        drift_share como float, ou None se não for possível extrair.
    """
    try:
        report_dict = report.as_dict()

        # Evidently v0.4+ estrutura: metrics[0].result.drift_share
        for metric in report_dict.get("metrics", []):
            result = metric.get("result", {})
            if "drift_share" in result:
                return float(result["drift_share"])

        logger.debug("drift_share não encontrado na estrutura do relatório.")
        return None
    except Exception as e:
        logger.warning(f"Erro ao extrair drift_share: {e}")
        return None


def generate_data_drift_report() -> str | None:
    """
    Gera o relatório de Data Drift comparando a base de treino (referência)
    com os logs de inferência (produção/current).

    Após a geração, avalia o drift_share e dispara alerta se ultrapassar
    o threshold configurado.

    Returns:
        Caminho do relatório HTML gerado ou None em caso de falha.
    """

    # ------------------------------------------------------------------
    # 0. Pré-requisito: evidently deve estar disponível no runtime
    # ------------------------------------------------------------------
    if not _EVIDENTLY_AVAILABLE:
        logger.error(
            "evidently não está disponível neste runtime (Python %s). "
            "O relatório de Data Drift requer Python 3.11 ou 3.12. "
            "Consulte o Dockerfile para o ambiente de produção recomendado.",
            sys.version.split()[0],
        )
        return None

    # ------------------------------------------------------------------
    # 1. Carregar dados de Referência (treino)
    # ------------------------------------------------------------------
    ref_path = os.getenv("PROCESSED_DATA_FILE", DEFAULT_PROCESSED_DATA)
    if not Path(ref_path).exists():
        logger.error(f"Arquivo de referência não encontrado: {ref_path}")
        return None

    logger.info(f"Carregando dados de referência de: {ref_path}")
    df_reference = pd.read_csv(ref_path)

    # Garantir que usamos apenas as colunas de features
    available_ref_cols = [c for c in FEATURE_COLUMNS if c in df_reference.columns]
    if not available_ref_cols:
        logger.error("Nenhuma coluna de feature encontrada nos dados de referência.")
        return None
    df_reference = df_reference[available_ref_cols]

    # ------------------------------------------------------------------
    # 2. Carregar dados de Produção (logs de inferência)
    # ------------------------------------------------------------------
    log_path = os.getenv("INFERENCE_LOG_FILE", DEFAULT_INFERENCE_LOG)
    if not Path(log_path).exists():
        logger.warning(f"Arquivo de logs de inferência não encontrado: {log_path}")
        logger.warning("Nenhum dado de produção disponível para comparação.")
        return None

    logger.info(f"Carregando logs de inferência de: {log_path}")
    df_current = pd.read_csv(log_path)

    # Filtrar apenas as colunas de features presentes
    available_cur_cols = [c for c in FEATURE_COLUMNS if c in df_current.columns]
    if not available_cur_cols:
        logger.error("Nenhuma coluna de feature encontrada nos logs de inferência.")
        return None
    df_current = df_current[available_cur_cols]

    if len(df_current) < 2:
        logger.warning(
            f"Apenas {len(df_current)} registros nos logs. "
            "São necessários pelo menos 2 registros para o relatório."
        )
        return None

    # ------------------------------------------------------------------
    # 3. Gerar relatório Evidently
    # ------------------------------------------------------------------
    logger.info(
        f"Gerando relatório de Data Drift — "
        f"Referência: {len(df_reference)} linhas | "
        f"Produção: {len(df_current)} linhas"
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=df_reference, current_data=df_current)

    # ------------------------------------------------------------------
    # 4. Salvar HTML
    # ------------------------------------------------------------------
    _ensure_dir(REPORTS_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"data_drift_report_{timestamp}.html"
    report_path = os.path.join(REPORTS_DIR, report_filename)

    report.save_html(report_path)
    logger.info(f"Relatório de Data Drift salvo em: {report_path}")

    # ------------------------------------------------------------------
    # 5. Sistema de Alertas — verificar drift_share contra threshold
    # ------------------------------------------------------------------
    drift_share = _extract_drift_share(report)
    if drift_share is not None:
        logger.info(
            f"Drift share detectado: {drift_share:.2%} (threshold: {DRIFT_THRESHOLD:.2%})"
        )
        if drift_share > DRIFT_THRESHOLD:
            send_alert(drift_share=drift_share, report_path=report_path)
        else:
            logger.info(
                "✅ Drift dentro dos limites aceitáveis. Nenhum alerta disparado."
            )
    else:
        logger.warning(
            "Não foi possível extrair o drift_share — verificação de alertas ignorada."
        )

    return report_path


# ---------------------------------------------------------------------------
# Execução direta (CLI)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = generate_data_drift_report()
    if result:
        print(f"\n✅ Relatório gerado com sucesso: {result}")
    else:
        print("\n❌ Falha ao gerar relatório. Verifique os logs acima.")
