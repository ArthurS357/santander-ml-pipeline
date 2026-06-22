"""generate_report.py — Data Drift Report próprio (sem Evidently).

Implementação leve de monitoramento de drift baseada em PSI (Population
Stability Index), comparando a distribuição do dataset de referência (treino)
contra os logs de inferência em produção. Não depende de Evidently nem de
bibliotecas pesadas — apenas numpy/pandas, já no projeto.

Saída: `reports/drift_report_YYYYMMDD_HHMMSS.json` (+ `.md`).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("BigDataReport")

# Features clínicas numéricas monitoradas (dataset Pima).
FEATURE_COLUMNS = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age"]

# [TALK & SHOW: Slide 7] Limiares de PSI: 0.2 (moderado) / 0.25 (alto)
# Limiares de PSI (convenção de mercado).
PSI_MODERATE = 0.2  # mudança moderada — investigar
PSI_HIGH = 0.25  # mudança significativa — ação recomendada

# [TALK & SHOW: Slide 7] Mínimo de inferências p/ report confiável (evita estatística fraca)
# Mínimo de inferências para um relatório estatisticamente útil.
_MIN_CURRENT_ROWS = 10

_DEFAULT_REFERENCE = "data/processed/pima_diabetes_processed.csv"
_DEFAULT_CURRENT = "data/logs/inference_logs.csv"


# [TALK & SHOW: Slide 7] PSI: mede drift por feature (distribuição treino × produção)
def calculate_psi(
    expected: np.typing.ArrayLike,
    actual: np.typing.ArrayLike,
    buckets: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Calcula o Population Stability Index entre duas distribuições.

    Os bins são definidos por quantis da distribuição esperada (referência).
    `epsilon` evita divisão por zero / log(0) em bins vazios. Distribuições
    idênticas retornam PSI ≈ 0; quanto maior, maior o drift.

    Args:
        expected: amostra de referência (treino).
        actual: amostra atual (produção).
        buckets: número de faixas (default 10).
        epsilon: piso de probabilidade por bin.

    Returns:
        Valor de PSI (>= 0).
    """
    expected_arr = np.asarray(expected, dtype=float)
    actual_arr = np.asarray(actual, dtype=float)
    expected_arr = expected_arr[~np.isnan(expected_arr)]
    actual_arr = actual_arr[~np.isnan(actual_arr)]

    if expected_arr.size == 0 or actual_arr.size == 0:
        return 0.0

    # Bordas por quantil da referência; bins degenerados (constante) → PSI 0.
    quantiles = np.linspace(0, 100, buckets + 1)
    edges = np.unique(np.percentile(expected_arr, quantiles))
    if edges.size < 2:
        return 0.0

    # Estende as bordas ao infinito para capturar valores fora do range de treino.
    edges[0], edges[-1] = -np.inf, np.inf

    expected_counts, _ = np.histogram(expected_arr, bins=edges)
    actual_counts, _ = np.histogram(actual_arr, bins=edges)

    expected_pct = np.clip(expected_counts / expected_counts.sum(), epsilon, None)
    actual_pct = np.clip(actual_counts / actual_counts.sum(), epsilon, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def _drift_level(psi: float) -> str:
    """Classifica o PSI em none/moderate/high."""
    if psi > PSI_HIGH:
        return "high"
    if psi > PSI_MODERATE:
        return "moderate"
    return "none"


def _build_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, object]:
    """Monta o dicionário do relatório de drift a partir dos dois DataFrames."""
    features = [
        c for c in FEATURE_COLUMNS if c in reference.columns and c in current.columns
    ]

    feature_report: dict[str, dict[str, object]] = {}
    alerts: list[str] = []
    for feature in features:
        psi = calculate_psi(reference[feature], current[feature])
        missing_rate = float(current[feature].isna().mean())
        level = _drift_level(psi)
        feature_report[feature] = {
            "psi": round(psi, 6),
            "missing_rate": round(missing_rate, 6),
            "drift_level": level,
        }
        if level != "none":
            alerts.append(f"{feature}: PSI={psi:.4f} ({level})")

    prediction: dict[str, float | None] = {
        "positive_rate": None,
        "confidence_mean": None,
        "confidence_std": None,
    }
    if "prediction" in current.columns:
        preds = pd.to_numeric(current["prediction"], errors="coerce").dropna()
        if not preds.empty:
            prediction["positive_rate"] = round(float((preds == 1).mean()), 6)
    if "probability" in current.columns:
        prob = pd.to_numeric(current["probability"], errors="coerce").dropna()
        if not prob.empty:
            prediction["confidence_mean"] = round(float(prob.mean()), 6)
            prediction["confidence_std"] = round(float(prob.std(ddof=0)), 6)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "thresholds": {"psi_moderate": PSI_MODERATE, "psi_high": PSI_HIGH},
        "features": feature_report,
        "prediction": prediction,
        "alerts": alerts,
        "drift_detected": bool(alerts),
    }


def _render_markdown(report: dict[str, object]) -> str:
    """Renderiza o relatório como Markdown legível."""
    features = report["features"]
    pred = report["prediction"]
    assert isinstance(features, dict) and isinstance(pred, dict)

    lines = [
        "# 📊 Data Drift Report — Santander ML Pipeline",
        "",
        f"- **Gerado em:** {report['generated_at']}",
        f"- **Linhas (referência / atual):** {report['reference_rows']} / {report['current_rows']}",
        f"- **Drift detectado:** {'⚠️ SIM' if report['drift_detected'] else '✅ não'}",
        "",
        "## PSI por feature",
        "",
        "| Feature | PSI | Missing | Nível |",
        "|---------|-----|---------|-------|",
    ]
    icon = {"none": "🟢", "moderate": "🟡", "high": "🔴"}
    for feature, stats in features.items():
        level = str(stats["drift_level"])
        lines.append(
            f"| {feature} | {stats['psi']:.4f} | {stats['missing_rate']:.2%} "
            f"| {icon.get(level, '')} {level} |"
        )

    lines += [
        "",
        "## Predições (produção)",
        "",
        f"- **Taxa de positivos:** {pred['positive_rate']}",
        f"- **Confiança média:** {pred['confidence_mean']}",
        f"- **Confiança (desvio padrão):** {pred['confidence_std']}",
        "",
        "## Alertas",
        "",
    ]
    alerts = report["alerts"]
    assert isinstance(alerts, list)
    if alerts:
        lines += [f"- ⚠️ {alert}" for alert in alerts]
    else:
        lines.append("- ✅ Nenhum drift acima do limiar.")
    lines.append("")
    return "\n".join(lines)


# [TALK & SHOW: Slide 7] Gera o Data Drift Report (JSON + Markdown) — PSI próprio, sem Evidently
def generate_data_drift_report(
    reference_path: str | Path | None = None,
    current_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> str | None:
    """Gera o relatório de Data Drift (PSI) em JSON + Markdown.

    Compara o dataset de referência (treino) com os logs de inferência. Os
    caminhos têm fallback via env vars (`PROCESSED_DATA_FILE`,
    `INFERENCE_LOG_FILE`) — mantendo compatibilidade com a chamada zero-arg
    do `pipeline_manager`.

    Returns:
        Caminho do JSON gerado, ou None se faltarem dados suficientes.
    """
    reference_p = Path(
        reference_path or os.getenv("PROCESSED_DATA_FILE", _DEFAULT_REFERENCE)
    )
    current_p = Path(current_path or os.getenv("INFERENCE_LOG_FILE", _DEFAULT_CURRENT))
    output_d = Path(output_dir or os.getenv("DRIFT_REPORT_DIR", "reports"))

    if not reference_p.exists():
        logger.warning(
            f"Dataset de referência ausente: {reference_p}. Report abortado."
        )
        return None
    if not current_p.exists():
        logger.warning(f"Logs de inferência ausentes: {current_p}. Report abortado.")
        return None

    reference = pd.read_csv(reference_p)
    current = pd.read_csv(current_p)

    if len(current) < _MIN_CURRENT_ROWS:
        logger.warning(
            f"Apenas {len(current)} inferências (< {_MIN_CURRENT_ROWS}). "
            "Dados insuficientes para um report confiável."
        )
        return None

    report = _build_report(reference, current)

    output_d.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_d / f"drift_report_{stamp}.json"
    md_path = output_d / f"drift_report_{stamp}.md"

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md_path.write_text(_render_markdown(report), encoding="utf-8")

    logger.info(
        f"Data Drift Report gerado: {json_path} "
        f"(drift_detected={report['drift_detected']}, alertas={len(report['alerts'])})"  # type: ignore[arg-type]
    )
    return str(json_path)


# ---------------------------------------------------------------------------
# Execução direta (CLI)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = generate_data_drift_report()
    if result:
        logger.info(f"Relatório gerado com sucesso: {result}")
    else:
        logger.warning("Relatório não gerado — dados insuficientes ou ausentes.")
