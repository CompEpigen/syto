"""Deconvolution results visualization: CLI summary + PDF report."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from rich.console import Console

from .deconvolution_data import BASELINE_DECONVOLVERS, build_report_model
from .deconvolution_cli import render_cli
from .deconvolution_pdf import render_pdf

__all__ = ["render_deconvolution_report"]


def render_deconvolution_report(
    results_df: pd.DataFrame,
    out_dir,
    cfg: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
    console: Optional[Console] = None,
) -> Path:
    """Build the ReportModel and emit both a CLI summary and a PDF report.

    Parameters
    ----------
    results_df : long-format deconvolution results
        (columns: FileName, CellType, Classifier, Deconvolver, Calibrator, PredictedProportion)
    out_dir : directory the PDF is written to (as ``deconvolution_report.pdf``)
    cfg : optional visualization config (top_n, known_truth, baseline_deconvolvers)
    """
    cfg = cfg or {}
    logger = logger or logging.getLogger(__name__)

    model = build_report_model(
        results_df,
        top_n=int(cfg.get("top_n", 10)),
        known_truth=cfg.get("known_truth"),
        baseline_deconvolvers=cfg.get("baseline_deconvolvers", BASELINE_DECONVOLVERS),
    )

    render_cli(model, console=console)

    pdf_path = Path(out_dir) / "deconvolution_report.pdf"
    render_pdf(model, pdf_path)
    logger.info(f"Saved deconvolution report to {pdf_path}")
    return pdf_path
