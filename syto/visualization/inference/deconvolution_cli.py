"""rich rendering of the deconvolution ReportModel to the terminal."""

from __future__ import annotations

from typing import Optional

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .deconvolution_data import ReportModel

_SPARK = "▁▂▃▄▅▆▇█"


def _sparkline(values: np.ndarray) -> str:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return ""
    vmax = np.nanmax(v)
    if not np.isfinite(vmax) or vmax <= 0:
        return _SPARK[0] * v.size
    idx = np.clip(np.round(v / vmax * (len(_SPARK) - 1)).astype(int), 0, len(_SPARK) - 1)
    return "".join(_SPARK[i] for i in idx)


def _bar(fraction: float, width: int = 16) -> str:
    fraction = max(0.0, min(1.0, float(fraction)))
    filled = int(round(fraction * width))
    return "█" * filled + " " * (width - filled)


def render_cli(model: ReportModel, console: Optional[Console] = None) -> None:
    console = console or Console()
    console.rule(f"[bold]Deconvolution consensus — {model.file_name}")

    # 1. consensus leaderboard (top-N cell types + optional "other" row)
    table = Table(show_edge=False, expand=False)
    table.add_column("Cell type", style="bold")
    table.add_column("Consensus", justify="right")
    table.add_column("")
    table.add_column("±spread", justify="right")
    table.add_column("methods")

    vmax = float(model.consensus.max()) if model.n_cell_types else 1.0
    for i in range(model.top_n):
        ct = model.cell_types[i]
        cons = model.consensus[i]
        spread = model.spread_max[i] - model.spread_min[i]
        spark = _sparkline(model.matrix[i, :])
        style = "gold1" if ct == model.known_truth else ""
        table.add_row(
            Text(ct, style=style),
            f"{cons * 100:5.1f}%",
            _bar(cons / vmax if vmax else 0.0),
            f"±{spread * 100:4.1f}",
            spark,
        )
    if model.top_n < model.n_cell_types:
        table.add_row(
            Text("other", style="dim"),
            f"{model.other_consensus * 100:5.1f}%",
            "",
            "",
            "",
        )
    console.print(table)

    # 2. method roster (baseline | syto), colored by group
    roster = Table(title="Methods", show_edge=False)
    roster.add_column("Method")
    roster.add_column("Group")
    roster.add_column("Dominant call")
    roster.add_column("Divergence", justify="right")
    for j, m in enumerate(model.methods):
        col = model.matrix[:, j]
        dom = model.cell_types[int(np.nanargmax(col))] if model.n_cell_types else "-"
        if m.is_baseline:
            group = Text("baseline", style="grey62")
            name = Text(m.key, style="grey62")
        else:
            group = Text("syto", style="cyan")
            name = Text(m.key, style="cyan")
        roster.add_row(name, group, dom, f"{m.divergence:.3f}")
    console.print(roster)

    # 3. truth check
    if model.known_truth is not None:
        if model.known_truth in model.cell_types:
            rank = model.cell_types.index(model.known_truth) + 1
            console.print(
                f"[gold1]Known truth[/] {model.known_truth}: "
                f"consensus rank {rank}/{model.n_cell_types}"
            )
        else:
            console.print(
                f"[red]Known truth {model.known_truth} not found among cell types[/]"
            )
