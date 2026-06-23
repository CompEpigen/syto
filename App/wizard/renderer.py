"""Preview, save, and report on a generated config."""

import os
from datetime import datetime

import yaml


def dump_yaml(config: dict) -> str:
    """Serialize to block-style YAML, preserving insertion order."""
    return yaml.safe_dump(config, sort_keys=False, default_flow_style=False)


def render_preview(config: dict) -> None:
    """Print a syntax-highlighted YAML preview."""
    from rich.console import Console
    from rich.syntax import Syntax

    console = Console()
    console.print(Syntax(dump_yaml(config), "yaml", theme="ansi_dark"))


def save_config(config: dict, path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(dump_yaml(config))


def default_save_path(task: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"App/config/rrbs/{task}_{stamp}.yaml"


def print_run_command(task: str, path: str) -> None:
    print(f"\nConfig saved to {path}")
    print("Run with:")
    print(f"  python App/main.py --task {task} --config {path}")
