"""Application layer: the ``syto`` CLI and the pipelines it drives.

Each pipeline module here turns one validated YAML config into a run
(:mod:`syto.app.inference`, :mod:`syto.app.pseudobulk_pipeline`, ...).
:mod:`syto.app.cli` is the ``syto`` console-script entry point, and
:mod:`syto.app.wizard` generates configs interactively.
"""
