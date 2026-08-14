"""Shared read-based baseline deconvolver sub-flow.

Both the ``inference`` and ``deconvolute_pseudobulk`` tasks configure the same
family of read-based baselines (uxm / celfieish / celfie / epidish) with the
same per-model fields, so the interactive sub-flow lives here instead of being
duplicated per task. The only difference is the answers key the resulting list
is stored under (``deconvolution.baselines`` for inference, ``baselines`` for
pseudobulk deconvolution).
"""

from App.wizard import validators as v
from App.wizard.fields import FieldSpec

BASELINES = ["uxm", "celfieish", "celfie", "epidish"]


def ask_baseline_entries(engine, answers, label: str) -> list[dict]:
    """Prompt for a set of baselines and return them as config entries."""
    selected = engine.ask_checkbox(label, BASELINES)
    baselines = []
    for model in selected:
        atlas = engine.ask_one(
            FieldSpec(
                key=f"_{model}_atlas",
                label=f"[{model}] atlas_path",
                kind="path",
                validate=v.path_exists,
            ),
            answers,
        )
        entry = {"model": model, "enabled": True, "atlas_path": atlas}
        if model == "uxm":
            ignore = engine.ask_one(
                FieldSpec(
                    key="_uxm_ignore",
                    label="[uxm] ignore_cells (comma-separated)",
                    kind="text",
                    default="Megakaryocytes",
                ),
                answers,
            )
            entry["ignore_cells"] = [c.strip() for c in ignore.split(",") if c.strip()]
        elif model == "epidish":
            entry["reference_genome"] = engine.ask_one(
                FieldSpec(
                    key="_epidish_ref",
                    label="[epidish] reference_genome",
                    kind="select",
                    choices=["hg38", "hg19"],
                    default="hg38",
                ),
                answers,
            )
            method = engine.ask_one(
                FieldSpec(
                    key="_epidish_method",
                    label="[epidish] method",
                    kind="select",
                    choices=["RPC", "CBS", "CP"],
                    default="RPC",
                ),
                answers,
            )
            entry["method"] = method
            if method == "RPC":
                entry["maxit"] = engine.ask_one(
                    FieldSpec(
                        key="_epidish_maxit",
                        label="[epidish] maxit (robust-regression iterations)",
                        kind="int",
                        default=50,
                        validate=v.positive_int,
                    ),
                    answers,
                )
            elif method == "CP":
                entry["constraint"] = engine.ask_one(
                    FieldSpec(
                        key="_epidish_constraint",
                        label="[epidish] constraint",
                        kind="select",
                        choices=["inequality", "equality"],
                        default="inequality",
                    ),
                    answers,
                )
            # Optional result-label override (blank -> auto: RPC='epidish',
            # CBS='epidish_cbs', CP='epidish_cp').
            name = engine.ask_one(
                FieldSpec(
                    key="_epidish_name",
                    label="[epidish] result name (blank = auto by method)",
                    kind="text",
                    default="",
                ),
                answers,
            )
            if name and name.strip():
                entry["name"] = name.strip()
        else:  # celfieish / celfie
            entry["reference_genome"] = engine.ask_one(
                FieldSpec(
                    key=f"_{model}_ref",
                    label=f"[{model}] reference_genome",
                    kind="select",
                    choices=["hg38", "hg19"],
                    default="hg38",
                ),
                answers,
            )
            ask_em_mode(engine, answers, model, entry)
            if model == "celfie":
                entry["random_restarts"] = engine.ask_one(
                    FieldSpec(
                        key="_celfie_rr",
                        label="[celfie] random_restarts",
                        kind="int",
                        default=1,
                        validate=v.positive_int,
                    ),
                    answers,
                )
                entry["sum_by_region"] = engine.ask_one(
                    FieldSpec(
                        key="_celfie_sum_by_region",
                        label="[celfie] sum_by_region (pool CpGs per region)",
                        kind="bool",
                        default=False,
                    ),
                    answers,
                )
                entry["freeze_gamma"] = engine.ask_one(
                    FieldSpec(
                        key="_celfie_freeze_gamma",
                        label="[celfie] freeze_gamma (hold atlas methylation)",
                        kind="bool",
                        default=False,
                    ),
                    answers,
                )
        baselines.append(entry)
    return baselines


def ask_em_mode(engine, answers, model, entry) -> None:
    """Ask whether a CelFiE(-ISH) baseline runs in convergence or checkpoint mode."""
    mode = engine.ask_one(
        FieldSpec(
            key=f"_{model}_emmode",
            label=f"[{model}] EM mode",
            kind="select",
            choices=["convergence", "em_checkpoints"],
            default="convergence",
        ),
        answers,
    )
    if mode == "convergence":
        entry["num_iterations"] = engine.ask_one(
            FieldSpec(
                key=f"_{model}_iter",
                label=f"[{model}] num_iterations",
                kind="int",
                default=400,
                validate=v.positive_int,
            ),
            answers,
        )
        entry["convergence_criteria"] = engine.ask_one(
            FieldSpec(
                key=f"_{model}_conv",
                label=f"[{model}] convergence_criteria",
                kind="float",
                default=0.001,
                validate=v.positive_float,
            ),
            answers,
        )
    else:
        checkpoints = engine.ask_one(
            FieldSpec(
                key=f"_{model}_ckpts",
                label=f"[{model}] em_checkpoints (comma-separated iteration counts)",
                kind="text",
                default="10, 50",
                validate=v.non_empty,
            ),
            answers,
        )
        entry["em_checkpoints"] = [
            int(c.strip()) for c in checkpoints.split(",") if c.strip()
        ]
