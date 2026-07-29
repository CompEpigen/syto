""" """

import numpy as np
import pandas as pd


def plan_splits(counts, *, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, seed=42):
    """Plan train/valid/test allocation from per-(label, file) read counts.

    Mirrors the reference split_by_file_and_class file-level logic: file-level
    isolation with an explicit coverage override so every class reaches all
    three splits. Single-file classes (and the smaller file of a 2-file class)
    are recorded for deterministic read-level splitting at apply time.
    """
    file_level, read_level = {}, {}
    grouped = counts.groupby("original_label")

    for label, group in grouped:
        fc = dict(zip(group["file"], group["n_reads"]))
        files_sorted = sorted(fc, key=lambda k: fc[k], reverse=True)
        n_files = len(files_sorted)
        total = sum(fc.values())

        if n_files >= 3:
            targets = {
                "train": total * train_ratio,
                "valid": total * valid_ratio,
                "test": total * test_ratio,
            }
            current = {"train": 0, "valid": 0, "test": 0}
            allocations = {"train": [], "valid": [], "test": []}
            for i, f in enumerate(files_sorted):
                remaining = n_files - i
                empty = [b for b in targets if not allocations[b]]
                if len(empty) >= remaining:
                    best = max(empty, key=lambda b: targets[b])
                else:
                    deficits = {k: targets[k] - current[k] for k in targets}
                    best = max(deficits, key=deficits.get)
                allocations[best].append(f)
                current[best] += fc[f]
            for split, files in allocations.items():
                for f in files:
                    file_level[(label, f)] = split
        elif n_files == 2:
            f1, f2 = files_sorted
            file_level[(label, f1)] = "train"
            read_level[(label, f2)] = {
                "seed": seed,
                "train_ratio": 0.0,
                "valid_ratio": valid_ratio / (valid_ratio + test_ratio),
            }
        else:
            read_level[(label, files_sorted[0])] = {
                "seed": seed,
                "train_ratio": train_ratio,
                "valid_ratio": valid_ratio
                / (valid_ratio + test_ratio)
                * (1 - train_ratio),
            }

    return {"file_level": file_level, "read_level": read_level}


def apply_splits(df, plan, *, remap=None):
    """Stamp a deterministic ``split`` column onto ``df`` from a plan.

    When ``remap`` is given, it is an old->new split-name mapping applied as the
    final step; unmapped values and NA pass through unchanged. The canonical
    partition from ``plan_splits`` is unaffected.
    """
    df = df.copy()
    df["split"] = pd.NA

    file_level = plan["file_level"]
    if file_level:
        keys = list(zip(df["original_label"], df["file"]))
        df["split"] = [file_level.get(k, pd.NA) for k in keys]

    for (label, file), cfg in plan["read_level"].items():
        mask = (df["original_label"] == label) & (df["file"] == file)
        idx = df.index[mask].to_numpy()
        rng = np.random.RandomState(cfg["seed"])
        order = rng.permutation(len(idx))
        n = len(idx)
        n_train = int(round(cfg["train_ratio"] * n))
        n_valid = int(round(cfg["valid_ratio"] * n))
        assign = np.array(["test"] * n, dtype=object)
        assign[order[:n_train]] = "train"
        assign[order[n_train : n_train + n_valid]] = "valid"
        df.loc[idx, "split"] = assign

    if remap:
        df["split"] = df["split"].map(lambda s: remap.get(s, s) if pd.notna(s) else s)

    return df
