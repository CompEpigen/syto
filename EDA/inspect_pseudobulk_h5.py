"""Report splits / matrix shapes / prediction-column counts of a pseudobulk HDF5."""

import sys

import h5py


def _cols(attrs, key):
    return [c.decode() if isinstance(c, bytes) else str(c) for c in attrs[key]]


def inspect(path):
    print("=" * 70)
    print(path)
    with h5py.File(path, "r") as f:
        if "outputs" not in f:
            print("  no 'outputs' group")
            return
        print("  splits:", list(f["outputs"].keys()))
        for split in f["outputs"]:
            g = f[f"outputs/{split}"]
            print(f"  --- {split} ---")

            if "pure_profiles" in g:
                pp = g["pure_profiles"]
                cols = _cols(pp.attrs, "numeric_columns")
                npred = sum(c.startswith("prediction_") for c in cols)
                print(
                    f"    pure feature_matrices {pp['feature_matrices'].shape}"
                    f" | uniform_prior {pp['uniform_prior'].shape}"
                    f" | {npred} prediction cols | last 3: {cols[-3:]}"
                )

            if "pseudobulks" in g:
                keys = list(g["pseudobulks"].keys())
                pb = g["pseudobulks"][keys[0]]
                cols = _cols(pb.attrs, "feature_columns")
                npred = sum(c.startswith("prediction_") for c in cols)
                print(
                    f"    {len(keys)} pseudobulks"
                    f" | aggregated_features {pb['aggregated_features'].shape}"
                    f" | {npred} prediction cols | last 3: {cols[-3:]}"
                )


if __name__ == "__main__":
    for p in sys.argv[1:]:
        inspect(p)
