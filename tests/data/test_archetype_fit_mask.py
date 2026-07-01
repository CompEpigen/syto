import unittest

import numpy as np
import pandas as pd

from syto.data.labelers.archetype_labeler import ArchetypeLabeler
from syto.data.omics_signatures_handlers.binary_cpg_signature import (
    BinaryCpGSignatureHandler,
)

NUM_CLASSES = 3


def _make_labeler():
    handler = BinaryCpGSignatureHandler(
        start_column="trimmed_start", methylation_pattern_column="pattern"
    )
    return ArchetypeLabeler(signature_handler=handler)


class TestArchetypeFitMask(unittest.TestCase):
    def test_unseen_signature_still_scored_from_region_model(self):
        sig_a = ((100, 1), (101, 1))  # fit, label 0
        sig_b = ((100, 0), (101, 0))  # fit, label 1
        sig_val = ((100, 1), (101, 1), (102, 0))  # val only; 102 unseen in fit
        rows = [{"name": "R", "sig": s, "original_label": 0} for s in [sig_a] * 20]
        rows += [{"name": "R", "sig": s, "original_label": 1} for s in [sig_b] * 20]
        rows += [{"name": "R", "sig": sig_val, "original_label": 0}]
        df = pd.DataFrame(rows)
        fit_mask = np.array([True] * 40 + [False])
        res = _make_labeler().compute_labels(
            df,
            num_classes=NUM_CLASSES,
            precomputed_signature_column="sig",
            fit_mask=fit_mask,
        )
        self.assertEqual(len(res), len(df))
        val_row = res[res["sig"] == sig_val].iloc[0]
        # unseen 102 is dropped; the (100,1),(101,1) evidence favors class 0
        self.assertEqual(int(np.argmax(val_row["soft_label"])), 0)

    def test_region_without_fit_reads_gets_fallback(self):
        sig = ((100, 1),)
        df = pd.DataFrame([{"name": "Ronly_val", "sig": sig, "original_label": 0}])
        fit_mask = np.array([False])
        fallback = np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
        res = _make_labeler().compute_labels(
            df,
            num_classes=NUM_CLASSES,
            precomputed_signature_column="sig",
            fit_mask=fit_mask,
            fallback_label=fallback,
        )
        np.testing.assert_allclose(res.iloc[0]["soft_label"], fallback)
