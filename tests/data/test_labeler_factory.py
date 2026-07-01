import numpy as np
import pandas as pd
import pytest

from syto.data.omics_signatures_handlers.binary_cpg_signature import (
    BinaryCpGSignatureHandler,
)
from syto.data.labelers.archetype_labeler import ArchetypeLabeler
from syto.data.labelers.hard_binary_labeler import HardBinaryLabeler
from syto.data.labelers.labeler_factory import (
    LabelerContext,
    labeler_factory,
    build_labeler,
    apply_labelers,
)


def _handler():
    return BinaryCpGSignatureHandler(
        start_column="read_start", methylation_pattern_column="methylation_ids"
    )


def _reads_df():
    rows = []
    for _ in range(4):
        rows.append(
            {
                "name": "r1",
                "read_start": 100,
                "methylation_ids": "01",
                "original_label": 0,
                "dmr_ctype_label": 0,
            }
        )
    for _ in range(4):
        rows.append(
            {
                "name": "r1",
                "read_start": 100,
                "methylation_ids": "11",
                "original_label": 1,
                "dmr_ctype_label": 0,
            }
        )
    return pd.DataFrame(rows)


def test_labeler_factory_constructs_known_type():
    assert isinstance(labeler_factory("hard_binary"), HardBinaryLabeler)


def test_labeler_factory_unknown_raises():
    with pytest.raises(ValueError):
        labeler_factory("nope")


def test_archetype_inv_global_freq_routes_to_predefined():
    ctx = LabelerContext(
        num_classes=3,
        signature_handler=_handler(),
        global_prior=np.array([0.2, 0.3, 0.5]),
    )
    labeler, kwargs, native = build_labeler(
        "archetype", {"ctype_prior_type": "inv_global_freq"}, ctx
    )
    assert isinstance(labeler, ArchetypeLabeler)
    assert kwargs["ctype_prior_type"] == "predefined"
    assert np.allclose(kwargs["predefined_prior"], [0.2, 0.3, 0.5])
    assert native == "soft_label"


def test_archetype_inv_global_freq_without_prior_raises():
    ctx = LabelerContext(num_classes=3, signature_handler=_handler())
    with pytest.raises(ValueError):
        build_labeler("archetype", {"ctype_prior_type": "inv_global_freq"}, ctx)


def test_unknown_type_raises():
    ctx = LabelerContext(num_classes=3, signature_handler=_handler())
    with pytest.raises(ValueError):
        apply_labelers(_reads_df(), {"dest": {"type": "nope"}}, ctx)


def test_two_soft_labelers_produce_distinct_columns():
    ctx = LabelerContext(num_classes=3, signature_handler=_handler())
    config = {
        "soft_pooled": {
            "type": "data_driven_soft",
            "perform_pooling": True,
            "min_reads": 2,
            "max_distance": 0.7,
        },
        "soft_raw": {"type": "data_driven_soft", "perform_pooling": False},
    }
    out = apply_labelers(_reads_df(), config, ctx)
    assert "soft_pooled" in out.columns
    assert "soft_raw" in out.columns
    assert "soft_label" not in out.columns


def test_hard_with_background_writes_destination_column():
    ctx = LabelerContext(num_classes=3, signature_handler=_handler())
    out = apply_labelers(
        _reads_df(), {"hard_bg": {"type": "hard_with_background"}}, ctx
    )
    assert "hard_bg" in out.columns
    assert "label" not in out.columns
    # label-1 reads are off-target (dmr=0) -> background class == num_classes
    off = out[out["original_label"] == 1]
    assert (off["hard_bg"] == 3).all()
