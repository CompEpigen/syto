"""
Module for generating binary CpG signatures from methylation data.
"""

from typing import Tuple, Literal

import pandas as pd

from syto.data.omics_signatures_handlers.abstract_omics_signature import (
    AbstractOmicsSignatureHandler,
)

type MethylationState = Literal[0, 1]  # 0 for unmethylated, 1 for methylated
# The binary cpg signature is a tuple of (position, state) pairs
# pylint: disable-next=invalid-name
type BinaryCpGSignature = Tuple[Tuple[int, MethylationState], ...]


class BinaryCpGSignatureHandler(AbstractOmicsSignatureHandler):
    """
    Handler for binary CpG signatures.

    Binary means that each position is either methylated (1) or unmethylated (0),
    and we disregard any positions that are not clearly defined as either state.
    """

    VALID_DISTANCE_NAMES = super().VALID_DISTANCE_NAMES.union({"jaccard"})

    @classmethod
    def extract_signature(cls, read_data: pd.Series) -> BinaryCpGSignature:
        """
        Extract a binary CpG signature from the given read data.

        The read data is expected to be a pandas Series with a 'start' position
        and a 'pattern' string.
        The pattern string consists of characters where '0' represents unmethylated,
        '1' represents methylated, and any other character is ignored.
        """
        start = read_data["start"]
        pattern = read_data["pattern"]
        return tuple(
            (start + i, int(val)) for i, val in enumerate(pattern) if val in ("0", "1")
        )

    @classmethod
    def compute_distance_dispatcher(cls, signature1, signature2, distance_name: str):
        match distance_name:
            case "jaccard":
                return cls.compute_jaccard_distance(signature1, signature2)
            case _:
                raise ValueError(
                    f"Unsupported distance metric: {distance_name}"
                    f"Supported metrics: {cls.VALID_DISTANCE_NAMES}"
                )

    @classmethod
    def compute_jaccard_distance(
        cls, sig1: BinaryCpGSignature, sig2: BinaryCpGSignature
    ) -> float:
        """
        Compute the Jaccard distance between two binary CpG signatures.

        The Jaccard distance is defined as 1 - (size of intersection / size of union).
        """
        set1 = set(sig1)
        set2 = set(sig2)
        intersection_size = len(set1.intersection(set2))
        union_size = len(set1.union(set2))
        if union_size == 0:
            return 0.0  # If both signatures are empty, they are identical
        return 1 - (intersection_size / union_size)
