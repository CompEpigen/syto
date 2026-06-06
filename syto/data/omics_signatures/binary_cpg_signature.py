"""
Module for generating binary CpG signatures from methylation data.
"""

from typing import Tuple, Literal

from syto.data.omics_signatures.abstract_omics_signature_generator import (
    AbstractOmicsSignatureGenerator,
)

type MethylationState = Literal[0, 1]  # 0 for unmethylated, 1 for methylated
# A tuple of (position, state) pairs
# pylint: disable-next:invalid-name
type BinaryCpGSignature = Tuple[Tuple[int, MethylationState], ...]


def is_valid_binary_cpg_signature(signature) -> bool:
    pass  # TODO: this is starting to get really complicated, I think we did not tackle it with the correct approach


class BinaryCpGSignatureGenerator(AbstractOmicsSignatureGenerator):
    """
    Generator for binary CpG signatures.

    Binary means that each position is either methylated (1) or unmethylated (0),
    and we disregard any positions that are not clearly defined as either state.
    """

    @classmethod
    def extract_signature(cls, start: int, pattern: str) -> BinaryCpGSignature:
        """
        Extract a binary CpG signature from the given start position and pattern string.

        The pattern string consists of characters where '0' represents unmethylated,
        '1' represents methylated, and any other character is ignored.
        """
        return tuple(
            (start + i, int(val)) for i, val in enumerate(pattern) if val in ("0", "1")
        )
