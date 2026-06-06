from syto.data.signature_distances.abstract_signature_distance import (
    AbstractSignatureDistance,
)


class JaccardDistance(AbstractSignatureDistance):
    """
    Jaccard distance
    """

    @classmethod
    def compute_distance(cls, signature1, signature2) -> float:
        