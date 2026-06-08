""" """

from abc import ABC, abstractmethod

import pandas as pd


class AbstractAtlas(ABC):
    """
    Abstract base class for atlases
    """

    REQUIRED_COLUMNS = {"chr", "start", "end", "target", "name"}
    VALID_REFERENCE_GENOMES = {"hg38", "hg19"}
    VALID_CHROMOSOMES = {f"chr{i}" for i in range(1, 23)}.union({"chrX", "chrY"})

    @property
    @abstractmethod
    def reference_genome(self) -> str:
        """
        Return the reference genome associated with the atlas
        """

    @property
    @abstractmethod
    def atlas(self) -> pd.DataFrame:
        """
        Return the atlas as a pandas DataFrame.

        The atlas DataFrame should have at least the following columns:
        - 'chr': Chromosome name
        - 'start': Start position of the genomic region
        - 'end': End position of the genomic region
        - 'name': the name of the genomic region (e.g. "chr1:1000-2000")
        - 'target': the name of the group that is characteristic for the genomic region
        (e.g. single cell type, set of cell type, tissue, etc.)
        """

    @classmethod
    def _check_atlas_format(cls, candidate_atlas: pd.DataFrame) -> None:
        """
        Check if the atlas DataFrame has the required format.
        """
        cls._check_required_columns(candidate_atlas)
        cls._check_chromosomes(candidate_atlas)

    @classmethod
    def _check_required_columns(cls, candidate_atlas: pd.DataFrame) -> None:
        """Check if the atlas DataFrame has the required columns."""
        missing_columns = cls.REQUIRED_COLUMNS - set(candidate_atlas.columns)
        if missing_columns:
            raise ValueError(
                f"Atlas DataFrame is missing required columns: {missing_columns}"
            )

    @classmethod
    def _check_reference_genome(cls, reference_genome: str) -> None:
        """Check if the reference genome is valid."""
        if reference_genome not in cls.VALID_REFERENCE_GENOMES:
            raise ValueError(
                f"Invalid reference genome: {reference_genome}. "
                f"Valid options are: {cls.VALID_REFERENCE_GENOMES}"
            )

    @classmethod
    def _check_chromosomes(cls, candidate_atlas: pd.DataFrame) -> None:
        """Check if the chromosome names in the atlas DataFrame are valid."""
        invalid_chromosomes = set(candidate_atlas["chr"]) - cls.VALID_CHROMOSOMES
        if invalid_chromosomes:
            raise ValueError(
                f"Atlas DataFrame contains invalid chromosome names: {invalid_chromosomes}. "
                f"Valid options are: {cls.VALID_CHROMOSOMES}"
            )


class AbstractMethylationAtlas(AbstractAtlas):
    """
    Abstract base class for methylation atlases
    """

    REQUIRED_COLUMNS = AbstractAtlas.REQUIRED_COLUMNS.union({"startCpG", "endCpG"})

    @property
    @abstractmethod
    def atlas(self) -> pd.DataFrame:
        """
        Return the atlas as a pandas DataFrame.

        In addition to the columns specified in the AbstractAtlas,
        the methylation atlas DataFrame should also have the following columns:
        - "startCpG": position of the first CpG site in the genomic region
        - "endCpG": position of the last CpG site in the genomic region
        """
