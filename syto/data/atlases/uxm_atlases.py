""" """

from typing import Optional

import pandas as pd

from syto.data.atlases.abstract_atlas import AbstractMethylationAtlas


class UXMMethylationAtlas(AbstractMethylationAtlas):
    """
    Methylation atlases of the UXM paper.
    """

    REQUIRED_COLUMNS = AbstractMethylationAtlas.REQUIRED_COLUMNS.union({"direction"})
    VALID_DIRECTIONS = {"U", "M"}

    def __init__(
        self,
        atlas_name: str,
        reference_genome: str,
        atlas_path: Optional[str] = None,
        atlas_df: Optional[pd.DataFrame] = None,
        sep: str = "\t",
    ):
        """
        Initialize the UXM methylation atlas.
        """
        # check that exactly one of atlas_path and atlas_df is provided
        if atlas_path is not None and atlas_df is not None:
            raise ValueError("Only one of atlas_path or atlas_df should be provided.")
        elif atlas_path is not None:
            self._atlas = pd.read_csv(atlas_path, sep=sep)
        elif atlas_df is not None:
            self._atlas = atlas_df
        else:
            raise ValueError("Either atlas_path or atlas_df must be provided.")

        super().__init__()
        self.atlas_name = atlas_name
        self._reference_genome = reference_genome

        # check its format
        self._check_atlas_format(self._atlas)

    @classmethod
    def _check_direction(cls, candidate_atlas: pd.DataFrame) -> None:
        """
        Check if the "direction" column in the atlas DataFrame contains valid values.
        """
        invalid_directions = set(candidate_atlas["direction"]) - cls.VALID_DIRECTIONS
        if invalid_directions:
            raise ValueError(
                f"Atlas DataFrame contains invalid direction values: {invalid_directions}. "
                f"Valid options are: {cls.VALID_DIRECTIONS}"
            )

    @classmethod
    def _check_atlas_format(cls, candidate_atlas: pd.DataFrame) -> None:
        """Check if the atlas DataFrame has the required format."""
        super()._check_atlas_format(candidate_atlas)
        cls._check_direction(candidate_atlas)

    @property
    def reference_genome(self) -> str:
        """Return the reference genome associated with the atlas"""
        return self._reference_genome

    @property
    def atlas(self) -> pd.DataFrame:
        """
        Return the atlas as a pandas DataFrame.

        In addition to the columns specified in the AbstractMethylationAtlas,
        the atlas DataFrame should also have the following columns:
        - "direction": contains "U" if the values in the colums for the cell types
        represent the average ratio of hypomethylated reads in the genomic region
        for that cell type, and "M" if they represent the average ratio of hypermethylated reads
        in the genomic region for that cell type
        - a column for each cell type in the atlas, containing the average ratio
        of hypomethylated reads (if direction is "U") or hypermethylated reads
        (if direction is "M") in the genomic region for that cell type
        """
