from typing import Dict, List, Optional

import numpy as np


def rearange_deconvolution_results(
    labels_dict_reversed: Dict[str, int],
    proportions: np.ndarray,
    ref_cells: List[str],
    n_labels: Optional[int] = None,
) -> List[float]:
    """Reorder deconvolution proportions to match the project's label index order.

    Parameters
    ----------
    labels_dict_reversed : dict
        ``cell_type_name → label_index`` mapping.
    proportions : ndarray(T,)
        Proportions in atlas cell-type order (``ref_cells`` ordering).
    ref_cells : list of str
        Cell-type names corresponding to axes of ``proportions``.
    n_labels : int, optional
        Number of labels; defaults to ``len(labels_dict_reversed)``.

    Returns
    -------
    list of float
        Proportions reindexed to ``label_index`` order (0 … n_labels-1).
    """
    if n_labels is None:
        n_labels = len(labels_dict_reversed)
    ref_pos = np.array([labels_dict_reversed.get(cell, -1) for cell in ref_cells])
    return [proportions[int(np.where(ref_pos == i)[0][0])] for i in range(n_labels)]
