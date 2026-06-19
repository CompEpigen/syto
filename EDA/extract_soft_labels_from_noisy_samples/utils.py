import os
from collections import defaultdict

import json
import numpy as np

global df_ctype_match
f_cell_type_match = "./data/files_to_cell_groups.json"
with open(f_cell_type_match, "r") as fp:
    df_ctype_match = json.load(fp)


def filename2ctype(f_name: str) -> str:
    """

    Extract cell types from the file name
    It returns the cell type matching to the cell types written in the region information
    e.g.,) Blood-Monocytes -> Blood-Mono+Macro

    """

    f_name = os.path.basename(f_name)
    f_name_short = "_".join(f_name.split("_")[1:]).split(".hg38")[0]
    return df_ctype_match.get(f_name_short)

    # try:
    #     return df_ctype_match[f_name_short]
    # except KeyError:
    #     print(f_name_short)
    #     print(f"{f_name} cannot be matched to the 39 cell types. NA is assigned for the cell type")
    #     # print(f_name, "---->", file_ctype)
    #     return "NA"


def nested_defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: nested_defaultdict_to_dict(v) for k, v in d.items()}
    return d


# utilities to manipulate signatures as binary vectors and convert them to indices for easier handling
def get_idx_from_sig(sig):
    if isinstance(sig, str):
        sig = [int(bit) for bit in sig]
    idx = 0
    for i, bit in enumerate(sig):
        idx += bit * (2**i)
    return idx


def get_sig_from_idx(idx, sig_length, return_str=False):
    assert idx < 2**sig_length, "Index is too large for the given signature length"
    sig = []
    for i in range(sig_length):
        bit = (idx // (2**i)) % 2
        sig.append(bit)
    if return_str:
        return "".join(map(str, sig))
    return np.array(sig)
