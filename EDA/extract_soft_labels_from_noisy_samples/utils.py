import os
import json

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
