library(data.table)
library(dplyr)

modkit_path = "E:\\methyldldata\\Curated\\melanoma_modkit_bed_rrms_2022.07\\normal\\COLO829BL_1.bed"
input_path = "E:\\methyldldata\\Curated\\melanoma_cpg_stats_rrms_2022.07\\normal\\COLO829BL_1_dss_input.txt"

original <- fread(input_path, nrows =100)

col_names <- c("chr", "start_position", "end_position", "modified_base_code_and_motif", "score",
               "strand", "start_postion2", "end_position2", "color","N_valid_cov",
               "percent_modified", "N_mod", "N_canonical","N_other_mod", "N_delete", 
               "N_fail", "N_diff", "N_nocall")
modkit <- fread(modkit_path, nrows =10000, header = FALSE, select = c(1,2,10,12)) 
colnames(modkit) <- col_names[c(1,2,10,12)]


colnames(original)