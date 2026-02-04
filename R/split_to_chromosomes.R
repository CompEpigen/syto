library(DSS)
require(bsseq)
library(dplyr)
library(data.table)
library(progress)

# input_path = "E:\\methyldldata\\Curated\\melanoma_cpg_stats_rrms_2022.07\\"
input_path = "E:\\methyldldata\\Curated\\melanoma_modkit_bed_rrms_2022.07\\"

col_names <- c("chr", "pos", "N", "X")

for (prefix in c("normal\\COLO829BL_", "tumor\\COLO829_")) {
  for (i in 1:5) {
    if(prefix == "normal\\COLO829BL_"){
      label ="normal"
    } else{
      label ="tumor"
    }
    file_path <- file.path(input_path, paste0(prefix, i, ".bed"))
    # Load file
    temp_data <- fread(file_path, header = FALSE, select = c(1,2,10,12))
    colnames(temp_data) <- col_names
    chromosomes = intersect.Vector(unique(temp_data$chr),c(paste0("chr", c(1:22)), "chrX", "chrY"))
    pb <- progress_bar$new(format = "(:spin) [:bar] :percent [Elapsed time: :elapsedfull || Estimated time remaining: :eta]",
                           total = length(chromosomes),
                           complete = "=",   # Completion bar character
                           incomplete = "-", # Incomplete bar character
                           current = ">",    # Current bar character
                           clear = FALSE,    # If TRUE, clears the bar when finish
                           width = 100,
                           show_after =0)      # Width of the progress bar
    pb$message(paste0("Processing chromosomes for file ", file_path))
    pb$tick(0)
    for (chromosome_name in chromosomes){
      save_dir <- file.path("processed_data", "dss_input_modkit", chromosome_name)
      dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
      save_path <- file.path(save_dir, paste0(label,"_",i,".fst"))
      fst::write.fst(temp_data %>% filter(chr==chromosome_name),save_path)
      pb$tick()
    }
    pb$terminate()
  }
}