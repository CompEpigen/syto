# Load required libraries
library(jsonlite)
library(dplyr)
library(purrr)
library(stringr)
library(glue)

# Directory containing your JSONs
json_dir <- "genes_metadata"
json_files <- list.files(json_dir, pattern = "\\.json$", full.names = TRUE)

# Canonical sequences 
canonical_sequences = data.table::fread("hg38.chromAlias.txt")
canonical_sequences %>% rename(
  Chromosome = V1,
  genomic_accession_version = `UCSC database: hg38`
) -> canonical_sequences 

chromosomes <- paste0("chr",c(c(1:22), "X", "Y")) # Canonical names
canonical_sequences$is_canonical <- canonical_sequences$Chromosome%in%chromosomes

# Function to extract locations from a single json
parse_gene_json <- function(json_path) {
  gene_data <- fromJSON(json_path)
  
  # Get gene symbol from filename or from data
  gene_symbol <- basename(json_path) %>%
    str_replace("\\.json$", "")
  
  # Flatten all annotations with their locations
  annots <- gene_data$reports$gene$annotations
  annots[sapply(annots, is.null)] <- NULL
  annots <- purrr::map(annots, ~({.x[stringr::str_detect(.x$assembly_name, "GRCh38"),]}))
  genomic_locations <- map(annots, ~({.x$genomic_locations}))
  extracted_locations <- purrr::map_df(
    genomic_locations, 
    ~{
      if(!is_empty(.x)){
       y <- purrr::map_df(.x, ~({.x$genomic_range}))
       # dat = .x$genomic_range 
       genomic_accession_version <- purrr::map(.x, ~({
          .x$genomic_accession_version
         }))
       genomic_accession_version[sapply(genomic_accession_version, is.null)] <- NULL
       y$genomic_accession_version <- sapply(genomic_accession_version, rbind)
       y <- y %>% left_join(canonical_sequences, by=c("genomic_accession_version"))
       y
      }
      })
  extracted_locations$gene <- gene_symbol
  return(extracted_locations)
}


# Parse all jsons and combine
all_locations <- map_dfr(json_files, parse_gene_json)
all_locations$`Hugo Symbol` <- all_locations$gene
all_locations$gene <- NULL
all_locations <- na.omit(all_locations)

perc_canonical = round(sum(na.omit(all_locations$is_canonical))/nrow(all_locations)*100,1)
print(glue('Percentage of canonical gene locations in database {perc_canonical}',
     perc_canonical = perc_canonical))

all_locations <- all_locations[all_locations$is_canonical,]

# Load annotated gene list 
gene_list <- data.table::fread("oncokb.org.cancerGeneList.tsv")
print(glue('Oncokb DB genes canonical locations coverage {cov}',
           cov = round(length(unique(all_locations$`Hugo Symbol`))/nrow(gene_list)*100,1)))


gene_list <- gene_list %>% left_join(all_locations, 'Hugo Symbol')
gene_list$begin <- gene_list$begin %>% as.numeric()
gene_list$end <- gene_list$end %>% as.numeric()
gene_list <- na.omit(gene_list)

data.table::fwrite(gene_list, "oncokb_with_locations.csv")
