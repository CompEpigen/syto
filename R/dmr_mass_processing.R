library(DSS)
require(bsseq)
library(dplyr)
library(data.table)
library(progress)
library(ggplot2)


parent_path <- "./processed_data/dml_modkit_test_results_non_smoothed"
dml_files <- list.files(parent_path)

dmr_p_tr = 0.0001
# imr_p_tr = 0.01

result = list()
for (target_file in dml_files){
  print(paste(Sys.time(), "Processing file", target_file))
  dml_res <- fst::read.fst(file.path(parent_path, target_file))
  dml_res_inverted <- dml_res
  dml_res_inverted$pval <- 1 - dml_res_inverted$pval
  dmrs = callDMR(dml_res, p.threshold=dmr_p_tr,minlen=500)
  # imrs = callDMR(dml_res_inverted, p.threshold=imr_p_tr,minlen=100)
  hypomethylated_dmrs = dmrs[dmrs$diff.Methy>0,]
  hypermethylated_dmrs = dmrs[dmrs$diff.Methy<0,]
  # imrs$type = "IMR"
  hypomethylated_dmrs$type = "DMR-Hypo"
  hypermethylated_dmrs$type = "DMR-Hyper"
  result[[target_file]] = plyr::ldply(list(hypomethylated_dmrs,hypermethylated_dmrs), fun = rbind) 
}

resultdf = plyr::ldply(result, fun = rbind)
dmrs = resultdf

# data.table::fwrite(resultdf, "computed_dmrs_non_smoothing_pdmr0.0001_minlen500.csv",row.names = FALSE)

#### Adding locations information ####

# dmrs = data.table::fread("computed_dmrs_25_smoothing_pdmr0.0001_pimr0.01_minlen500.csv")
print(paste("Percentage hypermethylated DMRs", round(mean(dmrs$type=="DMR-Hyper")*100,1)))
print(paste("Percentage hypomethylated DMRs", round(mean(dmrs$type=="DMR-Hypo")*100,1)))
# print(paste("Percentage IMRs", round(mean(dmrs$type=="IMR")*100,1)))

gene_list = data.table::fread('oncokb_with_locations.csv')

# Required packages
library(data.table)
library(GenomicRanges)
library(dplyr)

# If not installed:
# install.packages("BiocManager"); BiocManager::install("GenomicRanges")

# 1. Prepare DMRs as GRanges ----
dmrs <- as.data.table(dmrs)  # your DMR data.table
dmrs_gr <- GRanges(
  seqnames = gsub("^chr", "", dmrs$chr),  # remove "chr" if your gene locations are plain numbers, adjust if not
  ranges = IRanges(start=dmrs$start, end=dmrs$end)
)

# 2. Prepare gene locations as GRanges ----
gene_list <- as.data.table(gene_list)
gene_list <- gene_list %>% mutate(
  orientation = purrr::map_chr(gene_list$orientation,
                               ~({
                                 ifelse(.x=="plus", "+","-")
                               })
                               )
)


# Suppose your gene locations are in columns: chr, start, end
# If not, adjust accordingly. Let's assume you have already merged your locations
genes_gr <- GRanges(
  seqnames = gsub("^chr", "", gene_list$`# sequenceName`),  # adapt if your column is named differently
  ranges = IRanges(start=gene_list$begin, end=gene_list$end),
  Hugo_Symbol = gene_list$`Hugo Symbol`,
  strand = gene_list$orientation
)

# 3. Find overlaps ----
hits <- findOverlaps(dmrs_gr, genes_gr)
overlap_dt <- data.table(
  dmr_idx = queryHits(hits),
  gene_idx = subjectHits(hits)
)

# For each DMR, calculate overlap length and annotate gene
dmrs$Overlapping <- FALSE
dmrs$Gene <- NA_character_
dmrs$Overlap_bps <- 0L
dmrs$Closest_Dist <- NA_integer_

# If overlap, annotate
if (nrow(overlap_dt) > 0) {
  # Calculate overlap lengths
  overlap_dt[, overlap_start := pmax(start(dmrs_gr[dmr_idx]), start(genes_gr[gene_idx]))]
  overlap_dt[, overlap_end   := pmin(end(dmrs_gr[dmr_idx]), end(genes_gr[gene_idx]))]
  overlap_dt[, overlap_bps   := pmax(0, overlap_end - overlap_start + 1)]
  
  # Choose max overlap per DMR (in case of multiple genes)
  overlap_best <- overlap_dt[ , .SD[which.max(overlap_bps)], by=dmr_idx]
  
  dmrs[overlap_best$dmr_idx, `:=`(
    Overlapping = TRUE,
    Gene = mcols(genes_gr)$Hugo_Symbol[overlap_best$gene_idx],
    Overlap_bps = overlap_best$overlap_bps
  )]
}

# 4. If no overlap: find closest gene
no_overlap_idx <- which(!dmrs$Overlapping)

if (length(no_overlap_idx) > 0) {
  print(paste(Sys.time(), "Processing", length(no_overlap_idx), "non-overlapping DMRs..."))
  
  # Use GenomicRanges' optimized distance function
  dmr_no_ovl <- dmrs_gr[no_overlap_idx]
  
  # Find nearest TSS for each non-overlapping DMR
  nearest_hits <- GenomicRanges::distanceToNearest(dmr_no_ovl, genes_gr)
  
  # Extract results
  query_idx <- queryHits(nearest_hits)
  subject_idx <- subjectHits(nearest_hits)
  distances <- mcols(nearest_hits)$distance
  
  # Update DMRs with closest gene info
  dmrs[no_overlap_idx[query_idx], `:=`(
    Gene = mcols(genes_gr)$Hugo_Symbol[subject_idx],
    Closest_Dist = distances
  )]
}

print(paste(Sys.time(), "Analysis complete!"))

dmrs <- dmrs %>% mutate(
  Closest_Dist = ifelse(Overlapping, 0, Closest_Dist)
)

dmrs <- dmrs %>% left_join(gene_list %>% group_by(`Hugo Symbol`) %>%
                             summarise(
                               oncogene = first(`Is Oncogene`),
                               tumor_suppressor_gene = first(`Is Tumor Suppressor Gene`),
                               n_literature_sources = first(`# of occurrence within resources (Column J-P)`)
                             ),
                           by = join_by(Gene==`Hugo Symbol`))
dmrs <- dmrs %>% mutate(
  oncogene = ifelse(oncogene == "Yes",TRUE, FALSE),
  tumor_suppressor_gene = ifelse(tumor_suppressor_gene == "Yes", TRUE, FALSE)
)

print(dmrs %>% group_by(chr) %>% summarise(
  OverlapingDMRsProportion=mean(Overlapping),
  # DMRsWithin100bp = mean(Closest_Dist<=100),
  # DMRsWithin1000bp = mean(Closest_Dist<=1000),
  # DMRsWithin2500bp = mean(Closest_Dist<=2500),
  # DMRsWithin5000bp = mean(Closest_Dist<=5000),
  DMRsWithin10000bp = mean(Closest_Dist<=10000),
  DMRsWithin1000000bp = mean(Closest_Dist<=1000000),
  ),n=25)

dmrs$.id <- NULL
dmrs = dmrs[order(abs(dmrs$areaStat),decreasing=TRUE) ,]
dmrs$name = paste0("DMR_", seq(nrow(dmrs)))


data.table::fwrite(dmrs, "dmrs_with_merged_genes_non_smoothing_pdmr0.0001_minlen500.csv")

data.table::fwrite(dmrs[,c("chr", "start", "end", "name", "areaStat")], sep = "\t",col.names = FALSE,"dmrs_with_merged_genes_non_smoothing_pdmr0.0001_minlen500.bed")

dmrs <- data.table::fread("dmrs_with_merged_genes_25_smoothing_pdmr0.0001_pimr0.01_minlen500.csv")

library(ggplot2)

# # Hexagonal binning
# ggplot(dmrs[!dmrs$type == "IMR",], aes(x = diff.Methy, y = abs(areaStat))) +
#   geom_hex(bins = 50) +
#   scale_fill_viridis_c() +
#   facet_wrap(~type) +
#   labs(x ="Methylation levels difference", y = "Areastat")

# Marginal plot (requires ggExtra)
top_N = 3000
subset = rbind(
  dmrs[dmrs$type == "DMR-Hypo",][1:(top_N/2),], dmrs[dmrs$type == "DMR-Hyper",][1:(top_N/2),])
# subset = dmrs[abs(dmrs$areaStat)>=10000,]
subset = subset[subset$chr == "chr19",]
library(ggExtra)
p <- ggplot(subset, aes(x = meanMethy1, y = meanMethy2)) +
  geom_point(alpha = 0.1) +
  stat_density_2d(color = "red") + 
  labs(x ="Methylation level Normal", y = "Methylation level Tumor") +
  xlim(0,1) +
  ylim(0,1) + 
  geom_abline(slope=1, intercept = 0)
ggMarginal(p, type = "density")

ggplot(subset, aes(x = meanMethy1, y = meanMethy2)) +
  stat_density_2d_filled() +
  facet_wrap(~type) +
  theme_minimal() + 
  labs(x ="Methylation level Normal", y = "Methylation level Tumor")



df = dmrs[dmrs$chr=="chr1",]




library(ggplot2)
library(ggExtra)
library(gridExtra)

# Function to create individual plots
create_dmr_plot <- function(data, subset_condition, title_text, text_size = 28) {
  if (grepl("top_N", subset_condition)) {
    # Extract top_N value
    top_N <- as.numeric(gsub(".*top_N\\s*=\\s*(\\d+).*", "\\1", subset_condition))
    subset_data <- rbind(
      data[data$type == "DMR-Hypo",][1:(top_N/2),], 
      data[data$type == "DMR-Hyper",][1:(top_N/2),]
    )
  } else {
    # Extract areaStat threshold
    threshold <- as.numeric(gsub(".*>(\\d+).*", "\\1", subset_condition))
    subset_data <- data[abs(data$areaStat) >= threshold,]
  }
  
  p <- ggplot(subset_data, aes(x = meanMethy1, y = meanMethy2)) +
    geom_point(alpha = 0.1) +
    stat_density_2d(color = "red") + 
    labs(
      x = "Methylation level Normal", 
      y = "Methylation level Tumor",
      title = title_text
    ) +
    xlim(0, 1) +
    ylim(0, 1) +
    theme_minimal() +
    theme(
      text = element_text(size = text_size),
      axis.title = element_text(size = text_size, face = "bold"),
      axis.text = element_text(size = text_size),
      plot.title = element_text(size = text_size, face = "bold", hjust = 0.5),
      axis.ticks = element_line(size = 0.5),
      axis.ticks.length = unit(0.2, "cm")
    )+
    geom_abline(slope=1, intercept = 0)
  
  # Create marginal plot
  marginal_plot <- ggMarginal(p, type = "density")
  return(marginal_plot)
}

# Create all six plots
plot1 <- create_dmr_plot(dmrs, "top_N = 30000", "All DMRs",text_size=24)
plot2 <- create_dmr_plot(dmrs, "top_N = 3000", "Top 1500 Hypo & 1500 Hyper",text_size=24)
plot3 <- create_dmr_plot(dmrs, "top_N = 1000", "Top 500 Hypo & 500 Hyper")
plot4 <- create_dmr_plot(dmrs, "areaStat>300", "areaStat > 300")
plot5 <- create_dmr_plot(dmrs, "areaStat>1000", "areaStat > 1,000")
plot6 <- create_dmr_plot(dmrs, "areaStat>2000", "areaStat > 2,000")


# Arrange in 2x3 grid
final_plot <- grid.arrange(
  plot1, plot2,
  nrow = 1, ncol = 2,
  top = ""
)
subset_condition =  "top_N = 3000"
top_N <- as.numeric(gsub(".*top_N\\s*=\\s*(\\d+).*", "\\1", subset_condition))
subset_data <- rbind(
  dmrs[dmrs$type == "DMR-Hypo",][1:(top_N/2),], 
  dmrs[dmrs$type == "DMR-Hyper",][1:(top_N/2),]
)
df = subset_data[subset_data$chr=="chr4",]
create_dmr_plot(df, "top_N = 30000", "Chr4 Filtered DMRs",text_size=24)

df

