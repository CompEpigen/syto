library(DSS)
require(bsseq)
library(dplyr)
library(data.table)
library(progress)

makeBSseqDataCustom <- function(dat, sampleNames) {
  n0 <- length(dat)
  if (missing(sampleNames)) 
    sampleNames <- paste("sample", 1:n0, sep = "")
  alldat <- dat[[1]]
  if (any(alldat[, "N"] < alldat[, "X"], na.rm = TRUE)) 
    stop("Some methylation counts are greater than coverage.\n")
  ix.X <- which(colnames(alldat) == "X")
  ix.N <- which(colnames(alldat) == "N")
  colnames(alldat)[ix.X] <- "X1"
  colnames(alldat)[ix.N] <- "N1"
  pb <- progress_bar$new(format = "(:spin) [:bar] :percent [Elapsed time: :elapsedfull || Estimated time remaining: :eta]",
                         total = length(dat)-1,
                         complete = "=",   # Completion bar character
                         incomplete = "-", # Incomplete bar character
                         current = ">",    # Current bar character
                         clear = FALSE,    # If TRUE, clears the bar when finish
                         width = 100,
                         show_after =0)      # Width of the progress bar
  pb$message("Merging samples data")
  pb$tick(0)
  if (n0 > 1) {
    for (i in 2:n0) {
      thisdat <- dat[[i]]
      if (any(thisdat[, "N"] < thisdat[, "X"], na.rm = TRUE)) 
        stop("Some methylation counts are greater than coverage.\n")
      ix.X <- which(colnames(thisdat) == "X")
      ix.N <- which(colnames(thisdat) == "N")
      colnames(thisdat)[c(ix.X, ix.N)] <- paste(c("X", 
                                                  "N"), i, sep = "")
      alldat <- data.table::merge.data.table(alldat, thisdat, by = c("chr", "pos"), all = TRUE)
      pb$tick()
    }
  }
  pb$terminate()
  ix.X <- grep("X", colnames(alldat))
  ix.N <- grep("N", colnames(alldat))
  alldat[is.na(alldat)] <- 0
  M <- as.matrix(alldat[, ix.X, drop = FALSE])
  Cov <- as.matrix(alldat[, ix.N, drop = FALSE])
  colnames(M) <- colnames(Cov) <- sampleNames
  idx <- split(1:length(alldat$chr), alldat$chr)
  M.ordered <- M
  Cov.ordered <- Cov
  pos.ordered <- alldat$pos
  for (i in seq(along = idx)) {
    thisidx = idx[[i]]
    thispos = alldat$pos[thisidx]
    dd = diff(thispos)
    if (min(dd) < 0) {
      warning(paste0("CG positions in chromosome ", names(idx)[i], 
                     " is not ordered. Reorder CG sites.\n"))
      iii = order(thispos)
      M.ordered[thisidx, ] <- M[thisidx, ][iii, ]
      Cov.ordered[thisidx, ] <- Cov[thisidx, ][iii, ]
      pos.ordered[thisidx] <- alldat$pos[thisidx][iii]
    }
  }
  chr = alldat$chr
  rm(alldat)
  result <- BSseq(chr = chr, pos = pos.ordered, M = M.ordered, 
                  Cov = Cov.ordered)
  result
}
assignInNamespace("makeBSseqData", makeBSseqDataCustom, ns = "DSS")

processed_data_folders <- list.dirs()[list.dirs() %>% stringr::str_detect("dss_input_modkit/chr")]

chromosomes = processed_data_folders %>% basename()
chromosomes = chromosomes[2:length(chromosomes)]
# chromosomes = chromosomes[c(1:11,13:24)]

for (i in c(1:length(chromosomes))){
  file_paths <- file.path(processed_data_folders[i], list.files(processed_data_folders[i]))
  chromosome <- chromosomes[i]
  t1 <- Sys.time()
  suppressMessages(dat <- purrr::set_names(file_paths, file_paths %>% tools::file_path_sans_ext() %>% basename() ) %>%
    purrr::map(~ fst::read.fst(.x) %>% group_by(chr, pos) %>% summarise(N=sum(N), X=sum(X)) %>% ungroup()))
  
  sample_names <- names(dat)
  t2 <- Sys.time()
  t2-t1
  BSobj = DSS::makeBSseqData(dat, sample_names)
  rm(dat)
  
  t1 <- Sys.time()
  dmlTest = DSS::DMLtest(BSobj, group1=paste("normal", c(1:5), sep="_"), group2=paste("tumor", c(1:5), sep="_"),
                         ncores = 1, smoothing=TRUE, smoothing.span=25)
  fst::write.fst(dmlTest, paste0("processed_data/dml_modkit_test_results_25_smoothed/dml_",chromosome,".fst"))
  t2 <- Sys.time()
  t2-t1
  rm(BSobj)
  rm(dmlTest)
}






