# This file was create to do temporally DEG analysis by TSCAN
library(TSCAN)
source("public/R/Methods.R")
# read simulation data with single trajectory and temporally expressed genes
data <-
    readRDS(
        "internal_compares/Doublet-Detection-Benchmark/synthetic_datasets/sim_temporally_DE.rds"
    )
sim.doublet <- data[[1]]
dim(sim.doublet)
sim.types <- data[[2]]
table(sim.types)
DE <- data[[3]]
temporalDEbyTSCAN <- function(counts, DE) {
    procdata <- preprocess(counts)@assays$RNA@scale.data
    lpsmclust <- exprmclust(procdata)
    lpsorder <- TSCANorder(lpsmclust)
    diffval <- difftest(procdata, lpsorder)
    # select temporally expressed genes under qvlue cutoff of 0.05
    findDE <-
        row.names(diffval)[diffval$qval <= 0.05]
    length(findDE)
    findnonDE <-
        row.names(diffval)[diffval$qval > 0.05]
    length(findnonDE)
    nonDE <- setdiff(row.names(counts), DE)
    # calculate precision, recall, and true negative rate
    tp <- length(intersect(findDE, DE))
    tp
    fp <- length(setdiff(findDE, DE))
    fp
    fn <- length(setdiff(findnonDE, nonDE))
    fn
    tn <- length(intersect(findnonDE, nonDE))
    tn
    return(c(
        precision = tp / (tp + fp),
        recall = tp / (tp + fn),
        tnr = tn / (tn + fp)
    ))
}
################################################################################
# Positive control
################################################################################
# data preprocess
temporalDEbyTSCAN(counts = sim.doublet[, sim.types == 0], DE)
################################################################################
# Negative control
################################################################################
temporalDEbyTSCAN(counts = sim.doublet, DE)
################################################################################
# SoCube
################################################################################
f <- function() {
    score <-
        read.csv(
            "outputs/sim_psudotime_temporally_expressed_genes/20220331-162025-965/test_score_Bagging.csv",
            header = FALSE
        )$V1
    pred.index <- which(as.numeric(score > 0.5) == 1)
    temporalDEbyTSCAN(counts = sim.doublet[, -pred.index], DE)
}
f()
rm(f)

################################################################################
# Doubletfinder
################################################################################
f <- function() {
    score <- preditctByDoubletFinder(sim.doublet)
    pred.index <- which(as.numeric(score > 0.5) == 1)
    temporalDEbyTSCAN(counts = sim.doublet[, -pred.index], DE)
}
f()
rm(f)

################################################################################
#scDblFinder
################################################################################
library(scDblFinder)
f <- function() {
  score <- scDblFinder(sim.doublet)$scDblFinder.score
  pred.index <- which(as.numeric(score > 0.5) == 1)
  temporalDEbyTSCAN(counts = sim.doublet[, -pred.index], DE)
}
f()
rm(f)