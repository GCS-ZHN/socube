# This file was create to do temporally DEG analysis by Slingshot
source("public/R/Methods.R")
sim.data <-
    readRDS(
        "internal_compares/Doublet-Detection-Benchmark/synthetic_datasets/sim_temporally_DE.rds"
    )
sim.doublet <- sim.data[[1]]
dim(counts)
sim.type <- sim.data[[2]]
table(sim.type)
temporalDEbySlingshot <- function(counts, types, DE) {
    sim <- trajectoryBySlingshot(counts)
    sim$slingPseudotime_1[which(types == 0)]
    cor(
        1:length(sim$slingPseudotime_1[which(types == 0)]),
        sim$slingPseudotime_1[which(types == 0)]
    )
    # temporally expressed genes analysis
    require(gam)
    t <- sim$slingPseudotime_1
    Y <- log1p(assays(sim)$norm)
    dim(Y)
    # fit a GAM with a loess term for pseudotime
    gam.pval <- apply(Y, 1, function(z) {
        d <- data.frame(z = z, t = t)
        suppressWarnings({
            tmp <- suppressWarnings(gam(z ~ lo(t), data = d))
        })
        p <- summary(tmp)[3][[1]][2, 3]
        p
    })
    # p-value threshold
    p <- .05
    # find temporally expressed genes
    # calculated precision, recall, true negative rate (TNR)
    findDE <- names(gam.pval[gam.pval <= p])
    length(findDE)
    findnonDE <- names(gam.pval[gam.pval > p])
    length(findnonDE)
    nonDE <- setdiff(row.names(counts), DE)
    tp <- length(intersect(findDE, DE))
    fp <- length(setdiff(findDE, DE))
    fn <- length(setdiff(findnonDE, nonDE))
    tn <- length(intersect(findnonDE, nonDE))
    return(c(
        precision = tp / (tp + fp),
        recall = tp / (tp + fn),
        tnr = tn / (tn + fp)
    ))
}
################################################################################
# Positive control
################################################################################
f <- function() {
    counts <- sim.doublet[, sim.type == 0]
    types <- sim.type[sim.type == 0]
    DE <- row.names(counts)[501:750]
    temporalDEbySlingshot(
        counts = counts,
        types = types,
        DE = DE
    )
}
f()
rm(f)
################################################################################
# Negative control
################################################################################
f <- function() {
    counts <- sim.doublet
    types <- sim.type
    DE <- row.names(counts)[501:750]
    temporalDEbySlingshot(
        counts = counts,
        types = types,
        DE = DE
    )
}
f()
rm(f)
################################################################################
# Socube
################################################################################
f <- function() {
    score <-
        read.csv(
            "outputs/sim_psudotime_temporally_expressed_genes/20220331-162025-965/test_score_Bagging.csv",
            header = FALSE
        )$V1
    pred.index <- which(as.numeric(score > 0.5) == 1)
    counts <- sim.doublet[, -pred.index]
    dim(counts)
    types <- sim.type[-pred.index]
    table(types)
    DE <- row.names(counts)[501:750]
    temporalDEbySlingshot(
        counts = counts,
        types = types,
        DE = DE
    )
}
f()
rm(f)
################################################################################
# DoubletFinder
################################################################################
f <- function() {
    score <- preditctByDoubletFinder(sim.doublet)
    pred.index <- which(as.numeric(score > 0.5) == 1)
    counts <- sim.doublet[, -pred.index]
    dim(counts)
    types <- sim.type[-pred.index]
    table(types)
    DE <- row.names(counts)[501:750]
    temporalDEbySlingshot(
        counts = counts,
        types = types,
        DE = DE
    )
}
f()
rm(f)
# scDblFinder
library(scDblFinder)
f <- function() {
  score <- scDblFinder(sim.doublet)$scDblFinder.score
  pred.index <- which(as.numeric(score > 0.5) == 1)
  counts <- sim.doublet[, -pred.index]
  dim(counts)
  types <- sim.type[-pred.index]
  table(types)
  DE <- row.names(counts)[501:750]
  temporalDEbySlingshot(
    counts = counts,
    types = types,
    DE = DE
  )
}
f()
rm(f)