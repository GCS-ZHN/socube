# This file was created to do DEG analysis
source("public/R/Methods.R")
library(Matrix)
library(pbapply)
library(ggplot2)
library(scran)
library(scDblFinder)
set.seed(2020)
deg <- function(count, cluster, trueLabel, test = "wilcox") {
    gene <- rownames(count)
    count <- preprocess(count) %>%
        FindVariableFeatures(., selection.method = "vst", nfeatures = 2000) %>%
        RunPCA()

    Idents(count) <- as.factor(cluster)
    marker <- FindMarkers(
        count,
        ident.1 = "0",
        ident.2 = "1",
        test.use = test
    )

    de <- rownames(marker[marker$p_val_adj <= 0.05, ])

    tp <- length(intersect(de, trueLabel))
    fp <- length(setdiff(de, trueLabel))
    fn <- length(setdiff(trueLabel, de))
    tn <- length(intersect(setdiff(gene, trueLabel), setdiff(gene, de)))
    return(c(
        precision = tp / (tp + fp),
        recall = tp / (tp + fn),
        tnr = tn / (tn + fp)
    ))
}
# read simulation data with groud truth DE genes
sim.data <- readRDS("internal_compares/Doublet-Detection-Benchmark/paper_sim/sim_DE.rds")
sim.doublet <- sim.data[[1]]
sim.cluster <- sim.data[[2]]
# up and down regulated genes
de.up <- sim.data[[3]]
length(de.up)
de.down <- sim.data[[4]]
length(de.down)
de.truth <- c(de.up, de.down)
length(de.truth)
################################################################################
# DE Analysis for SoCube
################################################################################
f <- function() {
    file <- "internal_outputs/sim_DE/20220324-221950-309/test_score_Bagging.csv"
    score <- read.csv(file, header = FALSE)$V1
    pred.index <- which(score > sort(score, decreasing = TRUE)[667])
    length(pred.index)
    cluster <- sim.cluster[-pred.index]
    doublet <- sim.doublet[, -pred.index]
    wilcox <- deg(count = doublet, cluster = cluster, trueLabel = de.truth, test = "wilcox")
    mast <- deg(count = doublet, cluster = cluster, trueLabel = de.truth, test = "MAST")
    # bimod <- deg(count = doublet, cluster = cluster, trueLabel = de.truth, test = "bimod")
    print(wilcox)
    print(mast)
    # print(bimod)
}
f()
rm(f)

# DE Analyisis for scDblFinder
f <- function() {
  score <- scDblFinder(sim.doublet)$scDblFinder.score
  pred.index <- which(score > sort(score, decreasing = TRUE)[667])
  length(pred.index)
  cluster <- sim.cluster[-pred.index]
  doublet <- sim.doublet[, -pred.index]
  wilcox <- deg(count = doublet, cluster = cluster, trueLabel = de.truth, test = "wilcox")
  mast <- deg(count = doublet, cluster = cluster, trueLabel = de.truth, test = "MAST")
  # bimod <- deg(count = doublet, cluster = cluster, trueLabel = de.truth, test = "bimod")
  print(wilcox)
  print(mast)
  # print(bimod)
}
f()
rm(f)
