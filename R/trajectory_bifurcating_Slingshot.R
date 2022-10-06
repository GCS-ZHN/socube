# This file was create to do bifurcating cell trajectory inference
source("public/R/Methods.R")
# read simulation data with bifurcating trajectories
sim.data <-
    readRDS("internal_compares/Doublet-Detection-Benchmark/paper_sim/sim_psudotime_bifurcating.rds")
sim.doublet <- sim.data[[1]]
dim(sim.doublet)
sim.type <- sim.data[[2]]
table(sim.type)
################################################################################
# SoCube
################################################################################
f <- function() {
    path <- "sim_psudotime_bifurcating/20220329-154456-737"
    score <-
        read.csv(paste("outputs", path, "test_score_Bagging.csv", sep = "/"),
            header = FALSE
        )$V1
    length(score)
    hist(score)
    pred.index <- which(as.numeric(score > 0.5) == 1)
    counts <- sim.doublet[, -pred.index]
    dim(counts)
    types <- sim.type[-pred.index]
    table(types)
    sim <- trajectoryBySlingshot(counts)
    pca <- as.data.frame(reducedDims(sim)$PCA)
    pca$type <- as.numeric(types)
    pca$type <- ifelse(pca$type == 0, "singlet", "doublet")
    pca$type <- as.factor(pca$type)
    pdf(paste("plots", path, "trajectory_bifurcating_th_0.5.pdf", sep = "/"))
    palette(c("red", "grey"))
    plot(
        pca$PC1,
        pca$PC2,
        col = pca$type,
        pch = 16,
        asp = 0,
        main = "SoCube"
    )
    lines(SlingshotDataSet(sim), lwd = 2, col = "black")
    dev.off()
}
f()
rm(f)
################################################################################
# Solo
################################################################################
f <- function() {
    file <-
        "outputs/solo_result/sim_psudotime_bifurcating_2010/softmax_scores.csv"
    score <-
        read.csv(file, header = FALSE)$V1
    length(score)
    hist(score)
    pred.index <- which(as.numeric(score > 0.5) == 1)
    counts <- sim.doublet[, -pred.index]
    dim(counts)
    types <- sim.type[-pred.index]
    table(types)
    sim <- trajectoryBySlingshot(counts)
    pca <- as.data.frame(reducedDims(sim)$PCA)
    pca$type <- as.numeric(types)
    pca$type <- ifelse(pca$type == 0, "singlet", "doublet")
    pca$type <- as.factor(pca$type)
    pdf("plots/solo_result/trajectory_bifurcating_th_0.5_2010.pdf")
    palette(c("red", "grey"))
    plot(
        pca$PC1,
        pca$PC2,
        col = pca$type,
        pch = 16,
        asp = 0,
        main = "Solo"
    )
    lines(SlingshotDataSet(sim), lwd = 2, col = "black")
    dev.off()
}
f()
rm(f)
################################################################################
# DoubletFinder
################################################################################
f <- function() {
    score <-
        preditctByDoubletFinder(sim.doublet)
    length(score)
    hist(score)
    pred.index <- which(as.numeric(score > 0.5) == 1)
    counts <- sim.doublet[, -pred.index]
    dim(counts)
    types <- sim.type[-pred.index]
    table(types)
    sim <- trajectoryBySlingshot(counts)
    pca <- as.data.frame(reducedDims(sim)$PCA)
    pca$type <- as.numeric(types)
    pca$type <- ifelse(pca$type == 0, "singlet", "doublet")
    pca$type <- as.factor(pca$type)
    pdf("plots/doubletfinder_result/trajectory_bifurcating_th_0.5.pdf")
    palette(c("red", "grey"))
    plot(
        pca$PC1,
        pca$PC2,
        col = pca$type,
        pch = 16,
        asp = 0
    )
    lines(SlingshotDataSet(sim), lwd = 2, col = "black")
    dev.off()
}
f()
rm(f)
################################################################################
# Positive Control
################################################################################
f <- function() {
    counts <- sim.doublet[, sim.type == 0]
    dim(counts)
    types <- sim.type[sim.type == 0]
    table(types)
    sim <- trajectoryBySlingshot(counts)
    pca <- as.data.frame(reducedDims(sim)$PCA)
    pca$type <- as.numeric(types)
    pca$type <- ifelse(pca$type == 0, "singlet", "doublet")
    pca$type <- as.factor(pca$type)
    pdf("plots/trajectory_control/positive_control_bifurcating_th_0.5.pdf")
    palette(c("grey", "grey"))
    plot(
        pca$PC1,
        pca$PC2,
        col = pca$type,
        pch = 16,
        asp = 0
    )
    lines(SlingshotDataSet(sim), lwd = 2, col = "black")
    dev.off()
}
f()
rm(f)
################################################################################
# Negative Control
################################################################################
f <- function() {
    counts <- sim.doublet
    dim(counts)
    types <- sim.type
    table(types)
    sim <- trajectoryBySlingshot(counts)
    pca <- as.data.frame(reducedDims(sim)$PCA)
    pca$type <- as.numeric(types)
    pca$type <- ifelse(pca$type == 0, "singlet", "doublet")
    pca$type <- as.factor(pca$type)
    pdf("plots/trajectory_control/negative_control_bifurcating_th_0.5.pdf")
    palette(c("red", "grey"))
    plot(
        pca$PC1,
        pca$PC2,
        col = pca$type,
        pch = 16,
        asp = 0
    )
    lines(SlingshotDataSet(sim), lwd = 2, col = "black")
    dev.off()
}
f()
rm(f)

################################################################################
# scDblFinder
################################################################################
library(scDblFinder)
f <- function() {
  score <- scDblFinder(sim.doublet)$scDblFinder.score
  length(score)
  hist(score)
  pred.index <- which(as.numeric(score > 0.5) == 1)
  counts <- sim.doublet[, -pred.index]
  dim(counts)
  types <- sim.type[-pred.index]
  table(types)
  sim <- trajectoryBySlingshot(counts)
  pca <- as.data.frame(reducedDims(sim)$PCA)
  pca$type <- as.numeric(types)
  pca$type <- ifelse(pca$type == 0, "singlet", "doublet")
  pca$type <- as.factor(pca$type)
  pdf("internal_plots/scDblFinder_result/trajectory_bifurcating_th_0.5.pdf")
  palette(c("red", "grey"))
  plot(
    pca$PC1,
    pca$PC2,
    col = pca$type,
    pch = 16,
    asp = 0
  )
  lines(SlingshotDataSet(sim), lwd = 2, col = "black")
  dev.off()
}
f()
rm(f)