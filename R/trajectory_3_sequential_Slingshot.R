# This file was create to do 3 sequential cell trajectory inference
source("public/R/Methods.R")
sim.data <- readRDS(
    "internal_compares/Doublet-Detection-Benchmark/paper_sim/sim_psudotime_3_sequential.rds"
)
sim.doublet <- sim.data[[1]]
dim(sim.doublet)
sim.types <- sim.data[[2]]
table(sim.types)
################################################################################
# Negative control with doublet
################################################################################
f <- function() {
    counts <- sim.doublet
    types <- sim.types
    sim <- trajectoryBySlingshot(counts)
    pca <- as.data.frame(reducedDims(sim)$PCA)
    pca$type <- as.numeric(types)
    pca$type <- ifelse(pca$type == 0, "singlet", "doublet")
    pca$type <- as.factor(pca$type)
    pdf("plots/trajectory_control/negative_control_3_sequential_th_0.5.pdf")
    palette(c("red", "grey"))
    plot(pca$PC1, pca$PC2,
        col = pca$type, pch = 16, asp = 0, 
        xlab = NA, ylab = NA, xaxt = "n", yaxt = "n", cex.main = 2, cex = 1.3, font.main = 1
    )
    lines(SlingshotDataSet(sim), lwd = 7, type = "lineages", col = "black")
    dev.off()
}
f()
rm(f)
################################################################################
# Positive control with no doublet
################################################################################
f <- function() {
    counts <- sim.doublet[, sim.types == 0]
    types <- sim.types[sim.types == 0]
    sim <- trajectoryBySlingshot(counts)
    pca <- as.data.frame(reducedDims(sim)$PCA)
    pca$type <- as.numeric(types)
    pca$type <- ifelse(pca$type == 0, "singlet", "doublet")
    pca$type <- as.factor(pca$type)
    pdf("plots/trajectory_control/positive_control_3_sequential_th_0.5.pdf")
    palette(c("grey", "grey"))
    plot(pca$PC1, pca$PC2,
        col = pca$type, pch = 16, asp = 0, 
        xlab = NA, ylab = NA, xaxt = "n", yaxt = "n", cex.main = 2, cex = 1.3, font.main = 1
    )
    lines(SlingshotDataSet(sim), lwd = 7, type = "lineages", col = "black")
    dev.off()
}
f()
rm(f)
################################################################################
# SoCube
################################################################################
f <- function() {
    path <- "sim_psudotime_3_sequential/20220330-220112-142"
    score <- read.csv(
        paste(
            "outputs", path, "test_score_Bagging.csv",
            sep = "/"
        ),
        header = FALSE
    )$V1
    length(score)
    hist(score)
    pred.index <- which(as.numeric(score > 0.5) == 1)
    counts <- sim.doublet[, -pred.index]
    dim(counts)
    types <- sim.types[-pred.index]
    table(types)
    sim <- trajectoryBySlingshot(counts)
    pca <- as.data.frame(reducedDims(sim)$PCA)
    pca$type <- as.numeric(types)
    pca$type <- ifelse(pca$type == 0, "singlet", "doublet")
    pca$type <- as.factor(pca$type)
    pdf(
        paste("plots", path, "trajectory_3_sequential_th_0.5.pdf", sep = "/")
    )
    palette(c("red", "grey"))
    plot(
        pca$PC1,
        pca$PC2,
        col = pca$type,
        pch = 16,
        asp = 0,
        xlab = NA,
        ylab = NA,
        xaxt = "n",
        yaxt = "n",
        cex.main = 2,
        cex = 1.3,
        font.main = 1
    )
    lines(SlingshotDataSet(sim), lwd = 7, type = "lineages", col = "black")
    dev.off()
}
f()
rm(f)
################################################################################
# Solo
################################################################################
f <- function() {
    score <- read.csv(
        "outputs/solo_result/sim_psudotime_3_sequential_2020/softmax_scores.csv",
        header = FALSE
    )$V1
    length(score)
    hist(score)
    pred.index <- which(as.numeric(score > 0.5) == 1)
    counts <- sim.doublet[, -pred.index]
    dim(counts)
    types <- sim.types[-pred.index]
    table(types)
    sim <- trajectoryBySlingshot(counts)
    pca <- as.data.frame(reducedDims(sim)$PCA)
    pca$type <- as.numeric(types)
    pca$type <- ifelse(pca$type == 0, "singlet", "doublet")
    pca$type <- as.factor(pca$type)
    pdf("plots/solo_result/trajectory_3_sequential_th_0.5_2020.pdf")
    palette(c("red", "grey"))
    plot(
        pca$PC1,
        pca$PC2,
        col = pca$type,
        pch = 16,
        asp = 0,
        xlab = NA,
        ylab = NA,
        xaxt = "n",
        yaxt = "n",
        cex.main = 2,
        cex = 1.3,
        font.main = 1
    )
    lines(SlingshotDataSet(sim), lwd = 7, type = "lineages", col = "black")
    dev.off()
}
f()
rm(f)
################################################################################
# DoubletFinder
################################################################################
f <- function() {
    score <- preditctByDoubletFinder(sim.doublet)
    hist(score)
    pred.index <- which(as.numeric(score > 0.5) == 1)
    counts <- sim.doublet[, -pred.index]
    dim(counts)
    types <- sim.types[-pred.index]
    table(types)
    sim <- trajectoryBySlingshot(counts)
    pca <- as.data.frame(reducedDims(sim)$PCA)
    pca$type <- as.numeric(types)
    pca$type <- ifelse(pca$type == 0, "singlet", "doublet")
    pca$type <- as.factor(pca$type)
    pdf("plots/doubletfinder_result/trajectory_3_sequential_th_0.5.pdf")
    palette(c("red", "grey"))
    plot(
        pca$PC1,
        pca$PC2,
        col = pca$type,
        pch = 16,
        asp = 0,
        xlab = NA,
        ylab = NA,
        xaxt = "n",
        yaxt = "n", cex.main = 2, cex = 1.3, font.main = 1
    )
    lines(SlingshotDataSet(sim), lwd = 7, type = "lineages", col = "black")
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
  hist(score)
  pred.index <- which(as.numeric(score > 0.5) == 1)
  counts <- sim.doublet[, -pred.index]
  dim(counts)
  types <- sim.types[-pred.index]
  table(types)
  sim <- trajectoryBySlingshot(counts)
  pca <- as.data.frame(reducedDims(sim)$PCA)
  pca$type <- as.numeric(types)
  pca$type <- ifelse(pca$type == 0, "singlet", "doublet")
  pca$type <- as.factor(pca$type)
  pdf("internal_plots/scDblFinder_result/trajectory_3_sequential_th_0.5.pdf")
  palette(c("red", "grey"))
  plot(
    pca$PC1,
    pca$PC2,
    col = pca$type,
    pch = 16,
    asp = 0,
    xlab = NA,
    ylab = NA,
    xaxt = "n",
    yaxt = "n", cex.main = 2, cex = 1.3, font.main = 1
  )
  lines(SlingshotDataSet(sim), lwd = 7, type = "lineages", col = "black")
  dev.off()
}
f()
rm(f)
