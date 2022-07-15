# This file was created to integrate some general methods
library(DoubletFinder)
library(Seurat)
library(dplyr)
library(slingshot, quietly = TRUE)
library(mclust, quietly = TRUE)
library(SingleCellExperiment, quietly = TRUE)
set.seed(2020)
preprocess <- function(counts) {
    return(
        CreateSeuratObject(
            counts = as.matrix(counts),
            project = "doublet",
            min.cells = 1,
            min.features = 1
        ) %>%
            NormalizeData() %>%
            ScaleData()
    )
}
preditctByDoubletFinder <- function(counts, cores = 1) {
    ## Pre-process Seurat object (standard) ------------
    doublet.seurat <- preprocess(counts = counts)
    doublet.seurat <- FindVariableFeatures(doublet.seurat,
        selection.method = "vst",
        nfeatures = 2000
    )
    doublet.seurat <- RunPCA(doublet.seurat)
    ## pK Identification (no ground-truth) -------------
    sweep.res.doublet <- paramSweep_v3(
        doublet.seurat,
        PCs = 1:10,
        sct = FALSE,
        num.cores = cores
    )
    sweep.stats.doublet <- summarizeSweep(sweep.res.doublet,
        GT = FALSE
    )
    bcmvn.doublet <- find.pK(sweep.stats.doublet)
    pK <- bcmvn.doublet$pK[which.max(bcmvn.doublet$BCmetric)]
    pK <- as.numeric(levels(pK))[pK]
    doublet.seurat <- doubletFinder_v3(
        doublet.seurat,
        PCs = 1:10,
        pN = 0.25,
        pK = pK,
        nExp = 60
    )
    attribute <- paste("pANN", 0.25, pK, 60, sep = "_")
    score <- doublet.seurat@meta.data[[attribute]]
    return(score)
}
# quantile normalization recommended by slingshot
FQnorm <- function(counts) {
    rk <- apply(counts, 2, rank, ties.method = "min")
    dim(rk)
    counts.sort <- apply(counts, 2, sort)
    dim(counts.sort)
    refdist <- apply(counts.sort, 1, median)
    norm <- apply(rk, 2, function(r) {
        refdist[r]
    })
    rownames(norm) <- rownames(counts)
    return(norm)
}
trajectoryBySlingshot <- function(counts) {
    sim <- SingleCellExperiment(assays = List(counts = counts))
    assays(sim)$norm <- FQnorm(assays(sim)$counts)
    # pca dimension reduction
    pca <- prcomp(t(log1p(assays(sim)$norm)), scale. = FALSE)
    rd1 <- pca$x[, 1:2]
    reducedDims(sim) <- SimpleList(PCA = rd1)
    # GMM clustering requested by slingshot
    cl1 <- Mclust(rd1)$classification
    table(cl1)
    colData(sim)$GMM <- cl1
    # lineage reconstruction
    return(slingshot(sim, clusterLabels = "GMM", reducedDim = "PCA"))
}