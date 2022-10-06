# This file was created to cell clustering
library(pbapply)
source("public/R/Methods.R")
set.seed(2020)

data.list.cluster <- readRDS("internal_datasets/scPred_pbmc_1/scPred_pbmc_1_silco.rds")
count.list.cluster <- data.list.cluster$counts
label.list.cluster <- lapply(data.list.cluster$labels, function(label) {
    ifelse(label == "doublet", 1, 0)
})


runCluster <- function(threshold, score, counts, resolution = 1) {
    if (!is.null(score)) {
        pred <- as.numeric(score > threshold)
        pred.index <- which(pred == 1)
        # remove predicted doublets
        if (length(pred.index) > 0) {
            doublet.clean <- counts[, -pred.index]
        } else {
            doublet.clean <- counts
        }
    } else {
        doublet.clean <- counts
    }

    # cluster cleaned data by louvain clustering
    clean.seurat <- preprocess(counts = doublet.clean)
    clean.seurat <- FindVariableFeatures(
        clean.seurat, 
        selection.method = "vst", 
        nfeatures = 2000)
    clean.seurat <- RunPCA(clean.seurat)
    clean.seurat <- FindNeighbors(clean.seurat, dims = 1:10)
    clean.seurat <- FindClusters(
        clean.seurat, 
        resolution = resolution, 
        algorithm = 1)
    return(clean.seurat)
}

# Clean data for scPred_pbmc_1 cluster 9  2： 8 | 9 11
count <- count.list.cluster
label <- label.list.cluster
path <- "outputs/scPred_pbmc_1"
cluster <- runCluster(0.5, label, count, resolution = 0.5)

data <- data.frame(
    row.names = cluster@assays[["RNA"]]@counts@Dimnames[[2]],
    cluster = as.numeric(cluster@meta.data[["seurat_clusters"]])
)

write.csv(data, paste(path, "clean_cluster.csv", sep = "/"))
saveRDS(cluster, paste(path, "clean_cluster.rds", sep = "/"))

# dir data for scPred_pbmc_1 cluster 9  2： 8 | 9 11
count <- count.list.cluster
label <- rep(0, dim(count)[[2]])
path <- "internal_outputs/scPred_pbmc_1"
cluster <- runCluster(0.5, label, count, resolution = 0.5)

data <- data.frame(
  row.names = cluster@assays[["RNA"]]@counts@Dimnames[[2]],
  cluster = as.numeric(cluster@meta.data[["seurat_clusters"]])
)

write.csv(data, paste(path, "dir_cluster.csv", sep = "/"))
saveRDS(cluster, paste(path, "dir_cluster.rds", sep = "/"))

# SoCube
count <- count.list.cluster
label <- label.list.cluster
path <- "outputs/scPred_pbmc_1/20220416-172623-141"
# path <- "outputs/scPred_pbmc_2/20220416-171845-445"
score <- read.csv(
    paste(path, "test_score_Bagging.csv", sep = "/"), header = FALSE)$V1
cluster <- pbsapply(
    seq(from = 0.1, by = 0.1, to = 0.9), 
    function(x) {
        return(nlevels(runCluster(
            x, 
            score, 
            count, 
            resolution = 0.5)@meta.data[["seurat_clusters"]]))
    }
)


data <- data.frame(
    row.names = cluster@assays[["RNA"]]@counts@Dimnames[[2]],
    cluster = as.numeric(cluster@meta.data[["seurat_clusters"]])
)

write.csv(data, paste(path, "cluster.csv", sep = "/"))
saveRDS(cluster, paste(path, "cluster.rds", sep = "/"))


# DoubletFinder
count <- count.list.cluster
label <- label.list.cluster

path <- "outputs/doubletfinder_result/scPred_pbmc_1"
if (!dir.exists(path)) {
    dir.create(path)
}
# score <- preditctByDoubletFinder(count)
score <- read.csv(
    paste(path, "sorce.csv", sep = "/"), 
    header = TRUE)$x
cluster <- pbsapply(
    seq(from = 0.1, by = 0.1, to = 0.9), 
    function(x) {
        return(nlevels(runCluster(
            x, 
            score, 
            count, 
            resolution = 0.5)@meta.data[["seurat_clusters"]]))
    }
)
cluster <- runCluster(0.5, score, count, resolution = 0.5)

data <- data.frame(
    row.names = cluster@assays[["RNA"]]@counts@Dimnames[[2]],
    cluster = as.numeric(cluster@meta.data[["seurat_clusters"]])
)
write.csv(score, paste(path, "sorce.csv", sep = "/"))
write.csv(data, paste(path, "cluster.csv", sep = "/"))
saveRDS(cluster, paste(path, "cluster.rds", sep = "/"))

# Solo
count <- count.list.cluster
label <- label.list.cluster
path <- "outputs/solo_result/scPred_pbmc_1_2010"
score <- read.csv(
    paste(path, "softmax_scores.csv", sep = "/"), 
    header = FALSE)$V1
cluster <- pbsapply(
    seq(from = 0.1, by = 0.1, to = 0.9), 
    function(x) {
        return(nlevels(runCluster(
            x, 
            score, 
            count, 
            resolution = 0.5)@meta.data[["seurat_clusters"]]))
})
cluster <- runCluster(0.25, score, count, resolution = 0.5)

data <- data.frame(
    row.names = cluster@assays[["RNA"]]@counts@Dimnames[[2]],
    cluster = as.numeric(cluster@meta.data[["seurat_clusters"]])
)

write.csv(data, paste(path, "cluster.csv", sep = "/"))
saveRDS(cluster, paste(path, "cluster.rds", sep = "/"))


# scDblFinder
library(scDblFinder)
count <- count.list.cluster
label <- label.list.cluster

path <- "internal_outputs/scDblFinder_result/scPred_pbmc_1"
if (!dir.exists(path)) {
  dir.create(path)
}
score <- scDblFinder(count)$scDblFinder.score
# score <- read.csv(
#   paste(path, "sorce.csv", sep = "/"), 
#   header = TRUE)$x
cluster <- pbsapply(
  seq(from = 0.1, by = 0.1, to = 0.9), 
  function(x) {
    return(nlevels(runCluster(
      x, 
      score, 
      count, 
      resolution = 0.5)@meta.data[["seurat_clusters"]]))
  }
)
cluster <- runCluster(0.5, score, count, resolution = 0.5)

data <- data.frame(
  row.names = cluster@assays[["RNA"]]@counts@Dimnames[[2]],
  cluster = as.numeric(cluster@meta.data[["seurat_clusters"]])
)
write.csv(score, paste(path, "sorce.csv", sep = "/"))
write.csv(data, paste(path, "cluster.csv", sep = "/"))
saveRDS(cluster, paste(path, "cluster.rds", sep = "/"))
