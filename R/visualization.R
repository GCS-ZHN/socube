# This file was created to visualize cell clustering plots
library(Seurat)
library(stringr)
cluster <- readRDS("internal_datasets/scPred_pbmc_1/scPred_pbmc_1_8.rds")
cell_type <- cluster@meta.data$cell_type
cell_name <- cluster@assays$RNA@counts@Dimnames[[2]]
names(cell_type) <- cell_name
path <- "internal_outputs/scPred_pbmc_1"
cluster_clean <- readRDS(paste(path, "clean_cluster.rds", sep = "/"))
cluster_clean <- Seurat::RunUMAP(cluster_clean, dims = 1:30)
sim_cell_name <- cluster_clean@assays$RNA@counts@Dimnames[[2]]
sim_cell_type <- cell_type[sim_cell_name]
cluster_clean@meta.data$cell_type <- sim_cell_type
pdf(file = "materials/figures/Figure.7-A-Control.pdf",
    width = 8,
    height = 6,
    title = "Control")
Seurat::DimPlot(
    cluster_clean,
    group.by = "cell_type",
    repel = TRUE,
    label = TRUE,
    cols = c(
    "B cell" = "#F8766D",
    "CD4 T cell" = "#E68613",
    "CD8 T cell" = "#7CAE00",
    "cDC" = "#C77CFF",
    "Mono" = "#B2A7B8",
    "NK cell" = "#00A9FF",
    "pDC" = "#72464B",
    "Plasma cell" = "#00BC9F"
))
dev.off()
empty_cell_type <- rep("missing cell", length(sim_cell_type))
names(empty_cell_type) <- sim_cell_name
# SoCube
socube_path <- "internal_outputs/scPred_pbmc_1/20220416-172623-141"
socube_cluster <- readRDS(paste(socube_path, "cluster.rds", sep = "/"))
socube_cluster_name <- as.vector(socube_cluster@meta.data$seurat_clusters)
socube_cluster_name <- sapply(
    socube_cluster_name,
    function(x) paste("cluster", x, sep = "_"))
names(socube_cluster_name) <- socube_cluster@assays$RNA@counts@Dimnames[[2]]
socube_empty_type <- empty_cell_type
socube_cell_name <- intersect(
    socube_cluster@assays$RNA@counts@Dimnames[[2]],
    sim_cell_name)
socube_empty_type[socube_cell_name] <- socube_cluster_name[socube_cell_name]
cluster_clean@meta.data$cluster_socube <- socube_empty_type
pdf(file = "materials/figures/Figure.7-A-SoCube.pdf",
    width = 8,
    height = 6,
    title = "SoCube")
Seurat::DimPlot(
    cluster_clean,
    group.by = "cluster_socube",
    repel = TRUE,
    label = TRUE,
    cols = c(
        "cluster_0" = "#00BC9F",
        "cluster_1" = "#E68613",
        "cluster_2" = "#7CAE00",
        "cluster_3" = "#00A9FF",
        "cluster_4" = "#F8766D",
        "cluster_5" = "#B2A7B8",
        "cluster_6" = "#C77CFF",
        "cluster_7" = "#72464B",
        "missing cell" = "#E6E6FA"
))
dev.off()
rm(list = ls()[sapply(ls(), function(x) str_starts(x, "socube_"))])
# Solo
solo_path <- "internal_outputs/solo_result/scPred_pbmc_1_2010"
solo_cluster <- readRDS(paste(solo_path, "cluster.rds", sep = "/"))
solo_cluster_name <- as.vector(solo_cluster@meta.data$seurat_clusters)
solo_cluster_name <- sapply(
    solo_cluster_name,
    function(x) paste("cluster", x, sep = "_"))
names(solo_cluster_name) <- solo_cluster@assays$RNA@counts@Dimnames[[2]]
solo_empty_type <- empty_cell_type
solo_cell_name <- intersect(
    solo_cluster@assays$RNA@counts@Dimnames[[2]],
    sim_cell_name)
solo_empty_type[solo_cell_name] <- solo_cluster_name[solo_cell_name]
cluster_clean@meta.data$cluster_solo <- solo_empty_type
pdf(file = "materials/figures/Figure.7-A-Solo.pdf",
    width = 8,
    height = 6,
    title = "Solo")
Seurat::DimPlot(
    cluster_clean,
    group.by = "cluster_solo",
    repel = TRUE,
    label = TRUE,
    cols = c(
        "cluster_0" = "#00BC9F",
        "cluster_1" = "#E68613",
        "cluster_2" = "#7CAE00",
        "cluster_3" = "#00A9FF",
        "cluster_4" = "#F8766D",
        "cluster_5" = "#B2A7B8",
        "cluster_6" = "#C77CFF",
        "cluster_7" = "#72464B",
        "missing cell" = "#E6E6FA"
))
dev.off()
rm(list = ls()[sapply(ls(), function(x) str_starts(x, "solo_"))])
# DoubletFinder
df_path <- "internal_outputs/doubletfinder_result/scPred_pbmc_1"
df_cluster <- readRDS(paste(df_path, "cluster.rds", sep = "/"))
df_cluster_name <- as.vector(df_cluster@meta.data$seurat_clusters)
df_cluster_name <- sapply(
    df_cluster_name,
    function(x) paste("cluster", x, sep = "_"))
names(df_cluster_name) <- df_cluster@assays$RNA@counts@Dimnames[[2]]
df_empty_type <- empty_cell_type
df_cell_name <- intersect(
    df_cluster@assays$RNA@counts@Dimnames[[2]],
    sim_cell_name)
df_empty_type[df_cell_name] <- df_cluster_name[df_cell_name]
cluster_clean@meta.data$cluster_df <- df_empty_type
pdf(file = "materials/figures/Figure.7-A-DoubletFinder.pdf",
    width = 8,
    height = 6,
    title = "DoubletFinder")
Seurat::DimPlot(
    cluster_clean,
    group.by = "cluster_df",
    repel = TRUE,
    label = TRUE,
    cols = c(
        "cluster_0" = "#00BC9F",
        "cluster_1" = "#E68613",
        "cluster_2" = "#7CAE00",
        "cluster_3" = "#DC143C",
        "cluster_4" = "#4169E1",
        "cluster_5" = "#00A9FF",
        "cluster_6" = "#F8766D",
        "cluster_7" = "#B2A7B8",
        "cluster_8" = "#C77CFF",
        "cluster_9" = "#72464B",
        "missing cell" = "#E6E6FA"
))
dev.off()
rm(list = ls()[sapply(ls(), function(x) str_starts(x, "df_"))])

# scDblFinder
df_path <- "internal_outputs/scDblFinder_result/scPred_pbmc_1"
df_cluster <- readRDS(paste(df_path, "cluster.rds", sep = "/"))
df_cluster_name <- as.vector(df_cluster@meta.data$seurat_clusters)
df_cluster_name <- sapply(
  df_cluster_name,
  function(x) paste("cluster", x, sep = "_"))
names(df_cluster_name) <- df_cluster@assays$RNA@counts@Dimnames[[2]]
df_empty_type <- empty_cell_type
df_cell_name <- intersect(
  df_cluster@assays$RNA@counts@Dimnames[[2]],
  sim_cell_name)
df_empty_type[df_cell_name] <- df_cluster_name[df_cell_name]
cluster_clean@meta.data$cluster_df <- df_empty_type
pdf(file = "materials/figures/Figure.7-A-scDblFinder.pdf",
    width = 8,
    height = 6,
    title = "scDblFinder")
Seurat::DimPlot(
  cluster_clean,
  group.by = "cluster_df",
  repel = TRUE,
  label = TRUE,
  cols = c(
    "cluster_0" = "#E68613",
    "cluster_1" = "#7CAE00",
    "cluster_2" = "#72464B",
    "cluster_3" = "#DC143C",
    "cluster_4" = "#00A9FF",
    "cluster_5" = "#F8766D",
    "cluster_6" = "#B2A7B8",
    "cluster_7" = "#C77CFF",
    "cluster_8" = "#72464B",
    "missing cell" = "#E6E6FA"
  ))
dev.off()
rm(list = ls()[sapply(ls(), function(x) str_starts(x, "df_"))])


# dirty
df_path <- "internal_outputs/scPred_pbmc_1"
df_cluster <- readRDS(paste(df_path, "dir_cluster.rds", sep = "/"))
df_cluster_name <- as.vector(df_cluster@meta.data$seurat_clusters)
df_cluster_name <- sapply(
  df_cluster_name,
  function(x) paste("cluster", x, sep = "_"))
names(df_cluster_name) <- df_cluster@assays$RNA@counts@Dimnames[[2]]
df_empty_type <- empty_cell_type
df_cell_name <- intersect(
  df_cluster@assays$RNA@counts@Dimnames[[2]],
  sim_cell_name)
df_empty_type[df_cell_name] <- df_cluster_name[df_cell_name]
cluster_clean@meta.data$cluster_df <- df_empty_type
pdf(file = "materials/figures/Figure.7-A-Negative-Control.pdf",
    width = 8,
    height = 6,
    title = "Negative")
Seurat::DimPlot(
  cluster_clean,
  group.by = "cluster_df",
  repel = TRUE,
  label = TRUE,
  # cols = c(
  #   "cluster_0" = "#E68613",
  #   "cluster_1" = "#7CAE00",
  #   #"cluster_2" = "#7CAE00",
  #   "cluster_3" = "#F8766D",
  #   #"cluster_4" = "#4169E1",
  #   #"cluster_5" = "#00A9FF",
  #   "cluster_6" = "#00A9FF",
  #   #"cluster_7" = "#B2A7B8",
  #   "cluster_8" = "#B2A7B8",
  #   #"cluster_9" = "#72464B",
  #   "cluster_10" = "",
  #   "cluster"
  #   "missing cell" = "#E6E6FA"
  # )
  )
dev.off()
rm(list = ls()[sapply(ls(), function(x) str_starts(x, "df_"))])
