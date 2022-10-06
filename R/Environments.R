M# This file was created to install all require R packages

dependencies <- list(
    "official" = c(
        "BiocManager",
        "devtools",
        "mclust",
        "dplyr",
        "stringr"),
    "bioc" = c(
        "SingleCellExperiment",
        "Seurat",
        "slingshot",
        "AnnotationDbi",
        "GO.db",
        "preprocessCore",
        "impute",
        "limma",
        "scran",
        "MAST",
        "gam",
        "TSCAN"),
    "github" = c(
        "chris-mcginnis-ucsf/DoubletFinder",
        "GCS-ZHN/scWGCNA")
)
for (package in dependencies$official) {
    if (!require(package, quietly = TRUE)) {
        install.packages(package)
    }
}
for (package in dependencies$github) {
    if (!require(package, quietly = TRUE)) {
        devtools::install_github(package)
    }
}
for (package in dependencies$bioc) {
    if (!require(package, quietly = TRUE)) {
        BiocManager::install(package, version = "3.14")
    }
}
rm(package, dependencies)

