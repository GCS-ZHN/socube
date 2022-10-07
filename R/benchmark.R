data_path <- "internal_datasets/PMID33338399/real_datasets/"
datasets <- c(
  "pbmc-ch",
  "cline-ch",
  "HEK-HMEC-MULTI",
  "hm-12k",
  "hm-6k",
  "HMEC-orig-MULTI",
  "HMEC-rep-MULTI",
  "J293t-dm",
  "mkidney-ch",
  "nuc-MULTI",
  "pbmc-1A-dm",
  "pbmc-1B-dm",
  "pbmc-1C-dm",
  "pbmc-2ctrl-dm",
  "pbmc-2stim-dm",
  "pdx-MULTI"
)

library(scDblFinder)
predict <- function(data, path, methods = 1) {
  if (!dir.exists(path)) {
    dir.create(path)
  }
  score <- switch(
    methods,
    scDblFinder(data)$scDblFinder.score,
    preditctByDoubletFinder(data)
  )
  if (is.null(score)) {
    print("Invalid method options: ", methods)
  } else {
    write.csv(score, paste(path, "sorce.csv", sep = "/"))
  }
}

# Benchmark for scDblFinder
for (dataset_name in datasets) {
  cat(dataset_name, "\n")
  dataset <- readRDS(paste(data_path, dataset_name, ".rds", sep = ""))
  predict(
    dataset[[1]], 
    paste("internal_outputs/scDblFinder_result/", dataset_name, sep = ""),
    methods = 1)
}

# Benchmark for DoubletFinder
source("public/R/Methods.R")
for (dataset_name in datasets) {
  cat(dataset_name, "\n")
  dataset <- readRDS(paste(data_path, dataset_name, ".rds", sep = ""))
  predict(
    dataset[[1]], 
    paste("internal_outputs/doubletfinder_result/", dataset_name, sep = ""),
    methods = 2)
  rm(dataset)
  gc()
}
