# This file was create to do single cell WGCNA
# https://www.jianshu.com/p/d2991fa79a3e?from=singlemessage
# https://github.com/CFeregrino/scWGCNA
# devtools::install_github("GCS-ZHN/scWGCNA", ref="main")
source("./R/Methods.R")
library(scWGCNA)
doWGCNA <- function(dataset, socube_id) {
    data <-
        readRDS(paste(
            "datasets/PMID33338399/real_datasets/",
            dataset,
            ".rds",
            sep = ""
        ))[[1]]
    gene <-
        read.csv(
            paste(
                "datasets",
                dataset,
                socube_id,
                "data[log+std+norm+umap+jv].csv",
                sep = "/"
            ),
            header = TRUE
        )$X
    data <- as.data.frame(data)
    data <- data[gene, ]
    data <- as(as.matrix(data), "dgCMatrix")
    data <- preprocess(data)
    data <-
        FindVariableFeatures(data, selection.method = "vst", nfeatures = 2000)
    data <- RunPCA(data)
    data <- RunUMAP(data, dims = 1:30)
    # pseudoCells
    data.pcells <- calculate.pseudocells(
        s.cells = data,
        dims = 1:2,
        reduction = "umap",
        # nn = 50,
        seeds = 0.05
    )
    # Run scWGCNA
    data.scWGCNA <- run.scWGCNA(
        p.cells = data.pcells,
        s.cells = data,
        is.pseudocell = T,
        features = rownames(data)
    )
    scW.p.dendro(scWGCNA.data = data.scWGCNA, tree = 4)
    result <- NULL
    for (idx in seq_len(length(data.scWGCNA$module.genes))) {
        tmp <-
            data.frame(module = sapply(
                data.scWGCNA$module.genes[[idx]],
                function(x) {
                    paste("module", idx, sep = "_")
                }
            ))
        if (is.null(result)) {
            result <- tmp
        } else {
            result <- rbind(result, tmp)
        }
        rm(tmp)
    }
    data.scWGCNA$result <- result
    return(data.scWGCNA)
}
# data = “pbmc-1C-dm"
# socube_id = "20220115-174928-203"
# data <- "HMEC-rep-MULTI"
# socube_id <- "20220114-193253-794"
data <- "pbmc-1A-dm"
socube_id <- "20220115-174928-553"
r <- doWGCNA(data, socube_id)
saveRDS(
    r,
    paste("outputs/", data, "/scWGCNA[", socube_id, "].rds", sep = "")
)
write.csv(
    r$result,
    file = paste(
        "outputs/",
        data,
        "/co-expression-gene[",
        socube_id,
        "].csv",
        sep = ""
    )
)