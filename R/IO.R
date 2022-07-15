# This file was created to convert R object
library(rhdf5)
library(progress)
library(reticulate)
library(Matrix) # For sparse matrix operation
pd <- import("pandas")
cls <- c("singlet", "doublet")
# Save dgCMatrix object as HDF5 format
dgCMatrix2Hdf <- function(dgcmatrix, path, label = NULL) {
    h5file <- paste(path, "[csc_sparse_matrix].h5", sep = "")
    h5createFile(h5file)
    h5write(dgcmatrix@Dimnames[[1]], h5file, name = "/rowname")
    h5write(dgcmatrix@Dimnames[[2]], h5file, name = "/colname")
    h5write(dgcmatrix@i, h5file, name = "/indices")
    h5write(dgcmatrix@p, h5file, name = "/indptr")
    h5write(dgcmatrix@x, h5file, name = "/data")
    h5write(dgcmatrix@Dim, h5file, name = "/shape")
    h5write("csc", h5file, name = "/format")
    if (class(label) == "character") {
        h5write(label, h5file, name = "/label")
    }
    if (class(label) == "numeric") {
        h5write(sapply(label, function(i) cls[i + 1]), h5file, name = "/label")
    }
    h5closeAll()
}
# Convert matrix to pandas DataFrame and saved as HDF5 format
matrix2pandas <- function(mat, path) {
    df <- pd$DataFrame(mat, columns = colnames(mat), index = row.names(mat))
    pd$DataFrame$to_hdf(df, path, mode = "w", key = "data")
    return(df)
}
# Convert matrix to anndata object and saved as h5ad format
matrix2h5ad <- function(mat, path, label = NULL, transpose = FALSE, labelType = NULL) {
    sc <- import("scanpy")
    if (transpose) {
        mat <- t(mat)
    }
    anaData <- sc$AnnData(
        mat,
        var = pd$DataFrame(index = colnames(mat))
    )
    if (!is.null(label)) {
        anaData$obs <- pd$DataFrame(
            label,
            index = row.names(mat),
            columns = list(c("type")),
            dtype = labelType
        )
    }
    sc$write(path, anaData)
    return(anaData)
}
h5ad2List <- function(file) {
    sc <- import("scanpy")
    data <- sc$read_h5ad(file)
    sparse_data <- as(data$X, "dgCMatrix")
    sparse_data@Dimnames[[1]] <- data$obs_names$to_list()
    sparse_data@Dimnames[[2]] <- data$var_names$to_list()

    sparse_data <- t(sparse_data)
    label <- data$obs[["type"]]
    return(list(counts = sparse_data, labels = label))
}