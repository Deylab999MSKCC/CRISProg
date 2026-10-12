#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# AUCell scoring (Aibar et al. 2017, Bioconductor package AUCell), called by run.py.
#
# Input directory (written by perturb_programs/aucell.py):
#   matrix.h5     data / indices / indptr / shape of the expression matrix as
#                 genes x cells CSC (= the cells x genes CSR of the h5ad .X)
#   genes.txt     one gene per line (matrix rows)
#   cells.txt     one cell per line (matrix columns)
#   programs.tsv  program <TAB> gene, measured genes only
#
# Output: --out TSV, one row per cell: cell <TAB> program1 <TAB> program2 ...
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(AUCell); library(Matrix); library(rhdf5)
})

parse_args <- function(defaults) {
  a <- commandArgs(trailingOnly = TRUE)
  i <- 1
  while (i <= length(a)) {
    k <- sub("^--", "", a[i])
    if (!nzchar(k) || identical(k, a[i])) stop("expected --flag, got: ", a[i])
    if (i == length(a)) stop("--", k, " needs a value")
    defaults[[gsub("-", "_", k)]] <- a[i + 1]
    i <- i + 2
  }
  defaults
}

opt <- parse_args(list(input = "", out = "", cores = "1", max_rank = "", seed = "1", rankings = ""))
if (!nzchar(opt$input) || !nzchar(opt$out)) stop("--input and --out are required")
cores <- as.integer(opt$cores)
msg <- function(...) cat(format(Sys.time(), "[%H:%M:%S] "), "[R] ", ..., "\n", sep = "")
set.seed(as.integer(opt$seed))   # AUCell breaks ties in the cell rankings at random

# ---- expression (genes x cells) ----
h5 <- file.path(opt$input, "matrix.h5")
shp <- as.integer(h5read(h5, "shape"))
mat <- sparseMatrix(i = as.integer(h5read(h5, "indices")), p = as.integer(h5read(h5, "indptr")),
                    x = as.numeric(h5read(h5, "data")), dims = shp, index1 = FALSE)
H5close()
rownames(mat) <- readLines(file.path(opt$input, "genes.txt"))
colnames(mat) <- readLines(file.path(opt$input, "cells.txt"))
msg("expression: ", nrow(mat), " genes x ", ncol(mat), " cells")

prog <- read.delim(file.path(opt$input, "programs.tsv"), header = TRUE, stringsAsFactors = FALSE,
                   quote = "")
gene_sets <- lapply(split(prog$gene, prog$program), unique)
msg("programs: ", length(gene_sets))

# ---- rankings (independent of the programs; optionally cached) ----
if (nzchar(opt$rankings) && file.exists(opt$rankings)) {
  msg("loading cached rankings <- ", opt$rankings)
  rankings <- readRDS(opt$rankings)
  if (!identical(colnames(rankings), colnames(mat)) || !identical(rownames(rankings), rownames(mat)))
    stop("cached rankings do not match these cells/genes; delete ", opt$rankings, " to rebuild")
} else {
  msg("building cell rankings")
  # argument names differ across AUCell versions: use whatever this installation accepts
  fb <- names(formals(AUCell_buildRankings))
  bargs <- list(exprMat = mat, plotStats = FALSE, verbose = FALSE)
  if ("BPPARAM" %in% fb) bargs$BPPARAM <- BiocParallel::MulticoreParam(cores)
  else if ("nCores" %in% fb) bargs$nCores <- cores
  if ("splitByBlocks" %in% fb) bargs$splitByBlocks <- TRUE   # lower peak memory
  rankings <- do.call(AUCell_buildRankings, bargs)
  if (nzchar(opt$rankings)) {
    dir.create(dirname(opt$rankings), recursive = TRUE, showWarnings = FALSE)
    saveRDS(rankings, opt$rankings, compress = FALSE)
    msg("cached rankings -> ", opt$rankings)
  }
}
rm(mat); invisible(gc())

# ---- AUC ----
fc <- names(formals(AUCell_calcAUC))
aargs <- list(geneSets = gene_sets, rankings = rankings, verbose = FALSE)
if (nzchar(opt$max_rank)) {
  mr <- as.numeric(opt$max_rank)
  aargs$aucMaxRank <- if (mr < 1) ceiling(mr * nrow(rankings)) else as.integer(mr)
}
if ("nCores" %in% fc) aargs$nCores <- cores else if ("BPPARAM" %in% fc)
  aargs$BPPARAM <- BiocParallel::MulticoreParam(cores)
msg("AUCell_calcAUC, aucMaxRank = ",
    if (is.null(aargs$aucMaxRank)) paste0(ceiling(0.05 * nrow(rankings)), " (AUCell default, 5% of genes)")
    else aargs$aucMaxRank)
auc <- getAUC(do.call(AUCell_calcAUC, aargs))     # programs x cells

out <- data.frame(cell = colnames(auc), t(auc), check.names = FALSE)
write.table(out, opt$out, sep = "\t", quote = FALSE, row.names = FALSE)
msg("wrote ", ncol(auc), " cells x ", nrow(auc), " programs -> ", opt$out)
