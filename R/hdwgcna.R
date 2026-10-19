#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# hdWGCNA co-expression modules (Morabito et al. 2023, Cell Rep. Methods), called by crisprog/discover.py.
#
# Input directory (written by crisprog/discover.py):
#   matrix.h5     data / indices / indptr / shape of the raw counts as genes x cells CSC
#   genes.txt     one gene per line (matrix rows)
#   cells.txt     one cell per line (matrix columns)
#   pca.tsv       cells x PCs embedding (the same PCA Hotspot uses); metacells are built on it
#   groups.txt    optional, one group label per cell: metacells never mix groups (e.g. samples or time points)
#
# Steps: genes expressed in >= --fraction of cells -> metacells (k nearest neighbours on the PCA, within
# groups) -> normalize -> soft power (lowest power reaching scale-free fit 0.8 unless --soft-power) -> signed
# network and modules (dynamic tree cut) -> module eigengenes -> kME.
#
# Output: --out TSV with gene, module, kME (connectivity to its own module); grey (unassigned) genes dropped.
#         <out>_soft_powers.tsv: the scale-free fit per power (check that the chosen power is sensible).
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(Matrix); library(rhdf5); library(Seurat); library(WGCNA); library(hdWGCNA)
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

opt <- parse_args(list(input = "", out = "", threads = "1", seed = "0", fraction = "0.05", k = "25",
                       max_shared = "10", min_cells = "100", soft_power = "", min_module_size = "50",
                       deep_split = "4", merge_cut_height = "0.2"))
if (!nzchar(opt$input) || !nzchar(opt$out)) stop("--input and --out are required")
msg <- function(...) cat(format(Sys.time(), "[%H:%M:%S] "), "[R] ", ..., "\n", sep = "")
set.seed(as.integer(opt$seed))
threads <- as.integer(opt$threads)
if (threads > 1) enableWGCNAThreads(nThreads = threads) else disableWGCNAThreads()

# ---- counts (genes x cells), embedding, groups ----
h5 <- file.path(opt$input, "matrix.h5")
shp <- as.integer(h5read(h5, "shape"))
counts <- sparseMatrix(i = as.integer(h5read(h5, "indices")), p = as.integer(h5read(h5, "indptr")),
                       x = as.numeric(h5read(h5, "data")), dims = shp, index1 = FALSE)
H5close()
rownames(counts) <- make.unique(readLines(file.path(opt$input, "genes.txt")))
colnames(counts) <- readLines(file.path(opt$input, "cells.txt"))
pca <- as.matrix(read.delim(file.path(opt$input, "pca.tsv"), row.names = 1, check.names = FALSE))
colnames(pca) <- paste0("PC_", seq_len(ncol(pca)))
gfile <- file.path(opt$input, "groups.txt")
groups <- if (file.exists(gfile)) readLines(gfile) else rep("all", ncol(counts))
msg("counts: ", nrow(counts), " genes x ", ncol(counts), " cells; ", length(unique(groups)), " group(s)")

so <- CreateSeuratObject(counts = counts, min.cells = 0, min.features = 0)
so$group <- groups
so <- NormalizeData(so, verbose = FALSE)
so[["pca"]] <- CreateDimReducObject(embeddings = pca[colnames(so), , drop = FALSE], key = "PC_", assay = "RNA")

# ---- hdWGCNA ----
so <- SetupForWGCNA(so, gene_select = "fraction", fraction = as.numeric(opt$fraction), wgcna_name = "crisprog")
msg("genes expressed in >= ", opt$fraction, " of cells: ", length(GetWGCNAGenes(so)))
keep <- names(which(table(groups) >= as.integer(opt$min_cells)))
if (!length(keep)) stop("no group has >= ", opt$min_cells, " cells for metacells")
so <- MetacellsByGroups(so, group.by = "group", reduction = "pca", k = as.integer(opt$k),
                        max_shared = as.integer(opt$max_shared), ident.group = "group",
                        min_cells = as.integer(opt$min_cells), verbose = FALSE)
so <- NormalizeMetacells(so)
msg("metacells: ", ncol(GetMetacellObject(so)))
so <- SetDatExpr(so, group_name = keep, group.by = "group", assay = "RNA", layer = "data")
invisible(capture.output(so <- TestSoftPowers(so, networkType = "signed")))
pt <- GetPowerTable(so)
write.table(pt, sub("\\.tsv$", "_soft_powers.tsv", opt$out), sep = "\t", quote = FALSE, row.names = FALSE)
power <- if (nzchar(opt$soft_power)) as.integer(opt$soft_power) else {
  ok <- pt$Power[pt$SFT.R.sq >= 0.8]
  if (length(ok)) min(ok) else { msg("WARNING: no soft power reaches scale-free fit 0.8; using the best fit")
                                 pt$Power[which.max(pt$SFT.R.sq)] }
}
msg("soft power: ", power, " (scale-free fit R^2 = ", round(pt$SFT.R.sq[pt$Power == power], 3), ")")
tom_dir <- file.path(opt$input, "TOM")
so <- ConstructNetwork(so, soft_power = power, networkType = "signed", tom_name = "crisprog",
                       tom_outdir = tom_dir, overwrite_tom = TRUE, minModuleSize = as.integer(opt$min_module_size),
                       deepSplit = as.integer(opt$deep_split), mergeCutHeight = as.numeric(opt$merge_cut_height),
                       verbose = 0)
invisible(capture.output(so <- suppressWarnings(ModuleEigengenes(so, verbose = FALSE))))
invisible(capture.output(so <- ModuleConnectivity(so, group.by = "group", group_name = keep)))
m <- GetModules(so)
m <- m[as.character(m$module) != "grey", ]
own <- vapply(seq_len(nrow(m)), function(i) m[i, paste0("kME_", m$module[i])], numeric(1))
res <- data.frame(gene = m$gene_name, module = as.character(m$module), kME = own)
res <- res[order(res$module, -res$kME), ]
write.table(res, opt$out, sep = "\t", quote = FALSE, row.names = FALSE)
msg(length(unique(res$module)), " modules, ", nrow(res), " genes assigned, ",
    length(GetWGCNAGenes(so)) - nrow(res), " unassigned (grey)")
