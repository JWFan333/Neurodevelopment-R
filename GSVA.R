
# Load required packages
library(GSVA)
library(GSEABase)

# ============================================================
# User configuration
# ============================================================
data_dir <- "data"
gene_set_dir <- file.path(data_dir, "gene_sets")
output_dir <- file.path("output", "GSVA_results")

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# ============================================================
# 1. Define gene set names and corresponding intersection files
# ============================================================
# For public sharing, generic names are used here.
# Replace the file names with your own local file names if needed.
gene_set_files <- c(
  "GeneSet_01" = "intersection_01.csv",
  "GeneSet_02" = "intersection_02.csv",
  "GeneSet_03" = "intersection_03.csv",
  "GeneSet_04" = "intersection_04.csv",
  "GeneSet_05" = "intersection_05.csv",
)

# Read intersection gene lists
gene_sets <- lapply(gene_set_files, function(f) {
  df <- read.csv(file.path(intersection_dir, f))
  df$x
})
names(gene_sets) <- names(gene_set_files)

# ============================================================
# 2. Load expression matrix and group information
# ============================================================
load(file.path(data_dir, "exp.Rdata"))
load(file.path(data_dir, "group.Rdata"))

exp <- as.matrix(exp)

group <- as.data.frame(group, row.names = colnames(exp))
colnames(group) <- "Type"

# ============================================================
# 3. Run GSVA for each gene set
# ============================================================
for (gs_name in names(gene_sets)) {
  current_genes <- gene_sets[[gs_name]]
  
  # Find genes present in the expression matrix
  available_genes <- intersect(current_genes, rownames(exp))
  missing_genes <- setdiff(current_genes, rownames(exp))
  
  cat("\nProcessing gene set:", gs_name, "\n")
  cat("Total genes in set:", length(current_genes), "\n")
  cat("Available genes:", length(available_genes), "\n")
  
  if (length(missing_genes) > 0) {
    cat("Missing genes:", paste(missing_genes, collapse = ", "), "\n")
  }
  
  # Skip if no available genes
  if (length(available_genes) < 1) {
    cat("Warning: No available genes for", gs_name, "- skipping\n")
    next
  }
  
  # Create GeneSet object for the current gene set
  current_gset <- GeneSet(available_genes, setName = gs_name)
  geneSets <- GeneSetCollection(current_gset)
  
  # Create GSVA parameter object
  param <- gsvaParam(
    exprData = exp,
    geneSets = geneSets,
    minSize = 1,
    maxSize = Inf,
    checkNA = "auto",
    use = "everything",
    kcdf = "Gaussian"#raw count choose "Poisson"
  )
  
  # Run GSVA
  gsva_res <- gsva(param = param, verbose = FALSE)
  
  # Format results
  gsva_df <- as.data.frame(t(gsva_res))
  gsva_df$Group <- group$Type
  colnames(gsva_df) <- c("Score", "Group")
  
  # Generate a safe file name
  file_name <- gsub("[^[:alnum:]]", "_", gs_name)
  file_name <- gsub("_+", "_", file_name)
  file_name <- paste0("gsva_", file_name, ".csv")
  file_path <- file.path(output_dir, file_name)
  
  # Save results
  write.csv(gsva_df, file = file_path, row.names = FALSE)
  cat("Results saved to:", file_path, "\n")
}

# ============================================================
# 4. Summary
# ============================================================
cat("\nAnalysis completed.\n")
cat("Total gene sets processed:", length(gene_sets), "\n")
cat("Gene sets successfully analyzed:",
    length(list.files(output_dir, pattern = "\\.csv$")), "\n")

if (length(list.files(output_dir, pattern = "\\.csv$")) < length(gene_sets)) {
  cat("Warning: Some gene sets were skipped due to insufficient available genes.\n")
}