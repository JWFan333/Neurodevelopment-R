
# Load required packages
library(limma)
library(edgeR)

# ============================================================
# User configuration
# ============================================================
data_dir <- "data"
output_dir <- "output"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# ============================================================
# 1. Read count data
# ============================================================
exp <- read.table(file.path(data_dir, "gene_count.txt"), sep = "\t", header = TRUE)
exp <- exp[!duplicated(exp[, 1]), ]  # Remove duplicate gene names
rownames(exp) <- exp[, 1]
exp <- exp[, -1]  # Remove first column (gene name column)

# ============================================================
# 2. Prepare group information
# ============================================================
group <- data.frame(
  Type = substr(colnames(exp), 1, 2),
  row.names = colnames(exp)
)
group$Type <- as.factor(group$Type)

# ============================================================
# 3. Design matrix
# ============================================================
design <- model.matrix(~0 + group$Type)
# Adjust column names according to your group levels
colnames(design) <- c("HC", "SZ")
rownames(design) <- colnames(exp)

# ============================================================
# 4. Filter low-expression genes using CPM
# ============================================================
cpmMatrix <- cpm(exp)
cutoff <- 1
keep <- rowSums(cpmMatrix > cutoff) >= 1
filteredCountsMatrix <- exp[keep, ]

# ============================================================
# 5. Differential expression analysis (for raw count)
# ============================================================
dge <- DGEList(counts = filteredCountsMatrix)
dge <- calcNormFactors(dge)
v <- voom(dge, design, normalize = "quantile")
cont.matrix <- makeContrasts("SZ-HC", levels = design)

fit <- lmFit(v, design)
fit2 <- contrasts.fit(fit, cont.matrix)
fit2 <- eBayes(fit2)
DEG <- topTable(fit2, coef = 1, n = Inf)
DEG <- na.omit(DEG)

# ============================================================
# 6. Mark up/down regulated genes with logFC cutoff
# ============================================================
logFC_cutoff <- mean(abs(DEG$logFC)) + 2 * sd(abs(DEG$logFC))
k1 <- (DEG$adj.P.Val < 0.05) & (DEG$logFC < -logFC_cutoff)  # Significantly down
k2 <- (DEG$adj.P.Val < 0.05) & (DEG$logFC > logFC_cutoff)   # Significantly up
DEG$change <- ifelse(k1, "DOWN", ifelse(k2, "UP", "NOT"))

# ============================================================
# 7. Select significant DEGs
# ============================================================
DEG1 <- DEG[which(DEG$change != "NOT"), ]   # All significant DEGs
DEG2 <- DEG1[which(DEG1$change != "UP"), ]  # Down-regulated DEGs
DEG3 <- DEG1[which(DEG1$change != "DOWN"), ] # Up-regulated DEGs

# Extract expression data for significant genes
aim <- row.names(DEG1)
exp_sig <- exp[aim, ]
exp_sig <- as.data.frame(exp_sig)

# ============================================================
# 8. Save results
# ============================================================
write.table(DEG, file = file.path(output_dir, "DEG_all_gene.txt"),
            sep = "\t", row.names = TRUE, col.names = NA, quote = FALSE)
write.table(DEG1, file = file.path(output_dir, "significant_all_DEG.txt"),
            sep = "\t", row.names = TRUE, col.names = NA, quote = FALSE)
write.table(DEG2, file = file.path(output_dir, "significant_down_DEG.txt"),
            sep = "\t", row.names = TRUE, col.names = NA, quote = FALSE)
write.table(DEG3, file = file.path(output_dir, "significant_up_DEG.txt"),
            sep = "\t", row.names = TRUE, col.names = NA, quote = FALSE)