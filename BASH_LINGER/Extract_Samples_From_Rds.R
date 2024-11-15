library(Seurat)
library(data.table)
library(Matrix)

karamveer_dir = "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.KARAMVEER/CURRENT.DS14_mESC.FILTERED/"

outdir = "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_SC_DATA/FULL_MESC_SAMPLES/"

sample_dirs <- list.dirs(karamveer_dir, full.names = TRUE, recursive = FALSE)

# Iterate through each 
for (subsample_dir in sample_dirs) {
  
  cat("Processing directory: ", basename(subsample_dir), "\n")
  rds_files <- list.files(subsample_dir, pattern = "\\.rds$", full.names = TRUE)
  
  for (rds_file in rds_files) {
    cat("\tLoading: ", basename(rds_file), "\n")
    
    dataset <- readRDS(rds_file)

    rna_counts <- GetAssayData(dataset, assay = "RNA", slot = "counts")
    atac_counts <- GetAssayData(dataset, assay = "ATAC", slot = "counts")
    
    rna_output_file <- file.path(outdir, paste0(basename(rds_file), "_RNA.csv"))
    atac_output_file <- file.path(outdir, paste0(basename(rds_file), "_ATAC.csv"))
    
    rna_df <- as.data.frame(as.matrix(rna_counts))
    rownames(rna_df) <- rownames(rna_counts)
    colnames(rna_df) <- colnames(rna_counts)
    
    atac_df <- as.data.frame(as.matrix(atac_counts))
    rownames(atac_df) <- rownames(atac_counts)
    colnames(atac_df) <- colnames(atac_counts)
    
    fwrite(rna_df, file = rna_output_file, row.names = TRUE, col.names = TRUE)
    cat("\t\tSaved RNA counts data\n")
    
    fwrite(atac_df, file = atac_output_file, row.names = TRUE, col.names = TRUE)
    cat("\t\tSaved ATAC counts data\n")
    
    # Remove the dataset to clear the cache
    rm(dataset, rna_counts, atac_counts)
    
  }
}



