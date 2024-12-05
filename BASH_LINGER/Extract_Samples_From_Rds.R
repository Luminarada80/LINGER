library(Seurat)
library(data.table)
library(Matrix)
library(tools)

karamveer_dir = "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.KARAMVEER/Macrophase_data/STABILITY_ANALYSIS"

outdir = "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MACROPHAGE_STABILITY"

# Look through each subdirectory
sample_dirs <- list.dirs(karamveer_dir, full.names = TRUE, recursive = FALSE)

# Iterate through each subdirectory
for (subsample_dir in sample_dirs) {
  
  # Find a list of RDS files
  cat("Processing directory: ", basename(subsample_dir), "\n")
  rds_files <- list.files(subsample_dir, pattern = "\\.rds$", full.names = TRUE)
  
  for (rds_file in rds_files) {
    cat("\tProcessing file: ", basename(rds_file), "\n")
    
    # Load the dataset
    dataset <- readRDS(rds_file)
    
    base_filename <- file_path_sans_ext(basename(rds_file))

    
    # IF THE FILE IS ONLY ATAC DATA
    if (grepl("ATAC", basename(rds_file), ignore.case = TRUE)) {

      cat("\t\tDetected ATAC data\n")
      
      # Check if the dataset is a list
      if (is.list(dataset)) {
        cat("\t\tDetected multiple data frames in dataset\n")
      
        # Process each matrix in the list
        for (df_name in names(dataset)) {
          cat("\t\tProcessing data frame: ", df_name, "\n")
          
          # Extract the data frame
          df <- dataset[[df_name]]
          
          # Define output file name
          output_file <- file.path(outdir, paste0(base_filename, "_", df_name, ".csv"))
          
          # Write data frame to CSV
          fwrite(df, file = output_file, row.names = FALSE, col.names = TRUE)
          }
      
      } else {  
        
        atac_counts <- GetAssayData(dataset, assay = "peaks", slot = "counts")
        atac_df <- as.data.frame(as.matrix(atac_counts))
        rownames(atac_df) <- rownames(atac_counts)
        colnames(atac_df) <- colnames(atac_counts)

        atac_output_file <- file.path(outdir, paste0(base_filename, "_ATAC.csv"))
        
        fwrite(atac_df, file = atac_output_file, row.names = TRUE, col.names = TRUE)
        cat("\t\tSaved ATAC counts data\n")
      }
    
    # IF THE DATA IS ONLY RNA DATA
    } else if (grepl("RNA", basename(rds_file), ignore.case = TRUE)) {

      cat("\t\tDetected RNA data\n")
      
      # Check if the dataset is a list
      if (is.list(dataset)) {
        cat("\t\tDetected multiple dataframes in dataset\n")
        
        # Process each matrix in the list
        for (df_name in names(dataset)) {
          cat("\t\tProcessing data frame: ", df_name, "\n")
          
          # Extract the data frame
          df <- dataset[[df_name]]
          
          # Define output file name
          output_file <- file.path(outdir, paste0(base_filename, "_", df_name, ".csv"))
          
          # Write data frame to CSV
          fwrite(df, file = output_file, row.names = FALSE, col.names = TRUE)
          }
      } else { 
        
        rna_counts <- GetAssayData(dataset, assay = "RNA", slot = "counts")
        rna_df <- as.data.frame(as.matrix(rna_counts))
        rownames(rna_df) <- rownames(rna_counts)
        colnames(rna_df) <- colnames(rna_counts)
        
        rna_output_file <- file.path(outdir, paste0(base_filename, "_RNA.csv"))
        
        fwrite(rna_df, file = rna_output_file, row.names = TRUE, col.names = TRUE)
        cat("\t\tSaved RNA counts data\n")
      }

    # IF ITS COMBINED DATA  
    } else {
      cat("\t\tProcessing combined RNA and ATAC data\n")
      
      rna_counts <- GetAssayData(dataset, assay = "RNA", slot = "counts")
      atac_counts <- GetAssayData(dataset, assay = "peaks", slot = "counts")
      
      rna_output_file <- file.path(outdir, paste0(base_filename, "_RNA.csv"))
      atac_output_file <- file.path(outdir, paste0(base_filename, "_ATAC.csv"))
      
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
    }
    
    # Remove the dataset to clear the cache
    rm(dataset)
    gc()
  }
}





