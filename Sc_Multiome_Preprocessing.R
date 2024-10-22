library(Seurat)
library(dplyr)
library(patchwork)
library(SummarizedExperiment)
library(SingleCellExperiment)
library(tidyverse)

data_dir <- '/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_SC_DATA/'

# Load scRNA-seq and scATAC-seq data using correct path
scRNA_data <- readRDS(paste0(data_dir, '/scRNA.Seurat.rds'))
scATAC_data_ranged_summarized <- readRDS(paste0(data_dir, '/scATAC.PeakMatrix_summarized_experiment.rds'))

# Extract the peak matrix and meta-data (col_data)
peak_matrix <- assays(scATAC_data_ranged_summarized)$PeakMatrix
col_data <- as.data.frame(colData(scATAC_data_ranged_summarized))

# Use the peak matrix and meta data to create a Seurat object
scATAC_seurat <- CreateSeuratObject(counts = peak_matrix, assay = "ATAC")
scATAC_seurat <- AddMetaData(object = scATAC_seurat, metadata = col_data)

# Create a "cell" column in the scATACseq meta data that matches the scRNAseq format
meta_data <- scATAC_seurat@meta.data
meta_data$cell <- paste(meta_data$sample, meta_data$barcode, sep = "#")
scATAC_seurat@meta.data <- meta_data

# Find shared barcodes between scRNA and scATAC based on the "cell" identities
shared_barcodes <- intersect(scRNA_data@meta.data$cell, scATAC_seurat@meta.data$cell)

# The merged object will contain both RNA and ATAC data in separate assays
multiomic_data <- merge(scATAC_seurat, y = scRNA_data, add.cell.ids = c("ATAC", "RNA"))

# Subset the data to only contain E7.5 replicate 1
multiomic_data_E7.5 <- subset(multiomic_data, subset = sample == "E7.5_rep1")

view(str(multiomic_data_E7.5))

# Plot a violin plot of the QC metrics
VlnPlot(multiomic_data_E7.5,
        features = c(
          "nFeature_RNA", # Number of genes per cell
          "nCount_RNA", # Number of RNA per cell
          "mitochondrial_percent_RNA", # Percent mitochondrial
          "nFeature_ATAC", # Number of peaks
          "TSSEnrichment", # Enrichment of sequencing reads near TSS
          "FRIP" # Fraction of reads in peaks
        ),
        ncol = 3,
        pt.size = 0.15) & xlab("")

# Filter the dataset
multiomic_data_E7.5_filtered <- subset(
  multiomic_data_E7.5,
  subset = nFeature_RNA > 1000 &
    nFeature_RNA < 7500 &
    mitochondrial_percent_RNA < 30 &
    nFeature_ATAC > 1000 &
    nFeature_ATAC < 30000 &
    TSSEnrichment > 1)

# Plot a violin plot of the QC metrics
VlnPlot(multiomic_data_E7.5_filtered, 
        features = c(
          "nFeature_RNA", # Number of genes per cell
          "nCount_RNA", # Number of RNA per cell
          "mitochondrial_percent_RNA", # Percent mitochondrial
          "nFeature_ATAC", # Number of peaks
          "TSSEnrichment", # Enrichment of sequencing reads near TSS
          "FRIP" # Fraction of reads in peaks
        ),
        ncol = 3,
        pt.size = 0.15) & xlab("")
