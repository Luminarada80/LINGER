library(Seurat)
library(dplyr)
library(patchwork)

scRNA_data <- readRDS("/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_SC_DATA/scRNA.Seurat.rds")

scRNA_data

# Find the number of genes per cell, 
VlnPlot(scRNA_data, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), ncol = 3)


# nFeature_RNA - number of genes detected in each chell
# nCount_RNA: number of molecules detected within a cell
# percent.mt: percent of genes that are mitochondrial
mesc <- subset(
  scRNA_data,
  subset = nFeature_RNA > 200 & nCount_RNA < 2500 & percent.mt < 5
  )