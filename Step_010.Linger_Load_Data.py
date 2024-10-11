import os
import scanpy as sc
import scipy
import pandas as pd
from linger.preprocess import *
from linger.pseudo_bulk import *
import warnings

import shared_variables

# Filter warnings about copying objects from AnnData
warnings.filterwarnings("ignore", message="Received a view of an AnnData. Making a copy.")
warnings.filterwarnings("ignore", message="Trying to modify attribute `.obs` of view, initializing view as actual.")

# Specify the method
method='LINGER'

# ----- THIS PART DIFFERS BETWEEN PBMC AND H1 -----
print('\tReading in cell labels...')
# Read in the data files
matrix=scipy.io.mmread(shared_variables.matrix_path)
features=pd.read_csv(shared_variables.features_path,sep='\t',header=None)
barcodes=pd.read_csv(shared_variables.barcodes_path,sep='\t',header=None)
label=pd.read_csv(shared_variables.label_path,sep='\t',header=0)
# ---------------------------------------------------

print('\nExtracting the adata RNA and ATAC seq data...')
# Create AnnData objects for the scRNA-seq and scATAC-seq datasets
adata_RNA, adata_ATAC = get_adata(matrix, features, barcodes, label)  # adata_RNA and adata_ATAC are scRNA and scATAC

print(f'\tscRNAseq Dataset: {adata_RNA.shape[0]} genes, {adata_RNA.shape[1]} cells')
print(f'\tscATACseq Dataset: {adata_ATAC.shape[0]} peaks, {adata_ATAC.shape[1]} cells')

# Remove low counts cells and genes
print('\nFiltering Data')
print(f'\tFiltering out cells with less than 200 genes...')
sc.pp.filter_cells(adata_RNA, min_genes=200)
adata_RNA = adata_RNA.copy()  # Ensure adata_RNA is not a view after filtering
print(f'\t\tShape of the RNA dataset = {adata_RNA.shape[0]} genes, {adata_RNA.shape[1]} cells')

print(f'\tFiltering out genes expressed in fewer than 3 cells...')
sc.pp.filter_genes(adata_RNA, min_cells=3)
adata_RNA = adata_RNA.copy()  # Ensure adata_RNA is not a view after filtering
print(f'\t\tShape of the RNA dataset = {adata_RNA.shape[0]} genes, {adata_RNA.shape[1]} cells')

print(f'\tFiltering out cells with less than 200 ATAC-seq peaks...')
sc.pp.filter_cells(adata_ATAC, min_genes=200)
adata_ATAC = adata_ATAC.copy()  # Ensure adata_ATAC is not a view after filtering
print(f'\t\tShape of the ATAC dataset = {adata_ATAC.shape[0]} peaks, {adata_ATAC.shape[1]} cells')

print(f'\tFiltering out peaks expressed in fewer than 3 cells...')
sc.pp.filter_genes(adata_ATAC, min_cells=3)
adata_ATAC = adata_ATAC.copy()  # Ensure adata_ATAC is not a view after filtering
print(f'\t\tShape of the ATAC dataset = {adata_ATAC.shape[0]} peaks, {adata_ATAC.shape[1]} cells')

print('\nShape of the dataset after filtering')
print(f'\tscRNAseq Dataset: {adata_RNA.shape[0]} genes, {adata_RNA.shape[1]} cells')
print(f'\tscATACseq Dataset: {adata_ATAC.shape[0]} peaks, {adata_ATAC.shape[1]} cells')

print(f'\nCombining RNA and ATAC seq barcodes')
# Create a list of barcodes that match both the RNA-seq and ATAC-seq data
selected_barcode: list = list(set(adata_RNA.obs['barcode'].values) & set(adata_ATAC.obs['barcode'].values))

# Find the shared location of the barcodes so the RNA-seq and ATAC-seq data can be matched
rna_barcode_idx: pd.DataFrame = pd.DataFrame(range(adata_RNA.shape[0]), index=adata_RNA.obs['barcode'].values)
atac_barcode_idx: pd.DataFrame = pd.DataFrame(range(adata_ATAC.shape[0]), index=adata_ATAC.obs['barcode'].values)

# Only keep RNA-seq and ATAC-seq cell columns with shared barcodes
adata_RNA = adata_RNA[rna_barcode_idx.loc[selected_barcode][0]].copy()
adata_ATAC = adata_ATAC[atac_barcode_idx.loc[selected_barcode][0]].copy()

print(f'\nGenerating pseudo-bulk / metacells')
# Generate the pseudo-bulk/metacell
samplelist: list = list(set(adata_ATAC.obs['sample'].values))  # Extracts unique samples
tempsample = samplelist[0]

# Initialize empty DataFrames to store the pseudobulk data for target genes and regulatory elements
TG_pseudobulk: pd.DataFrame = pd.DataFrame([])
RE_pseudobulk: pd.DataFrame = pd.DataFrame([])

# Checks if the number of unique samples from adata_RNA is > 100. If so, runs pseudobulking
singlepseudobulk = (adata_RNA.obs['sample'].unique().shape[0] * adata_RNA.obs['sample'].unique().shape[0] > 100)
print(f'\tsinglepseudobulk = {singlepseudobulk}')

# Runs pseudobulking in chunks
for tempsample in samplelist:
    adata_RNAtemp = adata_RNA[adata_RNA.obs['sample'] == tempsample].copy()
    adata_ATACtemp = adata_ATAC[adata_ATAC.obs['sample'] == tempsample].copy()

    TG_pseudobulk_temp, RE_pseudobulk_temp = pseudo_bulk(adata_RNAtemp, adata_ATACtemp, singlepseudobulk)

    TG_pseudobulk = pd.concat([TG_pseudobulk, TG_pseudobulk_temp], axis=1)
    RE_pseudobulk = pd.concat([RE_pseudobulk, RE_pseudobulk_temp], axis=1)

    # Does not pseudobulk if less than 100 samples
    RE_pseudobulk[RE_pseudobulk > 100] = 100

if not os.path.exists(f'{shared_variables.data_dir}'):
    os.mkdir(f'{shared_variables.data_dir}')

# Writes out the AnnData objects as h5ad files
print(f'Writing adata_ATAC.h5ad and adata_RNA.h5ad')
adata_ATAC.write_h5ad(shared_variables.adata_ATAC_outpath)
adata_RNA.write_h5ad(shared_variables.adata_RNA_outpath)

# Change any NaN values to 0
TG_pseudobulk: pd.DataFrame = TG_pseudobulk.fillna(0)
RE_pseudobulk: pd.DataFrame = RE_pseudobulk.fillna(0)

print(f'Writing out peak gene ids')
pd.DataFrame(adata_ATAC.var['gene_ids']).to_csv(shared_variables.peak_gene_id_path, header=None, index=None)

print(f'Writing out pseudobulk...')
TG_pseudobulk.to_csv(shared_variables.TG_pseudobulk_path, sep='\t', index=True)
print(f'\tWrote to "...{shared_variables.TG_pseudobulk_path[-50:]}"')


RE_pseudobulk.to_csv(shared_variables.RE_pseudobulk_path, sep='\t', index=True)
print(f'\tWrote to "...{shared_variables.RE_pseudobulk_path[-50:]}"')
