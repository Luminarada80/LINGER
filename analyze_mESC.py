import scanpy as sc
import pandas as pd


adata_RNA = sc.read_h5ad('./data/adata_RNA.h5ad')
adata_ATAC = sc.read_h5ad('./data/adata_ATAC.h5ad')
ground_truth = pd.read_csv('./data/filtered_ground_truth_56TFs_3036TGs.csv')

# print(adata_RNA)
# print(adata_ATAC)

# print(adata_RNA.var['gene_ids'])

print(ground_truth['Source'])

for gene in ground_truth['Source']:
    if gene in ground_truth.var['gene_ids']:
        print(gene)

