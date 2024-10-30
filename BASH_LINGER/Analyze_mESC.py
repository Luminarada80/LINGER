import scanpy as sc
import pandas as pd


adata_RNA = sc.read_h5ad('./data/adata_RNA.h5ad')
adata_ATAC = sc.read_h5ad('./data/adata_ATAC.h5ad')
ground_truth = pd.read_csv('./data/filtered_ground_truth_56TFs_3036TGs.csv')

ground_truth_genes = []
adata_RNA_genes = []

for gene in ground_truth['Source']:
    if gene not in ground_truth_genes:
        ground_truth_genes.append(gene.upper())

for gene in adata_RNA.var['gene_ids']:
    if gene not in adata_RNA_genes:
        adata_RNA_genes.append(gene.upper())

print(len(set(ground_truth['Source'])))
print(len(set(ground_truth['Target'])))
print(len(ground_truth['Target']))

print(f'Common Genes between adata_RNA and ground truth: {len(list(set(ground_truth_genes) & set(adata_RNA_genes)))}')

