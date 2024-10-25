# Main paths
uzun_lab_dir: str = '/gpfs/Labs/Uzun'
data_dir: str = f'{uzun_lab_dir}/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_PBMC_SC_DATA'
ground_truth_dir: str = f'{uzun_lab_dir}/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_PBMC_CISTROME'

# PBMC file paths
matrix_path: str = f'{data_dir}/pbmc_matrix.mtx'
features_path: str = f'{data_dir}/pbmc_features.tsv'
barcodes_path: str = f'{data_dir}/pbmc_barcodes.tsv'
label_path: str = f'{data_dir}/PBMC_label.txt'

# H1 file paths

# Data preprocessing output paths
adata_ATAC_outpath: str = f'{data_dir}/adata_ATAC.h5ad'
adata_RNA_outpath: str = f'{data_dir}/adata_RNA.h5ad'

peak_gene_id_path: str = f'{data_dir}/Peaks.txt'
TG_pseudobulk_path: str = f'{data_dir}/TG_pseudobulk.tsv'
RE_pseudobulk_path: str = f'{data_dir}/RE_pseudobulk.tsv'

peak_file_path: str = f'{data_dir}/Peaks.txt'

# Bulk data path
bulk_model_dir: str = f'{data_dir}/../LINGER_BULK_MODEL/'

# Output directory
output_dir: str = f'{uzun_lab_dir}/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_PBMC_TRAINED_MODEL/'

# Decide which method to use (scNN or LINGER)
genome: str ='hg38'
method: str = 'LINGER'
celltype: str = 'all'
activef: str = 'ReLU' # Neural Network activation function. Choose from: 'ReLU','sigmoid','tanh'