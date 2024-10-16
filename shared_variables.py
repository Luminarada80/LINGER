# Main paths
uzun_lab_dir: str = '/gpfs/Labs/Uzun'
data_dir: str = f'{uzun_lab_dir}/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_SC_DATA'
ground_truth_dir: str = f'{uzun_lab_dir}/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_SC_DATA'

# PBMC file paths
rna_data_path: str = f'{data_dir}/subsampled_RNA_1000.csv'
atac_data_path: str = f'{data_dir}/subsampled_ATAC_1000.csv'

tss_motif_info_path: str = f'{uzun_lab_dir}//DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_OTHER_SPECIES_TF_MOTIF_DATA/provide_data/'

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
output_dir: str = f'{uzun_lab_dir}/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_TRAINED_MODEL/'

# Decide which method to use (scNN or LINGER)
genome: str ='mm10'
method: str = 'scNN'
celltype: str = 'all'
activef: str = 'ReLU' # Neural Network activation function. Choose from: 'ReLU','sigmoid','tanh'