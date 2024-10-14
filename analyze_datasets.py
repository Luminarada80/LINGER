import os
import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import shared_variables

# Set shared file paths
ADATA_RNA_PATH = shared_variables.adata_RNA_outpath
CHIP_SEQ_GROUND_TRUTH_PATH = '/gpfs/Labs/Uzun/RESULTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/PBMC_CISTROME_RESULTS/ground_truth_w_score.csv'
OUTPUT_DIR = f'/gpfs/Labs/Uzun/RESULTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/PBMC_CISTROME_RESULTS'

# Set plot parameters
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)


def load_data(adata_path, ground_truth_path):
    """Load RNA data and ground truth data."""
    adata_rna = sc.read_h5ad(adata_path)
    ground_truth = pd.read_csv(ground_truth_path, sep=',', header=0)
    return adata_rna, ground_truth


def calculate_cell_type_percentages(adata_rna):
    """Calculate the percentage of cells for each cell type."""
    cell_type_list = list(set(adata_rna.obs['label']))
    num_cells = adata_rna.shape[0]

    cell_type_percentage_list = []
    for cell_type in cell_type_list:
        cell_type_num_cells = adata_rna[adata_rna.obs['label'] == cell_type].shape[0]
        percent_of_total = (cell_type_num_cells / num_cells) * 100
        cell_type_percentage_list.append((cell_type, percent_of_total))

    # Sort by percentage in descending order
    cell_type_percentage_list = sorted(cell_type_percentage_list, key=lambda x: x[1], reverse=True)
    return cell_type_percentage_list


def plot_bar_chart(data, title, xlabel, ylabel, output_path, ylim_top=None, rotation=45):
    """Plot a bar chart for given data."""
    labels, values = zip(*data)
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color='skyblue')

    # Set title and labels
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    if ylim_top:
        plt.ylim(top=ylim_top)

    # Rotate x-axis labels
    plt.xticks(rotation=rotation, ha='right', fontsize=10)

    # Add value labels on top of each bar
    for i, value in enumerate(values):
        plt.text(i, value + 0.5, f'{value:.1f}%', ha='center', fontsize=10)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_histogram(data, title, xlabel, ylabel, output_path):
    """Plot a histogram."""
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, color='skyblue', edgecolor='black')

    # Set title and labels
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def calculate_gene_expression_stats(adata_rna, ground_truth_tfs):
    """Calculate percentage of cells expressing each transcription factor in the ground truth."""
    num_cells = adata_rna.shape[0]
    gene_expr_dict = {'gene': [], 'percent_expression': []}

    for gene in ground_truth_tfs:
        if gene in adata_rna.var['gene_ids']:
            # Find the index of the gene
            gene_idx = adata_rna.var_names.get_loc(gene)

            # Calculate number of cells expressing the gene
            cell_expression = adata_rna[:, gene_idx].X
            num_cells_expressing_gene = np.sum(cell_expression > 0)

            # Calculate the percentage of total cells expressing the gene
            percent_expression = round((num_cells_expressing_gene / num_cells) * 100, 2)
            gene_expr_dict['gene'].append(gene)
            gene_expr_dict['percent_expression'].append(percent_expression)

    # Convert to DataFrame and sort
    gene_expr_df = pd.DataFrame(gene_expr_dict)
    return gene_expr_df.sort_values(by='percent_expression', ascending=False)


def plot_gene_expression(gene_expr_df, output_path):
    """Plot a bar chart of gene expression percentages."""
    plt.figure(figsize=(7, 4))
    plt.bar(gene_expr_df['gene'], gene_expr_df['percent_expression'], color='skyblue')

    # Set title and labels
    plt.title('Percent of PBMC cells expressing each ground truth TF', fontsize=MEDIUM_SIZE)
    plt.xlabel('Transcription Factor', fontsize=MEDIUM_SIZE)
    plt.ylabel('Percent gene expression', fontsize=MEDIUM_SIZE)
    plt.ylim(0, 100)
    plt.xticks(rotation=45, fontsize=7)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def calculate_trans_reg_potential(ground_truth_trans_reg, ground_truth_tfs):
    """Calculate trans-regulatory potential score for each transcription factor."""
    filtered_df = ground_truth_trans_reg[ground_truth_trans_reg['TF'].isin(ground_truth_tfs)]
    tf_trans_reg_stats = filtered_df.groupby('TF')['Score'].agg(['mean', 'std']).reset_index()
    return tf_trans_reg_stats.sort_values(by='mean', ascending=False)


def plot_trans_reg_potential(tf_trans_reg_stats, output_path):
    """Plot trans-regulatory potential score with error bars."""
    plt.figure(figsize=(7, 4))
    plt.bar(tf_trans_reg_stats['TF'], np.log10(tf_trans_reg_stats['mean']),
            yerr=tf_trans_reg_stats['std'], color='skyblue', capsize=5)

    # Set title and labels
    plt.title('Average trans-regulatory potential score for each TF in PBMC cells', fontsize=MEDIUM_SIZE)
    plt.xlabel('Transcription Factor', fontsize=MEDIUM_SIZE)
    plt.ylabel('Log10 Trans-regulatory potential score', fontsize=MEDIUM_SIZE)
    plt.xticks(rotation=45, fontsize=7)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# Main script
if __name__ == '__main__':
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    adata_rna, ground_truth = load_data(ADATA_RNA_PATH, CHIP_SEQ_GROUND_TRUTH_PATH)

    # Calculate and plot cell type percentages
    cell_type_percentage_list = calculate_cell_type_percentages(adata_rna)
    plot_bar_chart(cell_type_percentage_list, 'Percentage of Cell Types', 'Cell Type', 'Percentage (%)',
                   f'{OUTPUT_DIR}/cell_type_bar_chart.png', ylim_top=25)

    # Calculate gene expression statistics
    num_cells, num_genes = adata_rna.shape
    ground_truth_tfs = list(set(ground_truth['TF']))
    gene_expr_df_sorted = calculate_gene_expression_stats(adata_rna, ground_truth_tfs)

    # Save gene expression statistics to a TSV file
    gene_expr_df_sorted.to_csv(f'{OUTPUT_DIR}/PBMC_Percent_Cells_Expressing_Tf.tsv', sep='\t', index=False)

    # Plot gene expression percentages
    plot_gene_expression(gene_expr_df_sorted, f'{OUTPUT_DIR}/PBMC_Percent_Cells_Expressing_TF_Barplot.png')

    # Load trans-regulatory potential data and calculate statistics
    ground_truth_trans_reg = pd.read_csv(CHIP_SEQ_GROUND_TRUTH_PATH, header=0, sep=',')
    tf_trans_reg_stats_sorted = calculate_trans_reg_potential(ground_truth_trans_reg, ground_truth_tfs)

    # Plot trans-regulatory potential with error bars
    plot_trans_reg_potential(tf_trans_reg_stats_sorted, f'{OUTPUT_DIR}/TF_Average_H1_Trans_Reg_Potential_Barplot_with_Errorbars.png')

