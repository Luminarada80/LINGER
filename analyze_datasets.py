import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import shared_variables

ADATA_RNA_PATH = shared_variables.adata_RNA_outpath
CHIP_SEQ_GROUND_TRUTH_PATH = '/gpfs/Labs/Uzun/RESULTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/PBMC_CISTROME_RESULTS/ground_truth_w_score.csv'
OUTPUT_DIR = f'/gpfs/Labs/Uzun/RESULTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/PBMC_CISTROME_RESULTS'

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

adata_rna = sc.read_h5ad(ADATA_RNA_PATH)
ground_truth = pd.read_csv(CHIP_SEQ_GROUND_TRUTH_PATH, sep=',', header=0)

print(adata_rna)

num_cells = adata_rna.shape[0]
num_genes = adata_rna.shape[1]

print(f'Total dataset size: {num_cells} cells x {num_genes} genes')

# Filter for cells expressing > 1000 genes
cells_high_expression = adata_rna[adata_rna.obs['n_genes'] > 1000]
print(f'{cells_high_expression.shape[0]} cells expressing >1000 genes')

print(f'Avg. num genes expressed: {np.mean(adata_rna.obs["n_genes"])}')
plt.hist(adata_rna.obs["n_genes"])
plt.title('Number of genes expressed in PBMCs')
plt.xlabel(f'Number of genes expressed ({num_genes} total genes)')
plt.ylabel(f'Number of cells ({num_cells} total cells)')
plt.savefig(f'{OUTPUT_DIR}/avg_gene_expr_hist.png', dpi=300)
plt.close()

ground_truth_tfs = list(set(ground_truth['TF']))
ground_truth_tgs = list(set(ground_truth['TG']))

# Iterate through the shared genes between the ground truth and the scRNAseq data
gene_expr_dict = {'gene': [], 'percent_expression': []}
for gene in ground_truth_tfs:
  if gene in adata_rna.var['gene_ids']:

    # Find the index of the gene name in the AnnData object
    gene_idx = adata_rna.var_names.get_loc(gene)

    # Isolate the cell expresion data
    cell_exprssion = [i for i in adata_rna[:, gene_idx].X]
    
    # Sum the number of H1 cells expressing the gene
    num_cells_expressing_gene = np.sum([1 if i > 0 else 0 for i in cell_exprssion])

    # Calculate the percentage of total cells expressing the gene
    percent_expression = round((num_cells_expressing_gene/num_cells)*100,2)

    # Append the gene and percent expression to dictionaries
    gene_expr_dict['gene'].append(gene)
    gene_expr_dict['percent_expression'].append(percent_expression)

# Convert the gene expression dictionary to a DataFrame
gene_expr_df = pd.DataFrame(gene_expr_dict)

# Sort the dataframe
gene_expr_df_sorted = gene_expr_df.sort_values(by='percent_expression', ascending=False)

# Write the gene expression dataframe to a tsv file
gene_expr_df_sorted.to_csv(f'{OUTPUT_DIR}/PBMC_Percent_Cells_Expressing_Tf.tsv', sep='\t', index=False)

print(f'\nAverage gene expression: {round(np.average(gene_expr_df["percent_expression"]),2)}%')
print(f'Std dev gene expression: {round(np.std(gene_expr_df["percent_expression"]),2)}%')
print(f'Min gene expression: {round(np.min(gene_expr_df["percent_expression"]),2)}%')
print(f'Max gene expression: {round(np.max(gene_expr_df["percent_expression"]),2)}%')

# Create the plot
fig, ax = plt.subplots(figsize=(7,4))

# Plot a bar graph of the sorted gene expression percentages
ax.bar(gene_expr_df_sorted['gene'], gene_expr_df_sorted['percent_expression'])

ax.set_title('Percent of PBMC cells expressing each ground truth TF', size=MEDIUM_SIZE)
ax.set_ylabel('Percent gene expression', size=MEDIUM_SIZE)
ax.set_xlabel('Transcription Factor', size=MEDIUM_SIZE)
ax.tick_params(axis='x', labelsize=7, rotation=45)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/PBMC_Percent_Cells_Expressing_TF_Barplot.png', dpi=300)
plt.close()


# Load the ground truth with trans-regulatory potential scores dataset
ground_truth_trans_reg = pd.read_csv(f'{OUTPUT_DIR}/ground_truth_w_score.csv', header=0, sep=',')
print(ground_truth_trans_reg.head())

# Filter for transcription factors that are in ground_truth_tfs
filtered_df = ground_truth_trans_reg[ground_truth_trans_reg['TF'].isin(ground_truth_tfs)]

# Group by 'TF' and calculate the mean score for each group
tf_trans_reg_score_dict = filtered_df.groupby('TF')['Score'].mean().to_dict()

# Group by 'TF' and calculate the mean and standard deviation for each group
tf_trans_reg_stats = filtered_df.groupby('TF')['Score'].agg(['mean', 'std']).reset_index()

# Sort by the mean score
tf_trans_reg_stats_sorted = tf_trans_reg_stats.sort_values(by='mean', ascending=False)

# ------ TRANS-REGULATORY POTENTIAL BARPLOT WITH ERROR BARS ------
# Create the plot
fig, ax = plt.subplots(figsize=(7,4))

# Plot a bar graph with error bars for the standard deviation
ax.bar(tf_trans_reg_stats_sorted['TF'],
       np.log10(tf_trans_reg_stats_sorted['mean']),
       yerr=tf_trans_reg_stats_sorted['std'],
       capsize=5)

# Set plot title and labels
ax.set_title('Average trans-regulatory potential score for each TF in PBMC cells', size=MEDIUM_SIZE)
ax.set_ylabel('Log10 Trans-regulatory potential score', size=MEDIUM_SIZE)
ax.set_xlabel('Transcription Factor', size=MEDIUM_SIZE)
ax.tick_params(axis='x', labelsize=7, rotation=45)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/TF_Average_H1_Trans_Reg_Potential_Barplot_with_Errorbars.png', dpi=300)
plt.close()



# Apply log10 transformation to the 'Score' column in filtered_df
filtered_df['log10_Score'] = np.log10(filtered_df['Score'].replace(0, np.nan))

# Group by 'TF' and calculate the mean and standard deviation in the log10 space
tf_trans_reg_stats_log10 = filtered_df.groupby('TF')['log10_Score'].agg(['mean', 'std']).reset_index()

# Merge the two datasets by 'TF' for overlapping barplots
merged_df = pd.merge(gene_expr_df_sorted, tf_trans_reg_stats_log10, left_on='gene', right_on='TF')

# ------ OVERLAPPING BAR PLOTS WITH LOG10 ERROR BARS ------
# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Bar width and offset for overlapping bars
bar_width = 0.4
index = np.arange(len(merged_df))

# Plot first set of bars (percent expression)
ax.bar(index - bar_width/2, merged_df['percent_expression'], bar_width, label='Percent Expression')

# Plot second set of bars (log-transformed trans-reg score) with error bars
ax.bar(index + bar_width/2, 
       merged_df['mean'],  # Mean log10-transformed score
       bar_width, 
       yerr=merged_df['std'],  # Standard deviation in the log10 space
       capsize=2, 
       label='Log10(Trans-reg Score)', 
       color='orange')

# Set plot title and labels
ax.set_title('Percent Expression and Log10 Trans-regulatory Scores for TFs in PBMC Cells', size=MEDIUM_SIZE)
ax.set_ylabel('Percent Expression / Log10(Trans-reg Score)', size=MEDIUM_SIZE)
ax.set_xlabel('Transcription Factor', size=MEDIUM_SIZE)

# Customize x-axis tick labels
ax.set_xticks(index)
ax.set_xticklabels(merged_df['TF'], rotation=45, ha='right', fontsize=7)

# Add a legend
ax.legend()

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/Overlapping_Barplot_TF_Percent_Trans_Reg_Scores.png', dpi=300)
plt.close()

# ------ VIOLIN PLOT ------
# Apply log10 transformation to the 'Score' column, handling any non-positive values
filtered_df['log2_Score'] = np.log2(filtered_df['Score'].replace(0, np.nan))

# Drop any rows where the log10 transformed score is NaN (which happens if the original score is 0 or negative)
filtered_df = filtered_df.dropna(subset=['log2_Score'])

# Create the violin plot
fig, ax = plt.subplots(figsize=(10, 6))

# Create the violin plot using seaborn with the log10-transformed scores
sns.violinplot(x='TF', y='log2_Score', data=filtered_df, ax=ax, scale='width', inner='quartile')

# Set plot title and labels
ax.set_title('Log2 Trans-regulatory potential scores for each TF in PBMC cells', size=MEDIUM_SIZE)
ax.set_ylabel('Log2 Trans-regulatory potential score', size=MEDIUM_SIZE)
ax.set_xlabel('Transcription Factor', size=MEDIUM_SIZE)

# Customize x-axis tick labels
ax.tick_params(axis='x', labelsize=7, rotation=45)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/TF_Log2_Trans_Reg_Potential_Violin_Plot.png', dpi=300)
plt.close()