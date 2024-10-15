import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

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


def read_dataset(ground_truth_sep: str, ground_truth_header):
    adata_rna = sc.read_h5ad(ADATA_RNA_PATH)
    ground_truth = pd.read_csv(CHIP_SEQ_GROUND_TRUTH_PATH, sep=ground_truth_sep, header=ground_truth_header)

    print(adata_rna)
    return adata_rna, ground_truth

def find_cell_types(adata_rna):
    cell_type_list = list(set([i for i in adata_rna.obs['label']]))
    cell_type_percentage_list = []

    for cell_type in cell_type_list:
        print(cell_type)
        cell_type_dataset = adata_rna[adata_rna.obs['label'] == cell_type]
        cell_type_num_cells = cell_type_dataset.shape[0]
        percent_of_total = cell_type_num_cells/num_cells*100

        print(f'\tNumber of cells for {cell_type}: {cell_type_num_cells} ({round(percent_of_total)}%)')

        cell_type_percentage_list.append(percent_of_total)

    return cell_type_list, cell_type_percentage_list

def cell_type_percentage_bar_plot(cell_type_percentage_list):
    # Sort the cell types and percentages together in reverse order
    sorted_indices = np.argsort(cell_type_percentage_list)[::-1]
    sorted_cell_type_percentage_list = [cell_type_percentage_list[i] for i in sorted_indices]
    sorted_cell_type_set = [cell_type_list[i] for i in sorted_indices]

    # Create a bar chart
    plt.figure(figsize=(10, 6))  # Set the figure size to ensure enough space for labels
    plt.bar(sorted_cell_type_set, sorted_cell_type_percentage_list, color='skyblue')

    # Set title and labels
    plt.title('Percentage of Cell Types', fontsize=14)
    plt.xlabel('Cell Type', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.ylim(top=25)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=10)

    # Add percentage labels on top of each bar
    for i, percentage in enumerate(sorted_cell_type_percentage_list):
        plt.text(i, percentage + 0.5, f'{percentage:.1f}%', ha='center', fontsize=10)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'{OUTPUT_DIR}/cell_type_bar_chart.png', dpi=300)
    plt.close()

    return sorted_cell_type_set

def plot_cell_expression_histogram(dataset, cell_type):
    n_cells = dataset.shape[0]
    
    # Create the histogram and calculate bin heights
    counts, bins, _ = plt.hist(dataset.obs["n_genes"], bins=30, edgecolor='black', weights=np.ones_like(dataset.obs["n_genes"]) / n_cells * 100)
    
    plt.title(f'Percentage of cells by number of genes expressed in {cell_type}s')
    plt.ylim((0, 20))
    plt.xlim((0, 5000))
    plt.xlabel(f'Number of genes expressed ({dataset.shape[1]} total genes)')
    max_percentage = 20
    plt.yticks(np.arange(0, max_percentage + 1, 5), [f'{i}%' for i in range(0, max_percentage + 1, 5)])
    plt.ylabel(f'Percentage of cells ({n_cells} total cells)')
    
    plt.savefig(f'{OUTPUT_DIR}/avg_gene_expr_hist.png', dpi=300)
    plt.close()

def find_tf_expression(ground_truth, cell_type):
    ground_truth_tfs = list(set(ground_truth['TF']))

    # Iterate through the shared genes between the ground truth and the scRNAseq data
    gene_expr_dict = {'gene': [], 'percent_expression': []}
    for gene in ground_truth_tfs:
        if gene in adata_rna.var['gene_ids']:
            # Find the index of the gene name in the AnnData object
            gene_idx = adata_rna.var_names.get_loc(gene)

            cell_type_data = adata_rna[adata_rna.obs['label'] == cell_type]

            # Isolate the cell expresion data
            cell_expression = [i for i in cell_type_data[:, gene_idx].X]

            # Sum the number of cells expressing the gene
            num_cells_expressing_gene = np.sum([1 if i > 0 else 0 for i in cell_expression])

            # Calculate the percentage of total cells expressing the gene
            percent_expression = round((num_cells_expressing_gene/len(cell_expression))*100,2)

            # Append the gene and percent expression to dictionaries
            gene_expr_dict['gene'].append(gene)
            gene_expr_dict['percent_expression'].append(percent_expression)

    # Convert the gene expression dictionaries to a DataFrame
    percent_tf_expr_df = pd.DataFrame(gene_expr_dict)

    return percent_tf_expr_df, ground_truth_tfs

def plot_gene_expression_violinplot(adata_rna, cell_type_list):
    # Initialize a list to hold the data for each cell type
    data_per_cell_type = []
    
    # Loop over each cell type and extract the percentage of genes expressed
    for cell_type in cell_type_list:
        # Extract the number of genes expressed for the given cell type
        cell_type_data = adata_rna[adata_rna.obs['label'] == cell_type].obs["n_genes"]
        
        # Calculate the percentage of genes expressed
        percentage_genes_expressed = cell_type_data
        data_per_cell_type.append(percentage_genes_expressed)

    # Flatten the data for seaborn
    flat_data = []
    flat_labels = []
    for idx, cell_type in enumerate(cell_type_list):
        flat_data.extend(data_per_cell_type[idx])
        flat_labels.extend([cell_type] * len(data_per_cell_type[idx]))
    
    # Create the violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=flat_labels, y=flat_data, order=cell_type_list, cut=0)
    
    plt.title('Distribution of Genes Expressed by Cell Type')
    plt.xlabel('Cell Type')
    plt.ylabel('Number of Genes Expressed')
    
    # Rotate the x-axis labels by 45 degrees
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout to avoid cutting off the bottom
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{OUTPUT_DIR}/gene_expression_violinplot_percentage_sorted.png', dpi=300)
    plt.close()

def plot_tf_expression_barplot_by_celltype(percent_tf_expr_df_sorted, cell_type):
    fig, ax = plt.subplots(figsize=(7, 4))

    # Plot a bar graph of the sorted gene expression percentages
    ax.bar(percent_tf_expr_df_sorted['gene'], percent_tf_expr_df_sorted['percent_expression'])

    # Set the title, labels, and limits
    ax.set_title(f'Percent of {cell_type} expressing each ground truth TF', size=MEDIUM_SIZE)
    ax.set_ylim(bottom=0, top=100)
    ax.set_ylabel('Percent of cells', size=MEDIUM_SIZE)
    ax.set_xlabel('Transcription Factor', size=MEDIUM_SIZE)

    # Add a dashed line at y=10
    ax.axhline(y=10, color='black', linestyle='--', linewidth=1)

    # Set y-axis ticks to show every 10%
    ax.set_yticks(range(0, 101, 10))

    # Rotate the x-axis labels
    ax.tick_params(axis='x', labelsize=7, rotation=45)

    # Adjust layout to prevent cutting off
    plt.tight_layout()

    # Save the figure
    os.makedirs(f'{OUTPUT_DIR}/{cell_type}', exist_ok=True)
    plt.savefig(f'{OUTPUT_DIR}/{cell_type}/Percent_Cells_Expressing_TF_Barplot.png', dpi=300)
    plt.close()

# Read in the AnnData files for the scRNAseq and ground truth datasets
adata_rna, ground_truth = read_dataset(ground_truth_sep=',', ground_truth_header=0)

# Find the number of cells and genes in the dataset
num_cells = adata_rna.shape[0]
num_genes = adata_rna.shape[1]
print(f'\nTotal dataset size: {num_cells} cells x {num_genes} genes')

# Calculate the overall gene expression across all cell types
print(f'\nAverage gene expression: {round(np.average(adata_rna.obs["n_genes"]/num_genes)*100,2)}% ({round(np.average(adata_rna.obs["n_genes"]))})')
print(f'Std dev gene expression: {round(np.std(adata_rna.obs["n_genes"])/num_genes*100,2)}% ({round(np.std(adata_rna.obs["n_genes"]))})')
print(f'Min gene expression: {round(np.min(adata_rna.obs["n_genes"])/num_genes*100,2)}% ({round(np.min(adata_rna.obs["n_genes"]))})')
print(f'Max gene expression: {round(np.max(adata_rna.obs["n_genes"])/num_genes*100,2)}% ({round(np.max(adata_rna.obs["n_genes"]))})')

# Find the individual cell types in the dataset
cell_type_list, cell_type_percentage_list = find_cell_types(adata_rna)

# Plot of the percentage of cell types in the dataset
sorted_cell_type_set = cell_type_percentage_bar_plot(cell_type_percentage_list)

# Filter for cells expressing > 1000 genes
cells_high_expression = adata_rna[adata_rna.obs['n_genes'] > 1000]
print(f'{cells_high_expression.shape[0]} cells expressing >1000 genes')

# Plot a histogram of the number of genes each cell is expressing
plot_cell_expression_histogram(adata_rna, cell_type="PBMC")

# Plot a violin plot of the percent of genex expressed by cell type
plot_gene_expression_violinplot(adata_rna, sorted_cell_type_set)


for cell_type in sorted_cell_type_set:
    # Sets the cell type to H1 in the H1 dataset
    if cell_type == '0':
        cell_type == 'H1'

    # Get the cell expression 
    percent_tf_expr_df, ground_truth_tfs = find_tf_expression(ground_truth, cell_type)

    # Sort the dataframes
    percent_tf_expr_df_sorted = percent_tf_expr_df.sort_values(by='percent_expression', ascending=False)

    # Write the gene expression dataframe to a tsv file
    percent_tf_expr_df_sorted.to_csv(f'{OUTPUT_DIR}/PBMC_Percent_Cells_Expressing_Tf.tsv', sep='\t', index=False)

    with open(f'{OUTPUT_DIR}/{cell_type}/{cell_type}_Tf_Expression.txt', 'w') as outfile:
        outfile.write(f'Cell_type\t{"_".join([i for i in cell_type.split(" ")])}\n')
        outfile.write(f'Average_TF_expression\t{round(np.average(percent_tf_expr_df["percent_expression"]),2)}%\n')
        outfile.write(f'Std_dev_TF_expression\t{round(np.std(percent_tf_expr_df["percent_expression"]),2)}%\n')
        outfile.write(f'Min_TF_expression\t{round(np.min(percent_tf_expr_df["percent_expression"]),2)}%\n')
        outfile.write(f'Max_TF_expression\t{round(np.max(percent_tf_expr_df["percent_expression"]),2)}%\n')

    print(f'\nTF Expression in {cell_type}')
    print(f'\tAverage TF expression: {round(np.average(percent_tf_expr_df["percent_expression"]),2)}%')
    print(f'\tStd dev TF expression: {round(np.std(percent_tf_expr_df["percent_expression"]),2)}%')
    print(f'\tMin TF expression: {round(np.min(percent_tf_expr_df["percent_expression"]),2)}%')
    print(f'\tMax TF expression: {round(np.max(percent_tf_expr_df["percent_expression"]),2)}%')

    # Plot the expression of each TF for the cell type as a barplot
    plot_tf_expression_barplot_by_celltype(percent_tf_expr_df_sorted, cell_type)

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
plt.savefig(f'{OUTPUT_DIR}/TF_Average_PBMC_Trans_Reg_Potential_Barplot_with_Errorbars.png', dpi=300)
plt.close()



# Apply log10 transformation to the 'Score' column in filtered_df
filtered_df['log10_Score'] = np.log10(filtered_df['Score'].replace(0, np.nan))

# Group by 'TF' and calculate the mean and standard deviation in the log10 space
tf_trans_reg_stats_log10 = filtered_df.groupby('TF')['log10_Score'].agg(['mean', 'std']).reset_index()

# Merge the two datasets by 'TF' for overlapping barplots
merged_df = pd.merge(percent_tf_expr_df_sorted, tf_trans_reg_stats_log10, left_on='gene', right_on='TF')

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