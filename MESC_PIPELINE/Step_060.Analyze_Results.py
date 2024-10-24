import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import logging
import os

from linger import Benchmk
import MESC_PIPELINE.shared_variables as shared_variables

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

CELL_TYPE = 'mESC' # H1

# Define constants for file paths
# TF_RE_BINDING_PATH = f'{shared_variables.output_dir}cell_population_TF_RE_binding.txt'
# CIS_REG_NETWORK_PATH = f'{shared_variables.output_dir}cell_population_cis_regulatory.txt'
# TRANS_REG_NETWORK_PATH = f'{shared_variables.output_dir}cell_population_trans_regulatory.txt'

TF_RE_BINDING_PATH = f'{shared_variables.output_dir}cell_type_specific_TF_RE_binding_{CELL_TYPE}.txt'
CIS_REG_NETWORK_PATH = f'{shared_variables.output_dir}cell_type_specific_cis_regulatory_{CELL_TYPE}.txt'
TRANS_REG_NETWORK_PATH = f'{shared_variables.output_dir}cell_type_specific_trans_regulatory_{CELL_TYPE}.txt'
CHIP_SEQ_GROUND_TRUTH_PATH = f'/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/data/filtered_ground_truth_56TFs_3036TGs.csv'

RESULT_DIR: str = '/gpfs/Labs/Uzun/RESULTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/mESC_RESULTS'

def load_data():
    """Load transcription factor (TF), cis-regulatory, and trans-regulatory network data."""
    logging.info("Loading TF-RE binding, cis-regulatory, and trans-regulatory network data.")
    tf_re_binding = pd.read_csv(TF_RE_BINDING_PATH, sep='\t', index_col=0)
    cis_reg_network = pd.read_csv(CIS_REG_NETWORK_PATH, sep='\t', index_col=0)
    trans_reg_network = pd.read_csv(TRANS_REG_NETWORK_PATH, sep='\t', index_col=0)

    # Ensure the headers (column names) are in uppercase
    tf_re_binding.columns = tf_re_binding.columns.str.upper()
    cis_reg_network.columns = cis_reg_network.columns.str.upper()
    trans_reg_network.columns = trans_reg_network.columns.str.upper()

    # Ensure the row names are in uppercase
    tf_re_binding.index = tf_re_binding.index.str.upper()
    cis_reg_network.index = cis_reg_network.index.str.upper()
    trans_reg_network.index = trans_reg_network.index.str.upper()

    return tf_re_binding, cis_reg_network, trans_reg_network


def load_ground_truth():
    """Load ChIP-seq ground truth data."""
    logging.info("Loading ground truth data.")
    ground_truth = pd.read_csv(CHIP_SEQ_GROUND_TRUTH_PATH, sep=',', header=[0], index_col=[0])
    return ground_truth


def process_tf_tg_pairs(ground_truth: pd.DataFrame, trans_reg_network: pd.DataFrame):
    """Process ground truth TF-TG pairs and retrieve the corresponding scores from the trans-regulatory network."""
    tf_list, tg_list, value_list = [], [], []
    num_nan = 0
    num_rows = 0

    for index, row in ground_truth.iterrows():
        try:
            # Get the trans-regulatory score for the TF-TG pair
            value = trans_reg_network.loc[row[1], row[0]]
            tf_list.append(row[0])
            tg_list.append(row[1])
            value_list.append(value)
        except KeyError:
            # Handle missing value case
            num_nan += 1
        
        num_rows += 1

    return tf_list, tg_list, value_list, num_nan


def save_ground_truth_scores(tf_list: list, tg_list: list, value_list: list):
    """Save the processed ground truth TF-TG scores to a CSV file."""
    ground_truth_scores = {'TF': tf_list, 'TG': tg_list, 'Score': value_list}
    ground_truth_df = pd.DataFrame(ground_truth_scores)
    ground_truth_df.to_csv(f'{RESULT_DIR}/ground_truth_w_score.csv', header=True, index=False)
    logging.info("\nGround truth scores saved to 'ground_truth_w_score.csv'.")
    return ground_truth_df


def plot_trans_reg_distribution(trans_reg_network: pd.DataFrame, ground_truth_df: pd.DataFrame):
    """Plots a histogram of all trans-regulatory potential scores compared to the ground truth scores."""
    linger_scores = trans_reg_network['Score'].dropna()
    ground_truth_scores = ground_truth_df['Score'].dropna()

    # Handle zeros and small values by adding a small constant (1e-6)
    linger_scores = np.where(linger_scores > 0, linger_scores, 1e-6)
    ground_truth_scores = np.where(ground_truth_df['Score'] > 0, ground_truth_df['Score'], 1e-6)

    # Plot histograms for the LINGER trans-reg network and ground truth network
    plt.hist(np.log2(linger_scores), bins=150, log=True, alpha=0.7, label='LINGER trans-reg network')
    plt.hist(np.log2(ground_truth_scores), bins=150, log=True, alpha=0.7, label='Ground truth network')

    plt.title('Trans-regulatory potential scores')
    plt.rc('')
    plt.ylabel('Frequency (log)')
    plt.xlabel('log2 LINGER trans-regulatory potential score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{RESULT_DIR}/trans_reg_distribution.png', dpi=300)
    plt.close()
    logging.info("\nTrans-regulatory distribution plot saved to 'trans_reg_distribution.png'.")


def plot_box_whisker(trans_reg_network: pd.DataFrame, ground_truth_df: pd.DataFrame):
    """Create box and whisker plots to compare trans-regulatory network scores and ground truth scores."""
    # Handle zeros and small values by adding a small constant (1e-6)
    trans_reg_scores = np.where(trans_reg_network['Score'] > 0, trans_reg_network['Score'], 1e-6)
    ground_truth_scores = np.where(ground_truth_df['Score'] > 0, ground_truth_df['Score'], 1e-6)

    # Apply log2 transformation after ensuring no zero or negative values
    trans_reg_scores = np.log2(trans_reg_scores)
    ground_truth_scores = np.log2(ground_truth_scores)

    # Combine both into a single DataFrame for comparison
    scores_df = pd.DataFrame({
        'Score': np.concatenate([trans_reg_scores, ground_truth_scores]),
        'Source': ['Trans-Regulatory Network'] * len(trans_reg_scores) + ['Ground Truth'] * len(ground_truth_scores)
    })

    # Plot the box and whisker plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Source', y='Score', data=scores_df)
    plt.ylabel('log2 Score')
    plt.xlabel('')
    plt.title('Box and Whisker Plot of Trans-Regulatory Network vs Ground Truth Scores')
    plt.tight_layout()
    plt.savefig(f'{RESULT_DIR}/trans_reg_box_whisker_plot.png', dpi=300)
    plt.close()
    logging.info("Box and whisker plot saved to 'trans_reg_box_whisker_plot.png'.")


def plot_violin(trans_reg_network: pd.DataFrame, ground_truth_df: pd.DataFrame):
    """Create violin plots to compare trans-regulatory network scores and ground truth scores."""
    # Flatten the trans-regulatory network scores
    trans_reg_scores = trans_reg_network.values.flatten()

    # Handle zeros and small values by adding a small constant (1e-6)
    trans_reg_scores = np.where(trans_reg_scores > 0, trans_reg_scores, 1e-6)
    ground_truth_scores = np.where(ground_truth_df['Score'] > 0, ground_truth_df['Score'], 1e-6)

    # Apply log2 transformation after ensuring no zero or negative values
    trans_reg_scores = np.log2(trans_reg_scores)
    ground_truth_scores = np.log2(ground_truth_scores)

    # Combine both into a single DataFrame for comparison
    scores_df = pd.DataFrame({
        'Score': np.concatenate([trans_reg_scores, ground_truth_scores]),
        'Source': ['Trans-Regulatory Network'] * len(trans_reg_scores) + ['Ground Truth'] * len(ground_truth_scores)
    })

    # Plot the violin plot
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='Source', y='Score', data=scores_df, inner="quartile")
    plt.ylabel('log2 Score')
    plt.xlabel('')
    plt.title('Violin Plot of Trans-Regulatory Network vs Ground Truth Scores')
    plt.tight_layout()
    plt.savefig(f'{RESULT_DIR}/trans_reg_violin_plot.png', dpi=300)
    plt.close()
    logging.info("Violin plot saved to 'trans_reg_violin_plot.png'.")


def plot_violin_without_outliers(trans_reg_network: pd.DataFrame, ground_truth_df: pd.DataFrame, lower_percentile=1, upper_percentile=99):
    """Create violin plots for trans-regulatory network scores and ground truth scores after removing outliers."""
    # Flatten the trans-regulatory network scores
    trans_reg_scores = trans_reg_network.values.flatten()

    # Handle zeros and small values by adding a small constant (1e-6)
    trans_reg_scores = np.where(trans_reg_scores > 0, trans_reg_scores, 1e-6)
    ground_truth_scores = np.where(ground_truth_df['Score'] > 0, ground_truth_df['Score'], 1e-6)

    # Apply log2 transformation after ensuring no zero or negative values
    trans_reg_scores = np.log2(trans_reg_scores)
    ground_truth_scores = np.log2(ground_truth_scores)

    # Remove outliers based on specified percentiles
    trans_reg_lower = np.percentile(trans_reg_scores, lower_percentile)
    trans_reg_upper = np.percentile(trans_reg_scores, upper_percentile)
    ground_truth_lower = np.percentile(ground_truth_scores, lower_percentile)
    ground_truth_upper = np.percentile(ground_truth_scores, upper_percentile)

    # Filter the data within the percentile range
    trans_reg_filtered = trans_reg_scores[(trans_reg_scores >= trans_reg_lower) & (trans_reg_scores <= trans_reg_upper)]
    ground_truth_filtered = ground_truth_scores[(ground_truth_scores >= ground_truth_lower) & (ground_truth_scores <= ground_truth_upper)]

    # Combine both into a single DataFrame for comparison
    scores_df = pd.DataFrame({
        'Score': np.concatenate([trans_reg_filtered, ground_truth_filtered]),
        'Source': ['Trans-Regulatory Network'] * len(trans_reg_filtered) + ['Ground Truth'] * len(ground_truth_filtered)
    })

    # Plot the violin plot without outliers
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='Source', y='Score', data=scores_df, inner="quartile")
    plt.ylabel('log2 Score')
    plt.xlabel('')
    plt.title(f'Violin Plot of Trans-Regulatory Network vs Ground Truth Scores (No Outliers)')
    plt.tight_layout()
    plt.savefig(f'{RESULT_DIR}/trans_reg_violin_plot_no_outliers.png', dpi=300)
    plt.close()
    logging.info("Violin plot without outliers saved to 'trans_reg_violin_plot_no_outliers.png'.")


def create_network_graph(ground_truth_df: pd.DataFrame, output_file: str):
    """Create a directed transcription factor-target gene regulatory network using networkx and save it as a GEXF file."""
    edges = [(row['TF'], row['TG'], row['Score']) for index, row in ground_truth_df.iterrows()]

    G = nx.DiGraph()
    for edge in edges:
        G.add_edge(edge[0], edge[1], weight=edge[2])

    nx.write_gexf(G, output_file)
    logging.info(f"\nNetwork graph saved to '{output_file}'.")


def summarize_ground_truth_and_trans_reg(ground_truth_df: pd.DataFrame, trans_reg_network: pd.DataFrame, decimal_places: int = 2):
    """Generate summary statistics for both ground truth scores and trans-regulatory network scores and display them."""
    
    # Generate summary statistics for the 'Score' column in the ground_truth_df
    ground_truth_stats = ground_truth_df['Score'].describe()
    
    # Generate summary statistics for the flattened trans_reg_network scores
    trans_reg_stats = pd.Series(trans_reg_network.values.flatten()).describe()

    # Combine the two sets of summary statistics into a single DataFrame
    summary_stats = pd.DataFrame({
        'Ground Truth': ground_truth_stats,
        'Trans-Regulatory Network': trans_reg_stats
    })

    # Round the summary statistics to the specified number of decimal places
    summary_stats = summary_stats.round(decimal_places)

    # Print the rounded summary statistics to the terminal
    print("\nSummary Statistics for Ground Truth Scores and Trans-Regulatory Network Scores (Rounded):")
    print(summary_stats)

    # Save the summary to a csv file
    summary_stats.to_csv(f'{RESULT_DIR}/summary_statistics.csv')
    logging.info("Summary statistics saved to 'summary_statistics.csv'.")


def main():
    """Main function to run the analysis of the LINGER output"""

    # ----- DATA LOADING -----

    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR, exist_ok=True)

    tf_re_binding, cis_reg_network, trans_reg_network = load_data()
    ground_truth = load_ground_truth()

    # ----- GENERAL INFORMATION -----
    logging.info(f'\n----- TF-RE Binding Potential -----')
    logging.info(f'REs (rows): {tf_re_binding.shape[0]}')
    logging.info(f'TFs (columns): {tf_re_binding.shape[1]}')
    logging.info(tf_re_binding.head())

    logging.info(f'\n----- Cis-Regulatory Network -----')
    logging.info(f'Network size: {cis_reg_network.shape[0]}')
    logging.info(cis_reg_network.head())

    logging.info(f'\n----- Trans-Regulatory Network -----')
    logging.info(f'Number of TFs (columns): {trans_reg_network.shape[1]}')
    logging.info(f'Number of TGs (rows): {trans_reg_network.shape[0]}')
    logging.info(trans_reg_network.head())

    # Process the TF-TG pairs and retrieve scores from the trans-regulatory network
    tf_list, tg_list, value_list, num_nan = process_tf_tg_pairs(ground_truth, trans_reg_network)

    # Save the processed ground truth scores
    ground_truth_df = save_ground_truth_scores(tf_list, tg_list, value_list)

    logging.info(f'\n----- Ground Truth Network -----')
    logging.info(f'Number of TF-TG pairs: {ground_truth.shape[0]}')
    logging.info(ground_truth.head())

    logging.info(f'\nGround truth scores')
    logging.info(ground_truth_df.head())

    logging.info(f'\n----- Summary Statistics -----')
    # Generating summary statistics for the Score column
    summarize_ground_truth_and_trans_reg(ground_truth_df, trans_reg_network, decimal_places=2)

    # Calculate metrics to see how many of the ground truth TFs, TGs, and pairs are in the full 
    # trans-regulatory potential dataset
    num_rows = ground_truth.shape[0]
    
    num_tf_ground_truth = len(set(ground_truth_df["TF"]))
    num_tf_trans_net = len(set(tf_list))

    num_tg_ground_truth = len(set(ground_truth_df['TG']))
    tg_overlap = len([tg for tg in set(ground_truth_df["TG"]) if tg not in tf_list])

    logging.info(f'\n{num_tf_trans_net}/{num_tf_ground_truth} of ground truth TFs are represented in the trans-regulatory network')
    logging.info(f'{tg_overlap}/{num_tg_ground_truth} of ground truth TGs are represented in the trans-regulatory network')

    logging.info(f'{num_nan} / {num_rows} ({round(num_nan/num_rows*100,2)}%) of TF to TG pairs did not have a score in the LINGER dataset')

    # Find any TFs that are present in the ground truth dataset that are not in the e full dataset
    missing_tfs = [tf for tf in set(ground_truth_df["TF"]) if tf not in tf_list]

    if len(missing_tfs) > 0:
        logging.info(f'TFs present in ground truth missing in the trans-regulatory network')
        for i in missing_tfs:
            logging.info(i)

     # Reset index so the TGs become a column
    trans_reg_network.reset_index(inplace=True)

    # Melt the TRP dataset to have it in the same format as the ground truth
    trans_reg_net_melted = pd.melt(trans_reg_network, id_vars=['index'], var_name='TF', value_name='Score')

    # Add the column header for TGs
    trans_reg_net_melted.rename(columns={'index': 'TG'}, inplace=True)

    # Perform a left merge to find rows in trans_reg_pairs that are not in ground_truth_pairs
    difference_df = pd.merge(trans_reg_net_melted, ground_truth_df, on=['TF', 'TG'], how='left', indicator=True)

    # Store the true negatives as the TRP pairs not in the ground truth network (pairs that don't have a score in the ground truth dataframe)
    trans_reg_minus_ground_truth_df = difference_df[difference_df['_merge'] == 'left_only']

    # Keep only the relevant columns and rename for consistency
    trans_reg_minus_ground_truth_df = trans_reg_minus_ground_truth_df.drop(columns=['_merge', 'Score_y']).rename(columns={'Score_x': 'Score'})


    # ----- FIGURES -----
    # Histogram distribution of trans-regulatory potential scores and ground truth scores
    plot_trans_reg_distribution(trans_reg_minus_ground_truth_df, ground_truth_df)

    # Plot a box and whisker plot of the trans-regulatory potential scores and ground truth scores
    plot_box_whisker(trans_reg_minus_ground_truth_df, ground_truth_df)

    # Plot a violin plot of the trans-regulatory potential scores and ground truth scores
    plot_violin(trans_reg_minus_ground_truth_df, ground_truth_df)

    # Violin plot with outliers removed
    plot_violin_without_outliers(trans_reg_minus_ground_truth_df, ground_truth_df)


    # ----- NETWORKX NETWORK -----
    # Create and save the network graph
    create_network_graph(ground_truth_df, f'{RESULT_DIR}/transcription_factor_target_gene_network.gexf')

if __name__ == '__main__':
    main()