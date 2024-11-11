import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from copy import deepcopy
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# Temporarily disable SettingWithCopyWarning
pd.options.mode.chained_assignment = None

# Set font to Arial and adjust font sizes
rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 14,  # General font size
    'axes.titlesize': 18,  # Title font size
    'axes.labelsize': 16,  # Axis label font size
    'xtick.labelsize': 14,  # X-axis tick label size
    'ytick.labelsize': 14,  # Y-axis tick label size
    'legend.fontsize': 14  # Legend font size
})


def create_standard_dataframe(old_df: pd.DataFrame, source_col: str, target_col: str, score_col: str):
    """Standardizes the column names and capitalizes the gene names"""
    
    new_df = pd.DataFrame(columns=["Source", "Target", "Score"])
    new_df["Source"] = old_df[source_col].str.upper().str.strip()
    new_df["Target"] = old_df[target_col].str.upper().str.strip()
    new_df["Score"] = old_df[score_col]
    
    return new_df


def find_ground_truth_scores_from_inferred(ground_truth: pd.DataFrame, inferred_network: pd.DataFrame, score_column_name: str):
    """Merges the inferred network scores with the ground truth dataframe"""
    ground_truth_with_scores = pd.merge(
        ground_truth, 
        inferred_network[['Source', 'Target', 'Score']], 
        left_on=['Source', 'Target'], 
        right_on=['Source', 'Target'], 
        how='left'
    ).rename(columns={'Score': score_column_name})

    
    return ground_truth_with_scores


def only_keep_shared_genes(ground_truth: pd.DataFrame, inferred_network: pd.DataFrame):
    """Removes any TFs and TGs that are not in the ground truth from the inferred network"""
    # Extract unique TFs and TGs from the ground truth network
    ground_truth_tfs = set(ground_truth['Source'])
    ground_truth_tgs = set(ground_truth['Target'])
    ground_truth_edges = set(zip(ground_truth['Source'], ground_truth['Target']))
    
    # Subset cell_oracle_network to contain only rows with TFs and TGs in ground_truth
    inferred_network_subset = inferred_network[
        (inferred_network['Source'].isin(ground_truth_tfs)) &
        (inferred_network['Target'].isin(ground_truth_tgs))
    ]
    
    return inferred_network_subset


def separate_ground_truth_from_inferred(ground_truth: pd.DataFrame, inferred_network: pd.DataFrame):
    """Removes ground truth edges from the inferred, so the inferred only contains true negative values"""
    
    # Get a list of the ground truth edges to separate 
    ground_truth_edges = set(zip(ground_truth['Source'], ground_truth['Target']))
    
    inferred_network_separated = inferred_network[
        ~inferred_network.apply(lambda row: (row['Source'], row['Target']) in ground_truth_edges, axis=1)
    ]
    
    return inferred_network_separated


def calculate_and_plot_auroc_auprc(
    ground_truth_with_scores: pd.DataFrame,
    cell_oracle_subset: pd.DataFrame,
    linger_network_subset: pd.DataFrame,
    result_dir: str
    ):
    """Plots the AUROC and AUPRC"""
    
    # Create a deepcopy of the ground truth for each inference method
    oracle_ground_truth = deepcopy(ground_truth_with_scores)
    linger_ground_truth = deepcopy(ground_truth_with_scores)
    
    # rename the inference method score column to "Score" to match the inferred dataset (for concatenating)
    oracle_ground_truth = oracle_ground_truth.rename(columns={'Oracle_Score': 'Score'})
    linger_ground_truth = linger_ground_truth.rename(columns={'Linger_Score': 'Score'})
    
    def calculate_roc_prc(ground_truth: pd.DataFrame, inferred_df: pd.DataFrame):
        # Classify non-ground truth scores for Oracle
        df_ground_truth = ground_truth.copy()
        inferred_df["Score"] = inferred_df["Score"]
        df_ground_truth["Score"] = df_ground_truth["Score"]
        
        # Calculate mean and standard deviation of ground truth Oracle scores
        mean = df_ground_truth['Score'].mean()
        stdev = df_ground_truth['Score'].std()

        # Define the lower threshold
        threshold = mean - 1 * stdev
        
        # Classify ground truth scores for Oracle
        df_ground_truth['true_interaction'] = 1
        df_ground_truth['predicted_interaction'] = np.where(
            df_ground_truth['Score'] >= threshold, 1, 0)

        inferred_df['true_interaction'] = 0
        inferred_df['predicted_interaction'] = np.where(
            inferred_df['Score'] >= threshold, 1, 0)

        auc_df = pd.concat([
            df_ground_truth[['Source', 'Target', 'Score', 'true_interaction', 'predicted_interaction']],
            inferred_df[['Source', 'Target', 'Score', 'true_interaction', 'predicted_interaction']]
        ], ignore_index=True).dropna()
        
        # Concatenate Oracle dataframes for AUC analysis
        y_true = auc_df['true_interaction']
        y_scores = auc_df['Score']

        return y_true, y_scores
    
    # Find the true interactions vs all scores
    y_true_oracle, y_scores_oracle = calculate_roc_prc(oracle_ground_truth, cell_oracle_subset)
    y_true_linger, y_scores_linger = calculate_roc_prc(linger_ground_truth, linger_network_subset)
    
    # Calculate ROC and PRC for Oracle Scores if positives are present
    fpr_oracle, tpr_oracle, _ = roc_curve(y_true_oracle, y_scores_oracle)
    precision_oracle, recall_oracle, _ = precision_recall_curve(y_true_oracle, y_scores_oracle)
    roc_auc_oracle = auc(fpr_oracle, tpr_oracle)
    prc_auc_oracle = auc(recall_oracle, precision_oracle)
    
    # Calculate ROC and PRC for Oracle Scores if positives are present
    fpr_linger, tpr_linger, _ = roc_curve(y_true_linger, y_scores_linger)
    precision_linger, recall_linger, _ = precision_recall_curve(y_true_linger, y_scores_linger)
    roc_auc_linger = auc(fpr_linger, tpr_linger)
    prc_auc_linger = auc(recall_linger, precision_linger)

    # Define figure and subplots for combined AUROC and AUPRC plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot combined AUROC
    axes[0].plot(fpr_oracle, tpr_oracle, label=f'Oracle AUROC = {roc_auc_oracle:.2f}', color='blue')
    axes[0].plot(fpr_linger, tpr_linger, label=f'Linger AUROC = {roc_auc_linger:.2f}', color='orange')
    axes[0].plot([0, 1], [0, 1], 'k--')  # Diagonal line for random performance
    axes[0].set_title("Combined AUROC")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")

    # Plot combined AUPRC
    axes[1].plot(recall_oracle, precision_oracle, label=f'Oracle AUPRC = {prc_auc_oracle:.2f}', color='blue')
    axes[1].plot(recall_linger, precision_linger, label=f'Linger AUPRC = {prc_auc_linger:.2f}', color='orange')
    axes[1].set_title("Combined AUPRC")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower right")

    # Adjust layout and display the figure
    fig.tight_layout()
    plt.savefig(f'{result_dir}/5000_cell_oracle_linger_auroc_auprc.png', dpi=200)
    plt.show()


def plot_histogram(ground_truth_with_scores: pd.DataFrame, linger_network_subset: pd.DataFrame, cell_oracle_subset: pd.DataFrame, result_dir: str):
    # Define the figure and two subplots
    plt.figure(figsize=(18, 8))

    # Plot the Oracle histogram
    plt.subplot(1, 2, 1)
    plt.hist(np.log2(cell_oracle_subset['Score'].dropna()), bins=150, alpha=0.7, label='Non-ground truth scores')
    plt.hist(np.log2(ground_truth_with_scores['Oracle_Score'].dropna()), bins=150, alpha=0.7, label='Ground truth scores')
    plt.title("Oracle Score Distribution")
    plt.xlabel("log2 Oracle Score")
    plt.ylabel("Frequency")

    # Plot the Linger histogram
    plt.subplot(1, 2, 2)
    plt.hist(np.log2(linger_network_subset['Score'].dropna()), bins=150, alpha=0.7, label='Non-ground truth scores')
    plt.hist(np.log2(ground_truth_with_scores['Linger_Score'].dropna()), bins=150, alpha=0.7, label='Ground truth scores')
    plt.title("Linger Score Distribution")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xlabel("log2 Linger Score")
    plt.ylabel("Frequency")

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.savefig(f'{result_dir}/5000_cell_oracle_linger_histograms.png')
    

def plot_histogram_with_thresholds(ground_truth_with_scores: pd.DataFrame, linger_network_subset: pd.DataFrame, cell_oracle_subset: pd.DataFrame, result_dir: str):
    
    def find_inferred_network_accuracy_metrics(ground_truth: pd.DataFrame, inferred_network: pd.DataFrame, score_col: str):
    
        # Set the ground truth and inferred network scores to log2 scale
        inferred_network["Score"] = np.log2(inferred_network["Score"])
        ground_truth[score_col] = np.log2(ground_truth[score_col])
        
        # Set the threshold for the accuracy metrics based on the ground truth mean
        mean = ground_truth[score_col].mean()
        std = ground_truth[score_col].std()
        
        threshold = mean - 1 * std
        
        tp = ground_truth[ground_truth[score_col] >= threshold][score_col]
        fn = ground_truth[ground_truth[score_col] < threshold][score_col]
        
        fp = inferred_network[inferred_network["Score"] >= threshold]["Score"]
        tn = inferred_network[inferred_network["Score"] < threshold]["Score"]
        
        return tp, fp, tn, fn, threshold
    
    tp_oracle, fp_oracle, tn_oracle, fn_oracle, oracle_threshold = find_inferred_network_accuracy_metrics(
        ground_truth_with_scores, cell_oracle_subset, "Oracle_Score"
        )
    
    tp_linger, fp_linger, tn_linger, fn_linger, linger_threshold = find_inferred_network_accuracy_metrics(
        ground_truth_with_scores, linger_network_subset, "Linger_Score"
        )
    
    plt.figure(figsize=(18, 8))
    
    # Plot the Oracle_Score histogram
    plt.subplot(1, 2, 1)

    # Plot histograms for Oracle Score categories with calculated bin sizes
    plt.hist(tn_oracle, bins=75, alpha=1, color='#b6cde0', label='True Negative (TN)')
    plt.hist(fp_oracle, bins=150, alpha=1, color='#4195df', label='False Positive (FP)')
    plt.hist(fn_oracle, bins=75, alpha=1, color='#efc69f', label='False Negative (FN)')
    plt.hist(tp_oracle, bins=150, alpha=1, color='#dc8634', label='True Positive (TP)')

    
    # Plot Oracle threshold line
    plt.axvline(x=oracle_threshold, color='black', linestyle='--', linewidth=2)
    plt.title("CellOracle Score Distribution")
    plt.xlabel("log2 CellOracle Score")
    plt.ylabel("Frequency")

    # Plot the Linger_Score histogram
    plt.subplot(1, 2, 2)

    # Plot histograms for Linger Score categories
    plt.hist(tn_linger, bins=150, alpha=1, color='#b6cde0', label='True Negative (TN)')
    plt.hist(fp_linger, bins=150, alpha=1, color='#4195df', label='False Positive (FP)')
    plt.hist(fn_linger, bins=150, alpha=1, color='#efc69f', label='False Negative (FN)')
    plt.hist(tp_linger, bins=150, alpha=1, color='#dc8634', label='True Positive (TP)')

    # Plot Linger threshold line
    plt.axvline(x=linger_threshold, color='black', linestyle='--', linewidth=2)
    plt.title("LINGER Score Distribution")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xlabel("log2 LINGER Score")
    plt.ylabel("Frequency")

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(f'{result_dir}/5000_cell_oracle_linger_histogram_with_accuracy_threshold.png')


if __name__ == '__main__':

    # Load the datasets
    cell_oracle_network = pd.read_csv('/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.KARAMVEER/Celloracle/DS_014_mESC/Inferred_GRN/5000cells_E7.5_rep1_final_GRN.csv')
    linger_network = pd.read_csv('/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.KARAMVEER/LINGER/5000_cell_type_TF_gene.csv', index_col=0)
    ground_truth = pd.read_csv('/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_SC_DATA/filtered_ground_truth_56TFs_3036TGs.csv', sep=',', header=0, index_col=0)
    
    result_dir = '/gpfs/Labs/Uzun/RESULTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/mESC_RESULTS/comparing_methods'
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Make sure the gene names are capitalized for the ground truth
    ground_truth['Source'] = ground_truth['Source'].str.upper().str.strip()
    ground_truth['Target'] = ground_truth['Target'].str.upper().str.strip()
    
    # Standardize the format of the two inferred networks and capitalize gene names
    print('Standardizing the format of the network dataframes')
    linger_df = create_standard_dataframe(linger_network, "Source", "Target", "Score")
    oracle_df = create_standard_dataframe(cell_oracle_network, "source", "target", "coef_mean")
    
    # Find the inferred network scores for the ground truth edges
    print('Finding ground truth scores')
    ground_truth_with_oracle_scores = find_ground_truth_scores_from_inferred(ground_truth, oracle_df, "Oracle_Score")
    ground_truth_with_scores = find_ground_truth_scores_from_inferred(ground_truth_with_oracle_scores, linger_df, "Linger_Score")
    
    # Subset the inferred dataset to only keep TFs and TGs that are shared between the inferred network and the ground truth
    print('Removing any TFs and TGs not found in the ground truth network')
    linger_network_subset = only_keep_shared_genes(ground_truth, linger_df)
    oracle_network_subset = only_keep_shared_genes(ground_truth, oracle_df)
    
    # Remove ground truth edges from the inferred network
    print('Removing ground truth edges from the inferred network')
    linger_no_ground_truth = separate_ground_truth_from_inferred(ground_truth, linger_network_subset)
    oracle_no_ground_truth = separate_ground_truth_from_inferred(ground_truth, oracle_network_subset)
    
    # Calculate the AUROC and AUPRC
    print('Calculating AUROC and AUPRC')
    calculate_and_plot_auroc_auprc(ground_truth_with_scores, oracle_network_subset, linger_network_subset, result_dir)
    
    # Plot the histogram of expression values
    print('Creating histogram of inferred network scores')
    plot_histogram(ground_truth_with_scores, linger_network_subset, oracle_network_subset, result_dir)
    
    # Plot the histogram of expression values with the accuracy metric cutoffs
    print('Creating histogram of inferred network scores with accuracy metric cutoffs')
    plot_histogram_with_thresholds(ground_truth_with_scores, linger_network_subset, oracle_network_subset, result_dir)
    