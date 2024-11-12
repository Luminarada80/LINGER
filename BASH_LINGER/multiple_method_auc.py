import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from copy import deepcopy
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, average_precision_score, precision_recall_curve
import logging
import csv

import helper_functions

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


def create_standard_dataframe(old_df: pd.DataFrame, source_col=None, target_col=None, score_col=None):
    """Standardizes the column names and capitalizes the gene names"""

    # Capitalize the column names for consistency
    old_df.columns = old_df.columns.str.capitalize()
    
    
    # Detect if the DataFrame needs to be melted
    if "Source" in old_df.columns and "Target" in old_df.columns:
        # print("\nDataFrame appears to be in long format; no melting is required.")
        
        source_col = source_col.capitalize() or old_df.columns[0].capitalize()
        target_col = target_col.capitalize() or old_df.columns[1].capitalize()
        score_col = score_col.capitalize() or old_df.columns[2].capitalize()
        
        logging.info(f'\tThere are {len(set(old_df[source_col]))} TFs, {len(set(old_df[target_col]))} TGs, and {len(old_df[source_col])} edges')
        
        # If no melting is required, we just rename columns directly
        melted_df = old_df.rename(columns={source_col: "Source", target_col: "Target", score_col: "Score"})
    
    # The dataframe needs to be melted, there are more than 3 columns and no "Source" or "Target" columns
    elif old_df.shape[1] > 3:
        
        num_rows, num_cols = old_df.shape
        
        logging.debug(f'Original dataframe has {num_rows} rows and {num_cols} columns')
        
        logging.debug(f'\nOld df before melting:')
        logging.debug(old_df.head())
        
        # TFs are columns, TGs are rows
        if num_rows >= num_cols:
            logging.info(f'\tThere are {num_cols} TFs, {num_rows} TGs, and {num_cols * num_rows} edges')
            # Transpose the columns and rows to prepare for melting
            old_df = old_df.T
            
            # Reset the index to make the TFs a column named 'Source'
            old_df = old_df.reset_index()
            old_df = old_df.rename(columns={'index': 'Source'})  # Rename the index column to 'Source'
    
            melted_df = old_df.melt(id_vars="Source", var_name="Target", value_name="Score")
            
        # TFs are rows, TGs are columns
        elif num_cols > num_rows:
            logging.info(f'\tThere are {num_rows} TFs, {num_cols} TGs, and {num_cols * num_rows} edges')
            
            # Reset the index to make the TFs a column named 'Source'
            old_df = old_df.reset_index()
            old_df = old_df.rename(columns={'index': 'Source'})  # Rename the index column to 'Source'
    
            melted_df = old_df.melt(id_vars="Source", var_name="Target", value_name="Score")

    # Capitalize and strip whitespace for consistency
    melted_df["Source"] = melted_df["Source"].str.upper().str.strip()
    melted_df["Target"] = melted_df["Target"].str.upper().str.strip()

    # Select and order columns as per new standard
    new_df = melted_df[["Source", "Target", "Score"]]
    
    logging.debug(f'\nNew df after standardizing:')
    logging.debug(new_df.head())
    
    return new_df


def remove_ground_truth_edges_from_inferred(ground_truth: pd.DataFrame, inferred_network: pd.DataFrame):
    """
    Removes ground truth edges from the inferred network after setting the ground truth scores.
    
    After this step, the inferred network does not contain any ground truth edge scores. This way, the 
    inferred network and ground truth network scores can be compared.
    """
    
    # Get a list of the ground truth edges to separate 
    ground_truth_edges = set(zip(ground_truth['Source'], ground_truth['Target']))
    
    # Create a new dataframe without the ground truth edges
    inferred_network_separated = inferred_network[
        ~inferred_network.apply(lambda row: (row['Source'], row['Target']) in ground_truth_edges, axis=1)
    ]
    
    return inferred_network_separated

def calculate_accuracy_metrics(ground_truth_df: pd.DataFrame, inferred_network: pd.DataFrame, lower_threshold, num_edges: int, summary_file_path: str):
    # Classify ground truth scores
    ground_truth_df['true_interaction'] = 1
    ground_truth_df['predicted_interaction'] = np.where(
        ground_truth_df['Score'] >= lower_threshold, 1, 0)

    # Classify non-ground truth scores (trans_reg_minus_ground_truth_df)
    inferred_network['true_interaction'] = 0
    inferred_network['predicted_interaction'] = np.where(
        inferred_network['Score'] >= lower_threshold, 1, 0)

    # Concatenate dataframes for AUC and further analysis
    auc_df = pd.concat([ground_truth_df, inferred_network])

    # Calculate the confusion matrix
    y_true = auc_df['true_interaction']
    y_pred = auc_df['predicted_interaction']
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
        # Calculations for accuracy metrics
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    jaccard_index = tp / (tp + fp + fn)

    # Weighted Jaccard Index
    weighted_tp = ground_truth_df.loc[ground_truth_df['predicted_interaction'] == 1, 'Score'].sum()
    weighted_fp = inferred_network.loc[inferred_network['predicted_interaction'] == 1, 'Score'].sum()
    weighted_fn = ground_truth_df.loc[ground_truth_df['predicted_interaction'] == 0, 'Score'].sum()
    weighted_jaccard_index = weighted_tp / (weighted_tp + weighted_fp + weighted_fn)
    
    # Early Precision Rate for top 1000 predictions
    top_edges = auc_df.nlargest(int(num_edges), 'Score')
    early_tp = top_edges[top_edges['true_interaction'] == 1].shape[0]
    early_fp = top_edges[top_edges['true_interaction'] == 0].shape[0]
    early_precision_rate = early_tp / (early_tp + early_fp)
    
    # Write results to summary file
    with open(summary_file_path, 'w') as file:
        file.write(f'\n- **TP, TN, FP, FN Counts**\n')
        file.write(f'\t- True Positives: {tp:,}\n')
        file.write(f'\t- False Positives: {fp:,}\n')
        file.write(f'\t- True Negatives: {tn:,}\n')
        file.write(f'\t- False Negatives: {fn:,}\n')

        file.write(f'\n- **Accuracy Metrics**\n')
        file.write(f"\t- precision: {precision:.4f}\n")
        file.write(f"\t- recall: {recall:.4f}\n")
        file.write(f"\t- specificity: {specificity:.4f}\n")
        file.write(f"\t- accuracy: {accuracy:.4f}\n")
        file.write(f"\t- f1 score: {f1_score:.4f}\n")
        file.write(f"\t- Jaccard Index: {jaccard_index:.4f}\n")
        file.write(f"\t- Weighted Jaccard Index: {weighted_jaccard_index:.4f}\n")
        file.write(f"\t- Early Precision Rate (top {num_edges}): {early_precision_rate:.4f}")
    
    summary_dict = {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "f1_score": f1_score,
        "jaccard_index": jaccard_index,
        "weighted_jaccard_index": weighted_jaccard_index,
        "early_precision_rate": early_precision_rate,
    }
    
    return summary_dict

def calculate_edge_cutoff_accuracy_metrics(ground_truth_with_scores, linger_no_ground_truth, oracle_no_ground_truth, result_dir):
    # Create a deepcopy of the ground truth for each inference method
    oracle_ground_truth = deepcopy(ground_truth_with_scores)
    linger_ground_truth = deepcopy(ground_truth_with_scores)
    
    # Drop any scores with an nan value
    oracle_ground_truth = oracle_ground_truth.dropna(subset=['Oracle_Score'])
    linger_ground_truth = linger_ground_truth.dropna(subset=['Linger_Score'])
    
    # rename the inference method score column to "Score" to match the inferred dataset (for concatenating)
    oracle_ground_truth = oracle_ground_truth.rename(columns={'Oracle_Score': 'Score'})
    linger_ground_truth = linger_ground_truth.rename(columns={'Linger_Score': 'Score'})
    
    # Calculate the accuracy metrics
    top_edge_accuracy_outdir = f'{result_dir}/top_edges_accuracy_metrics'
    
    if not os.path.exists(top_edge_accuracy_outdir):
        os.makedirs(top_edge_accuracy_outdir)
    
    # Determine the number of slices to use when calculating the accuracy metrics by top edges
    num_linger_edges = linger_no_ground_truth.shape[0]
    num_oracle_edges = oracle_no_ground_truth.shape[0]
    num_oracle_ground_truth_edges = oracle_ground_truth.shape[0]
    num_linger_ground_truth_edges = linger_ground_truth.shape[0]
    
    min_inferred_edges = min(num_linger_edges, num_oracle_edges, num_oracle_ground_truth_edges, num_linger_ground_truth_edges)
    
    # print(f'\tnum_linger_edges: {num_linger_edges}')
    # print(f'\tnum_oracle_edges: {num_oracle_edges}')
    # print(f'\tnum_oracle_ground_truth_edges: {num_oracle_ground_truth_edges}')
    # print(f'\tnum_linger_ground_truth_edges: {num_linger_ground_truth_edges}')
    
    # print(f'\tMinimum number of inferred edges = {min_inferred_edges}')
        
    interval = 10000
    
    # The last full interval before hitting the max number of edges
    last_interval = (min_inferred_edges // interval) * interval
    
    full_range = [int(i) for i in range(10000, last_interval + 1, interval)]
    
    result_dict = {
        "linger": {},
        "cell_oracle": {}
        }
    
    linger_ground_truth_sorted = linger_ground_truth.sort_values(by="Score", ascending=False)
    cell_oracle_ground_truth_sorted = oracle_ground_truth.sort_values(by="Score", ascending=False)
    
    for num_edges in full_range:
        
        # Sort the DataFrame by "Score" in descending order
        linger_threshold = linger_ground_truth_sorted.iloc[num_edges-1]["Score"]
        cell_oracle_threshold = cell_oracle_ground_truth_sorted.iloc[num_edges-1]["Score"]
        
        # print(f'\n\tLinger threshold: {linger_threshold}')
        # print(f'\tCellOracle threshold: {cell_oracle_threshold}')
        
        # print(f'\tcalculating accuracy metrics for top {num_edges} edges')
        
        linger_summary_dict = calculate_accuracy_metrics(linger_ground_truth, linger_no_ground_truth, linger_threshold, num_edges, f'{top_edge_accuracy_outdir}/top_{num_edges}_linger.txt')
        
        if num_edges not in result_dict['linger']:
            result_dict['linger'][num_edges] = {}
        result_dict['linger'][num_edges] = linger_summary_dict
        
        cell_oracle_summary_dict = calculate_accuracy_metrics(oracle_ground_truth, oracle_no_ground_truth, cell_oracle_threshold, num_edges, f'{top_edge_accuracy_outdir}/top_{num_edges}_cell_oracle.txt')
        
        if num_edges not in result_dict['cell_oracle']:
            result_dict['cell_oracle'][num_edges] = {}
        result_dict['cell_oracle'][num_edges] = cell_oracle_summary_dict
    
    # Flatten the dictionary into rows for each combination of method, cutoff, and metrics
    flattened_data = []
    for method, edges_dict in result_dict.items():
        for edges, metrics in edges_dict.items():
            row = {"Method": method, "Edge Cutoff": edges}
            row.update(metrics)
            flattened_data.append(row)

    # Convert the flattened data into a DataFrame
    df = pd.DataFrame(flattened_data)
    
    df.to_csv(f'{top_edge_accuracy_outdir}/full_comparison.csv')

    # # Display the DataFrame
    # print(df)
    
    linger_df = df[df['Method'] == 'linger'].set_index('Edge Cutoff')
    cell_oracle_df = df[df['Method'] == 'cell_oracle'].set_index('Edge Cutoff')

    # Calculate the difference between `linger` and `cell_oracle` for each metric
    comparison_df = linger_df[['precision', 'recall', 'specificity', 'accuracy', 'f1_score', 
                            'jaccard_index', 'weighted_jaccard_index', 'early_precision_rate']] - \
                    cell_oracle_df[['precision', 'recall', 'specificity', 'accuracy', 'f1_score', 
                                    'jaccard_index', 'weighted_jaccard_index', 'early_precision_rate']]

    # Rename columns to indicate they are differences
    comparison_df.columns = [f"{col}_diff" for col in comparison_df.columns]

    # Reset index for easier viewing
    comparison_df = comparison_df.reset_index()

    # # Display the comparison DataFrame
    # print(comparison_df)

    plt.figure(figsize=(14, 16))
    metrics = ["precision", "recall", "specificity", "accuracy", "f1_score", "jaccard_index", "weighted_jaccard_index", "early_precision_rate"]
    edge_cutoffs = comparison_df["Edge Cutoff"]
    
    y_ranges = {
        "precision": (0, 1),
        "recall": (0, 1),
        "specificity": (0, 1),
        "accuracy": (0, 1),
        "f1_score": (0, 1),
        "jaccard_index": (0, 1),
        "weighted_jaccard_index": (0, 1),
        "early_precision_rate": (0, 1)
    }
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(4, 2, i)
        plt.plot(edge_cutoffs, linger_df[metric], marker='o', label='Linger', color='blue')
        plt.plot(edge_cutoffs, cell_oracle_df[metric], marker='x', label='Cell Oracle', color='orange')
        plt.fill_between(edge_cutoffs, linger_df[metric], cell_oracle_df[metric], color='grey', alpha=0.2, label='Difference')
        plt.title(f"{metric.capitalize()}")
        plt.xticks(rotation=45)
        plt.ylim(y_ranges[metric]) 

        y_labels = {
            "precision": "Precision",
            "recall": "Recall",
            "specificity": "Specificity",
            "accuracy": "Accuracy",
            "f1_score": "F1 Score",
            "jaccard_index": "Jaccard Index",
            "weighted_jaccard_index": "Weighted Jaccard Index",
            "early_precision_rate": "Early Precision Rate"
        }
        plt.ylabel(y_labels[metric])
                
    plt.suptitle("Linger and Cell Oracle Accuracy Metrics Across Top Edges", fontsize=18, x=0.42, y=0.96)
        
    # Add a single x-axis label and adjust layout
    plt.gcf().text(0.42, 0.04, 'Number of Top Edges', ha='center', fontsize=18)

    # Adjust layout to prevent cutting off
    plt.tight_layout(rect=[0, 0.06, 0.80, 0.96])

    # Set legend outside the plot area to the center-left
    plt.figlegend(labels=["Linger", "Cell Oracle", "Difference"], loc='center right', bbox_to_anchor=(0.98, 0.5), fontsize=18)


    plt.savefig(f'{top_edge_accuracy_outdir}/linger_vs_cell_oracle_comparison.png', dpi=200)
        

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
    
    roc_auc_oracle = roc_auc_score(y_true_oracle, y_scores_oracle)
    prc_auc_oracle = average_precision_score(y_true_oracle, y_scores_oracle)
    
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
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')   

    # Load the datasets
    cell_oracle_network = pd.read_csv('/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.KARAMVEER/Celloracle/DS_014_mESC/Inferred_GRN/5000cells_E7.5_rep1_final_GRN.csv')
    # linger_network = pd.read_csv('/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.KARAMVEER/LINGER/5000_cell_type_TF_gene.csv', index_col=0)
    linger_network = pd.read_csv('/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_TRAINED_MODEL/sample_5000/cell_type_specific_trans_regulatory_mESC.txt', sep='\t', index_col=0, header=0)
    ground_truth = pd.read_csv('/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_SC_DATA/RN111.tsv', sep='\t', quoting=csv.QUOTE_NONE, on_bad_lines='skip', header=0)
    
    result_dir = '/gpfs/Labs/Uzun/RESULTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/mESC_RESULTS/comparing_methods'
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Make sure the gene names are capitalized for the ground truth
    ground_truth['Source'] = ground_truth['Source'].str.upper().str.strip()
    ground_truth['Target'] = ground_truth['Target'].str.upper().str.strip()
    
    # Standardize the format of the two inferred networks and capitalize gene names
    print('Standardizing the format of the network dataframes')
    logging.info(f'Ground truth: \n\t{len(set(ground_truth["Source"]))} TFs, {len(set(ground_truth["Target"]))} TGs, and {len(ground_truth["Source"])} edges')
    logging.info('Linger:')
    linger_df = helper_functions.create_standard_dataframe(linger_network)
    
    logging.info(f'CellOracle:')
    oracle_df = helper_functions.create_standard_dataframe(cell_oracle_network, score_col="coef_mean")
    
    # Find the inferred network scores for the ground truth edges
    print('Finding ground truth scores')
    ground_truth_with_oracle_scores = helper_functions.find_ground_truth_scores_from_inferred(ground_truth, oracle_df, "Oracle_Score")
    ground_truth_with_scores = helper_functions.find_ground_truth_scores_from_inferred(ground_truth_with_oracle_scores, linger_df, "Linger_Score")
    
    # Subset the inferred dataset to only keep TFs and TGs that are shared between the inferred network and the ground truth
    print('Removing any TFs and TGs not found in the ground truth network')
    linger_network_subset = helper_functions.remove_tf_tg_not_in_ground_truth(ground_truth, linger_df)
    oracle_network_subset = helper_functions.remove_tf_tg_not_in_ground_truth(ground_truth, oracle_df)
    
    # Remove ground truth edges from the inferred network
    print('Removing ground truth edges from the inferred network')
    linger_no_ground_truth = remove_ground_truth_edges_from_inferred(ground_truth, linger_network_subset)
    oracle_no_ground_truth = remove_ground_truth_edges_from_inferred(ground_truth, oracle_network_subset)
    
    print('Calculating accuracy metrics using threshold of top 1000, 2000, 3000, 4000, and 5000 edges')
    calculate_edge_cutoff_accuracy_metrics(ground_truth_with_scores, linger_no_ground_truth, oracle_no_ground_truth, result_dir)
    
    # Calculate the AUROC and AUPRC
    print('Calculating AUROC and AUPRC')
    calculate_and_plot_auroc_auprc(ground_truth_with_scores, oracle_no_ground_truth, linger_no_ground_truth, result_dir)
    
    # Plot the histogram of expression values
    print('Creating histogram of inferred network scores')
    plot_histogram(ground_truth_with_scores, linger_no_ground_truth, oracle_no_ground_truth, result_dir)
    
    # Plot the histogram of expression values with the accuracy metric cutoffs
    print('Creating histogram of inferred network scores with accuracy metric cutoffs')
    plot_histogram_with_thresholds(ground_truth_with_scores, linger_no_ground_truth, oracle_no_ground_truth, result_dir)
    