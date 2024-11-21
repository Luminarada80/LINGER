import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import math
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
        
        linger_summary_dict, linger_accuracy_metrics = helper_functions.calculate_accuracy_metrics(linger_ground_truth, linger_no_ground_truth, linger_threshold, num_edges)
        
        if num_edges not in result_dict['linger']:
            result_dict['linger'][num_edges] = {}
        result_dict['linger'][num_edges] = linger_summary_dict
        
        cell_oracle_summary_dict, cell_oracle_accuracy_metrics = helper_functions.calculate_accuracy_metrics(oracle_ground_truth, oracle_no_ground_truth, cell_oracle_threshold, num_edges)
                
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
    accuracy_metrics_dict: dict,
    result_dir: str
    ):
    """Plots the AUROC and AUPRC"""
    
    # Define figure and subplots for combined AUROC and AUPRC plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for method, score_dict in enumerate(accuracy_metrics_dict.items()):
        y_true = score_dict['y_true']
        y_scores = score_dict['y_scores']
        
        fpr, tpr = roc_curve(y_true, y_scores)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        roc_auc = roc_auc_score(y_true, y_scores)
        prc_auc = auc(recall, precision)
        
        axes[0].plot(fpr, tpr, label=f'{method} AUROC = {roc_auc:.2f}', color='blue')
        axes[0].plot([0, 1], [0, 1], 'k--')  # Diagonal line for random performance
        
        axes[1].plot(recall, precision, label=f'{method} AUPRC = {prc_auc:.2f}', color='blue')
        
        
    axes[0].set_title("Combined AUROC")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")
        
        
    axes[1].set_title("Combined AUPRC")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower right")

    # Adjust layout and display the figure
    fig.tight_layout()
    plt.savefig(f'{result_dir}/5000_cell_oracle_linger_auroc_auprc.png', dpi=200)
    plt.show()
    

if __name__ == '__main__':
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')   

    # Load the datasets
    cell_oracle_network = pd.read_csv('/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.KARAMVEER/Celloracle/DS_014_mESC/Inferred_GRN/5000cells_E7.5_rep1_final_GRN.csv')
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
    logging.info(f'Ground truth: \n\t{len(set(ground_truth["Source"])):,} TFs, {len(set(ground_truth["Target"])):,} TGs, and {len(ground_truth["Source"]):,} edges')
    logging.info('Linger:')
    linger_df = helper_functions.create_standard_dataframe(linger_network)
    
    logging.info(f'CellOracle:')
    oracle_df = helper_functions.create_standard_dataframe(cell_oracle_network, score_col="coef_mean")
    
    inferred_network_dict: dict = {
        'linger': linger_df,
        'cell_oracle': oracle_df
        }
    
    ground_truth_dict: dict = {}
    
    method_score_distribution_dict: dict = {}
    
    accuracy_metric_dict: dict = {}
    
    # Process the ground truth and inferred networks
    print('\n----- Processing inferred networks and ground truths -----')
    for method, inferred_network_df in inferred_network_dict.items():
        
        ground_truth_dict[method] = deepcopy(ground_truth)
        
        method_ground_truth = ground_truth_dict[method]
        
        print(f'\tAdding inferred scores to the ground truth edges for {method}')
        method_ground_truth = helper_functions.add_inferred_scores_to_ground_truth(method_ground_truth, inferred_network_df)
        
        # Drop any NaN scores in the ground truth after adding scores
        method_ground_truth = method_ground_truth.dropna(subset=['Score'])
        
        # Remove ground truth edges from the inferred network
        print(f'\tRemoving ground truth edges from the inferred network for {method}')
        inferred_network_no_ground_truth_df = helper_functions.remove_tf_tg_not_in_ground_truth(method_ground_truth, inferred_network_df)
        print(f"\t\tGround truth shape: TFs = {len(set(method_ground_truth['Source']))}, TGs = {len(set(method_ground_truth['Target']))}, edges = {len(set(method_ground_truth['Score']))}")
        print(f"\t\tInferred net shape: TFs = {len(set(inferred_network_no_ground_truth_df['Source']))}, TGs = {len(set(inferred_network_no_ground_truth_df['Target']))}, edges = {len(set(inferred_network_no_ground_truth_df['Score']))}")
        
        # Drop any NaN scores in the inferred network after adding scores
        inferred_network_no_ground_truth_df = inferred_network_no_ground_truth_df.dropna(subset=['Score'])

        print(f'\tSeparating out TP, FP, TN, FN for {method}')
        separated_vals = helper_functions.find_inferred_network_accuracy_metrics(method_ground_truth, inferred_network_no_ground_truth_df)
        
        print(f'\n----- Calculating accuracy metrics for {method} -----')
        summary_dict, accuracy_metrics = helper_functions.calculate_accuracy_metrics(method_ground_truth, inferred_network_no_ground_truth_df)
        
        for metric_name, score in summary_dict.items():
            print(f'\t{metric_name}: {score:.4f}')
        
        print(f"\n\tTrue Positives: {accuracy_metrics['true_positive']:,}")
        print(f"\tTrue Negatives: {accuracy_metrics['true_negative']:,}")
        print(f"\tFalse Positives: {accuracy_metrics['false_positive']:,}")
        print(f"\tFalse Negatives: {accuracy_metrics['false_negative']:,}")
        
        accuracy_metric_dict[method] = accuracy_metrics
        
        ground_truth_dict[method] = method_ground_truth
        inferred_network_dict[method] = inferred_network_no_ground_truth_df
    
    helper_functions.plot_multiple_histogram_with_thresholds(ground_truth_dict, inferred_network_dict, result_dir)
    
    calculate_and_plot_auroc_auprc(accuracy_metric_dict, "./")
    
    
    
    