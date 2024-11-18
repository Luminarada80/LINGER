import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import auc
import os
from matplotlib import rcParams

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


def plot_accuracy_metrics_all_samples(results_dir, samples):
    metrics_data = {
        "precision": [],
        "recall": [],
        "specificity": [],
        "accuracy": [],
        "f1 score": [],
        "Jaccard Index": [],
        "Weighted Jaccard Index": [],
        "Early Precision Rate (top 1000)": [],
        "AUROC": [],
        "AUPRC": []
    }

    sample_dir_paths = [
        sample_dir for sample_dir in os.listdir(results_dir)
        if any(rep in sample_dir for rep in samples)
    ]

    for sample_dir in sample_dir_paths:
        sample_summary_file_path = f'{results_dir}/{sample_dir}/mESC/summary_statistics.txt'
        
        if os.path.exists(sample_summary_file_path):        
            with open(sample_summary_file_path, 'r') as summary_file:
                for line in summary_file:
                    if ":" in line:
                        line = line.strip()
                        metric_name = line.split("- ")[1].split(":")[0]
                        
                        if metric_name in metrics_data.keys():
                            metric_value = float(line.split(":")[1].strip())
                            metrics_data[metric_name].append(metric_value)
        else:
            print(f'{sample_summary_file_path} does not exist')
            
    # Ensure all arrays are of the same length
    max_length = max(len(value) for value in metrics_data.values())
    for key in metrics_data.keys():
        if len(metrics_data[key]) < max_length:
            metrics_data[key].extend([None] * (max_length - len(metrics_data[key])))  # Pad with None

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(metrics_data)

    # Create boxplots
    plt.figure(figsize=(14, 8))
    df.boxplot()
    plt.ylim(0, 1)  # Set y-scale to 0-1
    plt.title('Accuracy metrics for all mESC samples and subsamples')
    plt.ylabel('Metric Value')
    plt.xticks(rotation=45)
    plt.grid(axis='x', which='major', linestyle='--', alpha=0.5)
    plt.grid(axis='y', visible=False)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/boxplot_accuracy_metric_summary_all_mESC_samples.png', dpi=200)

def plot_auroc_auprc_all_samples(results_dir, samples):
    accuracy_metric_dir = f'{results_dir}/accuracy_metrics/individual_sample_metrics'

    # Find all ROC and PR curve files
    roc_curve_files = [
        file for file in os.listdir(accuracy_metric_dir)
        if 'roc' in file and any(rep in file for rep in samples)
    ]
    pr_curve_files = [
        file for file in os.listdir(accuracy_metric_dir)
        if 'pr' in file and any(rep in file for rep in samples)
    ]

    # Initialize dictionary to store FPR, TPR, Precision, and Recall for each subsample
    subsample_data = {}

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    auroc_list = []
    auprc_list = []

    # Plot ROC curves
    ax_roc = axes[0]
    for sample_roc_file in roc_curve_files:
        # Full path to the file
        sample_roc_path = os.path.join(accuracy_metric_dir, sample_roc_file)

        # Extract sample name and other identifiers
        sample_name = [sample for sample in samples if sample in sample_roc_file][0]
        split_sample_name = sample_roc_file.split(sample_name)
        cells = split_sample_name[0].replace('_', ' ')
        replicate = split_sample_name[1].replace('_', ' ').replace(' roc curve df.csv', '')

        # Create subsample name
        subsample_name = f'{sample_name} {replicate} {cells.strip()}'

        # Read the file as a CSV
        roc_data = pd.read_csv(sample_roc_path)

        # Extract FPR and TPR
        fpr = roc_data['fpr']
        tpr = roc_data['tpr']

        # Initialize subsample data with FPR and TPR
        subsample_data[subsample_name] = {'fpr': fpr, 'tpr': tpr}
        
        auroc_list.append(auc(fpr, tpr))
        

    # Plot PR curves
    ax_pr = axes[1]
    for sample_pr_file in pr_curve_files:
        # Full path to the file
        sample_pr_path = os.path.join(accuracy_metric_dir, sample_pr_file)

        # Extract sample name and other identifiers
        sample_name = [sample for sample in samples if sample in sample_pr_file][0]
        split_sample_name = sample_pr_file.split(sample_name)
        cells = split_sample_name[0].replace('_', ' ')
        replicate = split_sample_name[1].replace('_', ' ').replace(' pr curve df.csv', '')

        # Create subsample name
        subsample_name = f'{sample_name} {replicate} {cells.strip()}'

        # Read the file as a CSV
        pr_data = pd.read_csv(sample_pr_path)

        # Extract Precision and Recall
        precision = pr_data['precision']
        recall = pr_data['recall']

        # Add Precision and Recall to subsample data
        if subsample_name in subsample_data:
            subsample_data[subsample_name]['precision'] = precision
            subsample_data[subsample_name]['recall'] = recall
        else:
            subsample_data[subsample_name] = {'precision': precision, 'recall': recall}
        
        auprc_list.append(auc(recall, precision))
    
    # Calculate average ROC and PR values
    mean_roc = np.mean(auroc_list)
    max_roc = max(auroc_list)
    min_roc = min(auroc_list)
    
    mean_precision = np.mean(auprc_list)
    max_precision = max(auprc_list)
    min_precision = min(auprc_list)

    # Assign consistent colors for each subsample
    color_map = plt.colormaps['tab10']  # Use recommended colormap access
    subsample_colors = {subsample_name: color_map(i / (len(subsample_data) - 1))
                        for i, subsample_name in enumerate(subsample_data.keys())}

    # Plot each ROC curve using the stored data
    for subsample_name, data in subsample_data.items():
        fpr = data.get('fpr')
        tpr = data.get('tpr')
        if fpr is not None and tpr is not None:
            ax_roc.plot(fpr, tpr, label=subsample_name, color=subsample_colors[subsample_name])

    # Add labels, legend, and title for ROC plot
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7)  # Diagonal reference line
    ax_roc.set_xlabel('False Positive Rate (FPR)')
    ax_roc.set_ylabel('True Positive Rate (TPR)')
    ax_roc.set_ylim([0,1])
    ax_roc.set_xlim([0,1])
    ax_roc.set_title('ROC Curves')
    ax_roc.grid(False)

    # Plot each PR curve using the stored data
    for subsample_name, data in subsample_data.items():
        precision = data.get('precision')
        recall = data.get('recall')
        if precision is not None and recall is not None:
            ax_pr.plot(recall, precision, label=subsample_name, color=subsample_colors[subsample_name])

    # Add labels, legend, and title for PR plot
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_ylim([0,1])
    ax_pr.set_xlim([0,1])
    ax_pr.set_title('PR Curves')
    ax_pr.grid(False)
    
    plt.text(
        1.35, 0.3,  # x, y position (adjust as needed)
        f"Max ROC: {max_roc:.2f}\nMean ROC: {mean_roc:.2f}\nMin ROC: {min_roc:.2f}\n\nMax PR: {max_precision:.2f}\nMean PR: {mean_precision:.2f}\nMin PR: {min_precision:.2f}",
        fontsize=14,
        ha='right',  # Horizontal alignment
        va='bottom',     # Vertical alignment
        bbox=dict(facecolor='white', edgecolor='black')
    )

    # Adjust layout to prevent cutting off
    plt.tight_layout(rect=[0, 0.06, 0.98, 0.90])
    
    plt.suptitle("ROC and PR curves for all LINGER mESC samples and subsamples")

    # # Set legend outside the plot area to the center-left
    # plt.figlegend(loc='center right', bbox_to_anchor=(0.98, 0.5), fontsize=10, ncol=2)

    # Save the combined figure
    save_path = f'{results_dir}/roc_and_pr_all_mESC_samples_and_subsamples.png'
    plt.savefig(save_path, dpi=300)
    print(f'ROC and PR plot saved to {save_path}')
    
samples = ["E7.5", "E7.75", "E8.0"]

results_dir = "/gpfs/Labs/Uzun/RESULTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/mESC_RESULTS"

plot_accuracy_metrics_all_samples(results_dir, samples)

plot_auroc_auprc_all_samples(results_dir, samples)
