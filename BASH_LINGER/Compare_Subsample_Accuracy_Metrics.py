import pandas as pd
import glob
import sys
import os
import matplotlib.pyplot as plt

# Import the project directory to load the linger module
sys.path.insert(0, '/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/')

import BASH_LINGER.shared_variables as shared_variables

# Define the directory path where your CSV files are located
directory_path = f'{shared_variables.results_dir}/accuracy_metrics/individual_sample_metrics/'

# Get all files ending with '_accuracy_metrics_summary.csv' in the specified directory
accuracy_metric_files = glob.glob(os.path.join(directory_path, '*_accuracy_metrics_summary.csv'))
pr_files = glob.glob(os.path.join(directory_path, '*_pr_curve_df.csv'))
roc_files = glob.glob(os.path.join(directory_path, '*_roc_curve_df.csv'))

# Read each accuracy metric CSV file into a DataFrame and store in a list
accuracy_metric_dfs = [pd.read_csv(file) for file in accuracy_metric_files]

# Prepare data for PR and ROC plots, sorted by cell count
pr_data = [(int(os.path.basename(pr_file).split('_')[0]), pd.read_csv(pr_file)) for pr_file in pr_files]
roc_data = [(int(os.path.basename(roc_file).split('_')[0]), pd.read_csv(roc_file)) for roc_file in roc_files]

# Sort PR and ROC data by cell count
pr_data = sorted(pr_data, key=lambda x: x[0])
roc_data = sorted(roc_data, key=lambda x: x[0])

# Plotting PR and ROC curves
plt.figure(figsize=(12, 5))

# Subplot 1: Precision-Recall Curves
plt.subplot(1, 2, 1)
for sample, pr_curve in pr_data:
    plt.plot(pr_curve['recall'], pr_curve['precision'], label=f'Subsample {sample}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.grid(visible=False)

# Subplot 2: ROC Curves
plt.subplot(1, 2, 2)
for sample, roc_curve in roc_data:
    plt.plot(roc_curve['fpr'], roc_curve['tpr'], label=f'Subsample {sample}')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random chance
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.grid(visible=False)

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Save the combined plot
plt.tight_layout()
plt.savefig(f'{shared_variables.results_dir}/accuracy_metrics/AUROC_AUPRC_all_subsamples.png', dpi=300)
plt.show()

# Concatenate all accuracy metric DataFrames along rows, aligning by column names
combined_accuracy_metric_df = pd.concat(accuracy_metric_dfs, ignore_index=True).sort_values(by='sample').reset_index(drop=True)

# Display or save the combined DataFrame
print(combined_accuracy_metric_df)
combined_accuracy_metric_df.to_csv(f'{shared_variables.results_dir}/accuracy_metrics/accuracy_metrics_by_sample.tsv', sep='\t', index=False)

# Plot each metric with adjusted y-axis limits
plt.figure(figsize=(14, 6))
for i, column in enumerate(combined_accuracy_metric_df.columns[1:], 1):  # Skip 'sample' for plotting
    plt.subplot(2, 5, i)
    plt.plot(combined_accuracy_metric_df['sample'], combined_accuracy_metric_df[column], marker='o')
    plt.ylim((0, 1))  # Set the y-axis limits to (0,1)
    plt.title(column.capitalize())
    plt.xlabel('70 percent subsample')
    plt.xticks(combined_accuracy_metric_df['sample'], combined_accuracy_metric_df['sample'])  # Set x-ticks to display 1000, 2000, etc.
    plt.ylabel(column.capitalize())
    plt.grid()

# Save the accuracy metrics plot
plt.tight_layout()
plt.savefig(f'{shared_variables.results_dir}/accuracy_metrics/accuracy_metrics_by_sample.png', dpi=300)
plt.show()
