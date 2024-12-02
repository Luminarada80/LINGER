import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import math
from copy import deepcopy
import os
import logging
import csv

# Install using 'conda install luminarada80::grn_analysis_tools' 
# or update to the newest version using 'conda update grn_analysis_tools'
from grn_analysis_tools import grn_formatting, plotting, resource_analysis, grn_stats

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
        
        linger_summary_dict, linger_accuracy_metrics = grn_stats.calculate_accuracy_metrics(linger_ground_truth, linger_no_ground_truth, linger_threshold, num_edges)
        
        if num_edges not in result_dict['linger']:
            result_dict['linger'][num_edges] = {}
        result_dict['linger'][num_edges] = linger_summary_dict
        
        cell_oracle_summary_dict, cell_oracle_accuracy_metrics = grn_stats.calculate_accuracy_metrics(oracle_ground_truth, oracle_no_ground_truth, cell_oracle_threshold, num_edges)
                
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
    # logging.info(df)
    
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
    # logging.info(comparison_df)

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

def plot_resources_by_step(resource_dict: dict, output_dir: str):
    # Plot the resource requirements by step for each sample
    plotting.plot_metric_by_step_adjusted(
        sample_resource_dict=resource_dict,
        metric='user_time',
        ylabel='User Time (h) / Percent CPU Usage',
        title='User Time / Percent CPU Usage by Step for Each Sample',
        filename=f'{output_dir}/Step_User_Time_Summary.png',
        divide_by_cpu=True
    )

    plotting.plot_metric_by_step_adjusted(
        sample_resource_dict=resource_dict,
        metric='system_time',
        ylabel='System Time (h) / Percent CPU Usage',
        title='System Time / Percent CPU Usage by Step for Each Sample',
        filename=f'{output_dir}/Step_System_Time.png',
        divide_by_cpu=True
    )

    plotting.plot_metric_by_step_adjusted(
        sample_resource_dict=resource_dict,
        metric='wall_clock_time',
        ylabel='Wall Clock Time (h)',
        title='Wall Clock Time by Step for Each Sample',
        filename=f'{output_dir}/Step_Wall_Clock_Time.png',
        divide_by_cpu=False
    )

    plotting.plot_metric_by_step_adjusted(
        sample_resource_dict=resource_dict,
        metric='max_ram',
        ylabel='Max RAM Usage (GB)',
        title='Max RAM usage by Step for Each Sample',
        filename=f'{output_dir}/Step_Max_Ram.png',
        divide_by_cpu=False
    )

    plotting.plot_metric_by_step_adjusted(
        sample_resource_dict=resource_dict,
        metric='percent_cpu',
        ylabel='Percent CPU',
        title='Percent of the CPU Used',
        filename=f'{output_dir}/Step_Percent_Cpu.png',
        divide_by_cpu=False
    )

def plot_resources_by_sample(resource_dict: dict, output_dir: str):
    # Plot the resource requirements for running the entire pipeline
    plotting.plot_total_metric_by_sample(
        sample_resource_dict=resource_dict,
        metric='user_time',
        ylabel='Total User Time / Percent CPU Usage',
        title='Total User Time / Percent CPU Usage for Each Sample',
        filename=f'{output_dir}/Total_User_Time.png',
        divide_by_cpu=True
    )

    plotting.plot_total_metric_by_sample(
        sample_resource_dict=resource_dict,
        metric='system_time',
        ylabel='Total System Time / Percent CPU Usage',
        title='Total System Time / Percent CPU Usage for Each Sample',
        filename=f'{output_dir}/Total_System_Time.png',
        divide_by_cpu=True
    )

    plotting.plot_total_metric_by_sample(
        sample_resource_dict=resource_dict,
        metric='wall_clock_time',
        ylabel='Wall Clock Time (h)',
        title='Total Wall Clock Time',
        filename=f'{output_dir}/Total_Wall_Clock_Time.png',
        divide_by_cpu=False
    )

    plotting.plot_total_metric_by_sample(
        sample_resource_dict=resource_dict,
        metric='max_ram',
        ylabel='Max RAM Usage (GB)',
        title='Max RAM usage',
        filename=f'{output_dir}/Total_Max_Ram.png',
        divide_by_cpu=False
    )

    plotting.plot_total_metric_by_sample(
        sample_resource_dict=resource_dict,
        metric='max_ram',
        ylabel='Max RAM Usage (GB)',
        title='Max RAM usage',
        filename=f'{output_dir}/Total_Max_Ram.png',
        divide_by_cpu=False
    )

    plotting.plot_total_metric_by_sample(
        sample_resource_dict=resource_dict,
        metric='percent_cpu',
        ylabel='Percent CPU',
        title='Average Percent of the CPU Used',
        filename=f'{output_dir}/Total_Percent_Cpu.png',
        divide_by_cpu=False
    )

def create_resource_requirement_summary(resource_dict: dict, output_dir: str):
    summary_dict = {}

    for sample, step_dict in resource_dict.items():
        if sample not in summary_dict:
            summary_dict[sample] = {
                    "user_time": 0,
                    "system_time": 0,
                    "percent_cpu": [],
                    "wall_clock_time": 0,
                    "max_ram": []
                }
        for step, sample_resource_dict in step_dict.items():
            for resource_name, resource_value in sample_resource_dict.items():
                if resource_name == "percent_cpu":
                    summary_dict[sample][resource_name].append(round(resource_value,2))
                elif resource_name == "max_ram":
                    summary_dict[sample][resource_name].append(round(resource_value,2))
                else:
                    summary_dict[sample][resource_name] += round(resource_value,2)
        summary_dict[sample]["max_ram"] = max(summary_dict[sample]["max_ram"])
        summary_dict[sample]["percent_cpu"] = round(sum(summary_dict[sample]["percent_cpu"]) / len(summary_dict[sample]["percent_cpu"]),2)
        
    summary_df = pd.DataFrame(summary_dict)
    summary_df = summary_df.reindex(sorted(summary_df.columns), axis=1)
    print(summary_df.head())

    summary_df.to_csv(f'{output_dir}/Resource_Summary.tsv', sep='\t')
    
def main():
    samples = [
        "1",
        "2",
        "3",
        "4",
    ]
    
    ground_truth_path = f'/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MACROPHAGE/RN204_macrophage_ground_truth.tsv'
    ground_truth = pd.read_csv(ground_truth_path, sep='\t', quoting=csv.QUOTE_NONE, on_bad_lines='skip', header=0)
    ground_truth['Source'] = ground_truth['Source'].str.upper().str.strip()
    ground_truth['Target'] = ground_truth['Target'].str.upper().str.strip()
    
    all_method_stats_dir = f'/gpfs/Labs/Uzun/RESULTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/MACROPHAGE_RESULTS/STATISTICAL_ANALYSIS/RN204_ground_truth'
    
    path_dict = {
        'linger': {
            "resource_analysis_dir": f'/gpfs/Labs/Uzun/RESULTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/MACROPHAGE_RESULTS/RESOURCE_ANALYSIS',
            "resource_log_dir": f'/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER/LOGS',
            "statistical_analysis_dir": f'/gpfs/Labs/Uzun/RESULTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/MACROPHAGE_RESULTS/STATISTICAL_ANALYSIS/RN204_ground_truth',
        }
    }
    
    total_accuracy_metrics: dict = {}
    random_accuracy_metrics: dict = {}   
    
    # Iterate through each sample in the list of sample names / numbers
    logging.info(f'\n----- Inferred Networks vs Ground Truth Statistical Analysis -----')
    for i, sample in enumerate(samples):
        logging.info(f'Processing sample {sample} ({i+1}/{len(samples)})')
        
        # Load the datasets for the sample
        linger_network = pd.read_csv(f'/gpfs/Labs/Uzun/RESULTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/MACROPHAGE_RESULTS/LINGER_TRAINED_MODELS/{sample}/cell_type_specific_trans_regulatory_macrophage.txt', sep='\t', index_col=0, header=0)
        linger_df = grn_formatting.create_standard_dataframe(linger_network)
        logging.debug(f'Inferred Net: {len(set(linger_df["Source"])):,} TFs, {len(set(linger_df["Target"])):,} TGs, and {len(linger_df["Source"]):,} edges\n')
        
        inferred_network_dict: dict = {
            'linger': linger_df,
            # 'cell_oracle': oracle_df
            }
        
        ground_truth_dict: dict = {}
            
        confusion_matrix_dict_all_methods: dict = {}
        
        # Process the ground truth and inferred networks
        logging.debug('\n----- Processing inferred networks and ground truths -----')
        for method, inferred_network_df in inferred_network_dict.items():
            
            sample_result_dir = f'{path_dict[method]["statistical_analysis_dir"]}/{sample}'
                
            if not os.path.exists(sample_result_dir):
                os.makedirs(sample_result_dir)
            
            # Write out the network size for the ground truth
            with open(f'{sample_result_dir}/summary_statistics.txt', 'w') as summary_stat_file:
                summary_stat_file.write(f'Dataset\tTFs\tTGs\tEdges\n')
                summary_stat_file.write(f'Ground_truth\t{len(set(ground_truth["Source"]))}\t{len(set(ground_truth["Target"]))}\t{len(ground_truth["Source"])}\n')
                
            logging.debug(f'Ground truth: {len(set(ground_truth["Source"])):,} TFs, {len(set(ground_truth["Target"])):,} TGs, and {len(ground_truth["Source"]):,} edges\n')
            
            
            # Add the method and samples to the total and random accuracy metrics dictionary
            if method not in total_accuracy_metrics:
                total_accuracy_metrics[method] = {
                    'sample_name': [],
                    'precision': [],
                    'recall': [],
                    'specificity': [],
                    'accuracy': [],
                    'f1_score': [],
                    'jaccard_index': [],
                    'weighted_jaccard_index': [],
                    'early_precision_rate': [],
                    'auroc': [],
                    'auprc': []
                }
            
            if method not in random_accuracy_metrics:
                random_accuracy_metrics[method] = {
                    'sample_name': [],
                    'precision': [],
                    'recall': [],
                    'specificity': [],
                    'accuracy': [],
                    'f1_score': [],
                    'jaccard_index': [],
                    'weighted_jaccard_index': [],
                    'early_precision_rate': [],
                    'auroc': [],
                    'auprc': []
                }
            
            total_accuracy_metrics[method]['sample_name'].append(sample)
            random_accuracy_metrics[method]['sample_name'].append(sample)
            
            # Append the inferred network information to the summary stats file
            with open(f'{sample_result_dir}/summary_statistics.txt', 'a') as summary_stat_file:
                summary_stat_file.write(f'{method.capitalize()}\t{len(set(inferred_network_df["Source"]))}\t{len(set(inferred_network_df["Target"]))}\t{len(inferred_network_df["Source"])}\n')
            
            ground_truth_dict[method] = deepcopy(ground_truth)
            
            method_ground_truth = ground_truth_dict[method]
            
            logging.debug(f'\tAdding inferred scores to the ground truth edges for {method}')
            method_ground_truth = grn_formatting.add_inferred_scores_to_ground_truth(method_ground_truth, inferred_network_df)
            
            # Drop any NaN scores in the ground truth after adding scores
            method_ground_truth = method_ground_truth.dropna(subset=['Score'])
            
            # Set the inferred network and ground truth scores to log2 scale
            inferred_network_df["Score"] = np.log2(inferred_network_df["Score"])
            method_ground_truth["Score"] = np.log2(method_ground_truth["Score"])
            
            # Remove any TFs and TGs from the inferred network that are not in the ground truth network
            logging.debug(f'\tRemoving TFs and TGs from inferred network that are not in ground truth for {method}')
            inferred_network_only_shared_tf_tg_df = grn_formatting.remove_tf_tg_not_in_ground_truth(method_ground_truth, inferred_network_df)
            logging.debug(f"\t\tGround truth shape: TFs = {len(set(method_ground_truth['Source']))}, TGs = {len(set(method_ground_truth['Target']))}, edges = {len(set(method_ground_truth['Score']))}")
            logging.debug(f"\t\tInferred same genes: TFs = {len(set(inferred_network_only_shared_tf_tg_df['Source']))}, TGs = {len(set(inferred_network_only_shared_tf_tg_df['Target']))}, edges = {len(set(inferred_network_only_shared_tf_tg_df['Score']))}")
            
            # Remove ground truth edges from the inferred network
            inferred_network_no_ground_truth_df = grn_formatting.remove_ground_truth_edges_from_inferred(method_ground_truth, inferred_network_only_shared_tf_tg_df)
            logging.debug(f"\t\tInferred no ground truth: TFs = {len(set(inferred_network_no_ground_truth_df['Source']))}, TGs = {len(set(inferred_network_no_ground_truth_df['Target']))}, edges = {len(set(inferred_network_no_ground_truth_df['Score']))}")
            
            # Drop any NaN scores in the inferred network after adding scores
            inferred_network_no_ground_truth_df = inferred_network_no_ground_truth_df.dropna(subset=['Score'])
            
            # Classify the true and predicted interactions for the ground truth and inferred networks
            logging.debug(f'\t\tClassifying true and predicted interactions for the ground truth and inferred networks')
            method_ground_truth, inferred_network_no_ground_truth_df = grn_stats.classify_interactions_by_threshold(method_ground_truth, inferred_network_no_ground_truth_df)
            
            logging.debug(f'\n----- Calculating accuracy metrics for {method} -----')
            # Calculate the accuracy metrics and the confusion matrix (TP, FP, TN, FN)
            accuracy_metric_dict, confusion_matrix_score_dict = grn_stats.calculate_accuracy_metrics(method_ground_truth, inferred_network_no_ground_truth_df)
            confusion_matrix_dict_all_methods[method] = confusion_matrix_score_dict        
            
            # Calculate the accuracy metrics and confusion matrix when the edge scores for the ground truth and inferred networks are randomized
            logging.debug(f'\tCalculating randomized accuracy metrics for {method}')
            randomized_histogram_path = f'{sample_result_dir}/Histogram_Randomized_GRN_Scores'
            randomized_accuracy_metric_dict, randomized_confusion_matrix_dict = grn_stats.create_randomized_inference_scores(
                method_ground_truth,
                inferred_network_no_ground_truth_df,
                histogram_save_path=randomized_histogram_path
                )
            
            # Create a dictionary of the accuracy metrics for the randomized accuracy metrics and the original accuracy metrics
            randomized_method_dict = {
                f"{method} Original": confusion_matrix_score_dict,
                f"{method} Randomized": randomized_confusion_matrix_dict
            }
            
            # Calculate the AUROC and AUPRC for the randomized edge scores
            randomized_auroc = grn_stats.calculate_auroc(randomized_confusion_matrix_dict)
            randomized_auprc = grn_stats.calculate_auprc(randomized_confusion_matrix_dict)
            
            random_accuracy_metrics[method]["auroc"].append(randomized_auroc)
            random_accuracy_metrics[method]["auprc"].append(randomized_auprc)
            
            # Calculate the AUROC and AUPRC for the randomized and original edge scores
            logging.debug(f'\tGenerating AUROC and AUPRC comparing the randomized scores to the original scores')
            randomized_auc_path = f'{sample_result_dir}/{method}_randomized_auroc_auprc.png'
            plotting.plot_auroc_auprc(randomized_method_dict, randomized_auc_path)
            
            logging.debug(f'\n----- Accuracy Metrics -----')
            logging.debug(f'\tSaving accuracy metrics for {method}')
            with open(f'{sample_result_dir}/accuracy_metrics.tsv', 'w') as accuracy_metric_file:
                accuracy_metric_file.write(f'Metric\tScore\n')
                for metric_name, score in accuracy_metric_dict.items():
                    accuracy_metric_file.write(f'{metric_name}\t{score:.4f}\n')
                    total_accuracy_metrics[method][metric_name].append(score)
            
            logging.debug(f'\tSaving randomized score accuracy methods for {method}')
            with open(f'{sample_result_dir}/randomized_accuracy_method.tsv', 'w') as random_accuracy_file:
                random_accuracy_file.write(f'Metric\tOriginal Score\tRandomized Score\n')
                for metric_name, score in accuracy_metric_dict.items():
                    random_accuracy_file.write(f'{metric_name}\t{score:.4f}\t{randomized_accuracy_metric_dict[metric_name]:4f}\n')
                    random_accuracy_metrics[method][metric_name].append(randomized_accuracy_metric_dict[metric_name])
            
            ground_truth_dict[method] = method_ground_truth
            inferred_network_dict[method] = inferred_network_no_ground_truth_df
        
        logging.debug(f'\n ----- Generating Summary Plots -----')
        logging.debug(f'\tSaving histogram of ground truth vs inferred GRN scores with threshold')
        histogram_path = f"{sample_result_dir}/Histogram_GRN_Scores_With_Threshold"
        plotting.plot_multiple_histogram_with_thresholds(ground_truth_dict, inferred_network_dict, histogram_path)
        
        logging.debug(f'\tSaving combined AUROC and AUPRC graph')
        
        # Calculate the AUROC and AUPRC
        auroc = grn_stats.calculate_auroc(confusion_matrix_score_dict)
        auprc = grn_stats.calculate_auprc(confusion_matrix_score_dict)
        
        # Plot the AUROC and AUPRC
        auc_path = f'{sample_result_dir}/auroc_auprc.png'
        plotting.plot_auroc_auprc(confusion_matrix_dict_all_methods, auc_path)

        total_accuracy_metrics[method]['auroc'].append(auroc)
        total_accuracy_metrics[method]['auprc'].append(auprc)
    
    # Create a dataframe of the total accuracy metrics by each sample for each inference method
    logging.info(f'\n----- Saving Accuracy Metrics for All Samples -----')
    for method in inferred_network_dict.keys():
        total_accuracy_metrics_df = pd.DataFrame(total_accuracy_metrics[method]).T
        random_accuracy_metrics_df = pd.DataFrame(random_accuracy_metrics[method]).T

        total_accuracy_metrics_df.to_csv(f'{all_method_stats_dir}/total_accuracy_metrics.tsv', sep='\t')
        random_accuracy_metrics_df.to_csv(f'{all_method_stats_dir}/random_accuracy_metrics.tsv', sep='\t')
    
    logging.info(f'\nDone! Results written to {all_method_stats_dir}')
    
    logging.info(f'\n----- Resource Analysis -----')
    for method in path_dict:
        logging.info(f'\tAnalysing resource requirements for {method}')
        log_dir = path_dict[method]["resource_log_dir"]
        output_dir = path_dict[method]["resource_analysis_dir"]
        
        resource_dict = resource_analysis.parse_time_module_output(log_dir, samples)
        
        plot_resources_by_step(resource_dict, output_dir)
        plot_resources_by_sample(resource_dict, output_dir)
        create_resource_requirement_summary(resource_dict, output_dir) 
    
        logging.info(f'\nDone! Results written to {output_dir}')

if __name__ == '__main__':
    
    # Configure logging
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')   
    
    main()
    