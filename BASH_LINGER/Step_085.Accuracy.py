import pandas as pd

import MESC_PIPELINE.shared_variables as shared_variables

CELL_TYPE = 'mESC'

ground_truth_df = pd.read_csv(f'{shared_variables.ground_truth_dir}/filtered_ground_truth_56TFs_3036TGs.csv', header = True)
inferred_network = pd.read_csv(f'{shared_variables.output_dir}cell_type_specific_trans_regulatory_{CELL_TYPE}.txt')

print(f'Ground Truth:')
print(f'{ground_truth_df.head()}')

print(f'\nLINGER Trans-regulatory network:')
print(f'{inferred_network.head()}')

# Step 1: prepare the gorund truth data
# Add a true_interaction column to the ground truth with values of 1 (positive interactions)
ground_truth_df['true_interaction'] = 1

inferred_interaction_dict = {'true_interaction': []}

# Find the TFs and TGs in ground truth df

# Iterate through the inferred interaction dict
    # If the inferred interaction is in the ground truth
        # append 1 to inferred_interaction_dict
    # else 
        # append 0 to inferred_interaction_dict

# Add the inferred interaction dict as a column to inferred network

# Merge the ground truth and inferred networks together

# Separate out the positive and negative interactions

# Randomly sample the same number of negatives as positives

