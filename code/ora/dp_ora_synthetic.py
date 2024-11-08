'''
Description: Script for performing Over-Representation Analysis (ORA) for synthetic data with a fuzzy pathway set using probabilities precomputed through dynamci programming.

'''
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import sys
import os
import argparse

def load_pathway_ids(file_path):
    ids = {'cancer': [], 'infection': []}
    with open(file_path, 'r') as file:
        current_category = None
        for line in file:
            line = line.strip()
            if line.startswith('#'):
                if 'cancer' in line:
                    current_category = 'cancer'
                elif 'infection' in line:
                    current_category = 'infection'
            elif line:
                ids[current_category].append(line)
    return ids

# Add the project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

from ora.dp_ora import dp_ora

def dp_ora_null(random_query_memberships_path, pathway_file, n_it, null_distribution_path, pathway_ids, probability_folder, pathway_membership_type):
    pathways = pd.read_csv(pathway_file, sep='\t')['Pathway_Name'].unique()
    results = pd.DataFrame(columns=[f"Iter_{i}_p_value" for i in range(n_it)])

    # Filter only for specified pathway_ids
    pathways_to_include = [pathway for pathway in pathways if pathway in pathway_ids]
    
    for pathway in pathways_to_include:
        results.loc[pathway] = [None] * n_it  # Initialize rows for the specified pathways

    def run_iteration(i):
        column_name = f"Iter_{i}_Membership"
        iteration_result = dp_ora(
            query_file=random_query_memberships_path,
            pathway_file=pathway_file,
            query_membership_type=column_name,
            pathway_membership_type=pathway_membership_type,
            output_path=None,
            dataset_name='dataset',
            pathway_ids=pathway_ids,
            probability_folder=probability_folder
        )
        return iteration_result[['Pathway_Name', 'p_value']]

    iteration_results = Parallel(n_jobs=-1)(
        delayed(run_iteration)(i) for i in tqdm(range(n_it), desc="Running ORA iterations")
    )

    for i, iteration_result in enumerate(iteration_results):
        column_name = f"Iter_{i}_p_value"
        for pathway_name in iteration_result['Pathway_Name']:
            if pathway_name in results.index:
                results.at[pathway_name, column_name] = iteration_result.loc[
                    iteration_result['Pathway_Name'] == pathway_name, 'p_value'
                ].values[0]

    results.to_csv(null_distribution_path, sep='\t')
    print(f"Null distributions saved to {null_distribution_path}")


# Command-line arguments setup
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ORA null distribution analysis on synthetic query memberships.")
    
    # Define arguments
    parser.add_argument('--n_it', type=int, default=1, help="Number of iterations to run (default: 1)")
    parser.add_argument('--pathway_file', type=str, required=True, help="Path to the pathway file (TSV format)")
    parser.add_argument('--pathway_group_file', type=str, required=True, help="Path to the pathway group file (TXT format)")
    parser.add_argument('--membership_type', type=str, required=True, choices=['Overlap_Membership', 'Crisp_Membership'], help="Type of pathway membership to use")
    parser.add_argument('--probability_folder', type=str, required=True, help="Folder containing probability data for membership type")
    
    args = parser.parse_args()

    # Load pathway IDs from the pathway group file
    pathway_ids = load_pathway_ids(args.pathway_group_file)['infection']

    # Loop through each infection pathway ID
    for pathway_id in pathway_ids:
        # Define paths for synthetic memberships and null distribution files
        synthetic_membership_path = f'../../../data/query/synthetic/query_memberships/synthetic_memberships_{pathway_id}.tsv'
        null_distribution_path = f'../../../data/ora_output/synthetic/synthetic_p_values_dp_{args.membership_type.lower()}_{pathway_id}.tsv'
        
        # Run ORA null distribution function for each pathway
        dp_ora_null(synthetic_membership_path, args.pathway_file, args.n_it, null_distribution_path, pathway_ids, args.probability_folder, args.membership_type)
