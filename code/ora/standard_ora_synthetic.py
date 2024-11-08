'''
Description: Script for performing standard Over-Representation Analysis (ORA) for synthetic data.

'''

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import sys
import argparse

# Add the project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

from ora.standard_ora import standard_ora


def load_pathway_ids(file_path):
    """Load pathway IDs categorized by infection and cancer."""
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

# Run ORA on synthetic query sets to generate null distribution
def standard_ora_null(random_query_memberships_path, pathway_file, n_it, null_distribution_path, pathway_ids):
    """Run ORA on synthetic query sets and collect null distributions."""
    pathways = pd.read_csv(pathway_file, sep='\t')['Pathway_Name'].unique()
    results = pd.DataFrame(index=pathways, columns=[f"Iter_{i}_p_value" for i in range(n_it)])

    def run_iteration(i):
        """Perform standard ORA for one iteration."""
        column_name = f"Iter_{i}_Membership"
        iteration_result = standard_ora(
            query_file=random_query_memberships_path,
            pathway_file=pathway_file,
            query_membership_type=column_name,
            pathway_membership_type='Crisp_Membership',
            output_path=None,
            dataset_name='dataset',
            pathway_ids=pathway_ids
        )
        return iteration_result[['Pathway_Name', 'p_value']]

    # Run iterations in parallel and fill results
    iteration_results = Parallel(n_jobs=-1)(
        delayed(run_iteration)(i) for i in tqdm(range(n_it), desc="Running ORA iterations")
    )

    # Fill results DataFrame with p-values
    for i, iteration_result in enumerate(iteration_results):
        column_name = f"Iter_{i}_p_value"
        for pathway_name in iteration_result['Pathway_Name']:
            results.at[pathway_name, column_name] = iteration_result.loc[
                iteration_result['Pathway_Name'] == pathway_name, 'p_value'
            ].values[0]

    # Filter results to include only specified pathways
    results = results.loc[results.index.isin(pathway_ids)]

    # Save null distributions to file
    results.to_csv(null_distribution_path, sep='\t')
    print(f"Null distributions saved to {null_distribution_path}")
    return results


def main():
    """Main function to handle command-line arguments and run the synthetic membership generation."""
    parser = argparse.ArgumentParser(description="Run ORA on synthetic query sets and generate null distributions.")
    parser.add_argument('--n_it', type=int, default=1, help='Number of iterations to run for ORA analysis.')
    parser.add_argument('--pathway_file', type=str, required=True, help='Path to the pathway file containing target genes.')
    parser.add_argument('--pathway_groups', type=str, required=True, help='Path to the file containing pathway categories (e.g., cancer, infection).')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the generated null distributions.')
    parser.add_argument('--category', type=str, default='infection', choices=['cancer', 'infection'], help='Category of pathways to process (default: infection).')

    args = parser.parse_args()

    # Load pathway IDs based on the selected category
    pathway_ids = load_pathway_ids(args.pathway_groups)[args.category]

    # Loop through each selected pathway ID to generate null distributions
    for pathway_id in pathway_ids:
        # Define paths for synthetic memberships and null distribution files
        synthetic_membership_path = f'{args.output_dir}/synthetic_memberships_{pathway_id}.tsv'
        null_distribution_path = f'{args.output_dir}/synthetic_p_values_standard_{pathway_id}.tsv'
        
        # Run ORA null distribution function for each pathway
        standard_ora_null(synthetic_membership_path, args.pathway_file, args.n_it, null_distribution_path, pathway_ids)


if __name__ == "__main__":
    main()
