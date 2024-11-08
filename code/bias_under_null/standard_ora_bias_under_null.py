'''
Description: This script generates random query gene memberships to create a null distribution of p-values for each pathway for standard Over-Representation Analysis (ORA).

'''
import numpy as np
import pandas as pd
import os
import sys
import argparse  # Importing argparse for command-line argument parsing
from tqdm import tqdm  # Importing tqdm for progress bars
from joblib import Parallel, delayed

# Add the project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

from ora.standard_ora import standard_ora

# Function to generate random memberships
def random_membership(genome_genes):
    dataset_genes = list(genome_genes)
    query_genes = set(np.random.choice(dataset_genes, size=int(0.1 * len(dataset_genes)), replace=False))
    random_query_membership = [1 if gene in query_genes else 0 for gene in genome_genes]
    return random_query_membership

def save_random_memberships(n_it, genome_path, random_membership_path):
    # Read genes from file and convert to a list
    genome_genes = pd.read_csv(genome_path, sep='\t')['ENSEMBL'].dropna().drop_duplicates().tolist()

    # Generate random memberships
    random_query_memberships_list = []
    for _ in tqdm(range(n_it), desc="Generating random memberships"):
        # Generate a random membership for each iteration
        membership = random_membership(genome_genes)
        random_query_memberships_list.append(membership)

    # Create a DataFrame with the random memberships
    random_query_memberships = pd.DataFrame(
        data=np.array(random_query_memberships_list).T,  # Transpose the array
        columns=[f"{i}_Membership" for i in range(n_it)]
    )

    # Add Ensembl_ID as a column
    random_query_memberships['Ensembl_ID'] = genome_genes

    # Save the random memberships to a CSV file
    random_query_memberships.to_csv(random_membership_path, sep='\t', index=False)
    
    return random_query_memberships

def standard_ora_null(random_query_memberships_path, pathway_file, n_it, null_distribution_path):
    """Run standard ORA null distribution analysis using random memberships in parallel."""
    pathways = pd.read_csv(pathway_file, sep='\t')['Pathway_Name'].unique()
    results = pd.DataFrame(index=pathways, columns=[f"{i}_p_value" for i in range(n_it)])

    def run_iteration(i):
        """Perform standard ORA for a single iteration and return results."""
        column_name = f"{i}_Membership"
        # Perform standard ORA for the current iteration
        iteration_result = standard_ora(
            query_file=random_query_memberships_path,
            pathway_file=pathway_file,
            query_membership_type=column_name,
            pathway_membership_type='Crisp_Membership',
            output_path=None,
            dataset_name='dataset',
            pathway_ids=None
        )
        
        # Collect results for this iteration
        return iteration_result[['Pathway_Name', 'p_value']]

    # Run iterations in parallel with a progress bar
    iteration_results = Parallel(n_jobs=-1)(delayed(run_iteration)(i) for i in range(n_it))

    # Fill the results DataFrame with p-values
    for i, iteration_result in enumerate(iteration_results):
        column_name = f"{i}_p_value"
        for pathway_name in iteration_result['Pathway_Name']:
            results.at[pathway_name, column_name] = iteration_result.loc[iteration_result['Pathway_Name'] == pathway_name, 'p_value'].values[0]

    # Save the results to a file
    results.to_csv(null_distribution_path, sep='\t')
    return results

# Main script execution via argparse
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run standard ORA with synthetic query memberships to generate null distributions")
    
    # Adding command-line arguments
    parser.add_argument('--genome-path', type=str, required=True, help='Path to the genome gene file')
    parser.add_argument('--random-membership-path', type=str, required=True, help='Path to save random membership file')
    parser.add_argument('--null-distribution-path', type=str, required=True, help='Path to save null distribution results')
    parser.add_argument('--n-it', type=int, default=1000, help='Number of iterations for random memberships')
    parser.add_argument('--pathway-file', type=str, required=True, help='Pathway file path for ORA analysis')

    # Parse command-line arguments
    args = parser.parse_args()

    # Step 1: Generate and save random memberships (commented out for now)
    save_random_memberships(args.n_it, args.genome_path, args.random_membership_path)

    # Step 2: Run the standard ORA null distribution analysis with saved memberships
    standard_ora_null(args.random_membership_path, args.pathway_file, args.n_it, args.null_distribution_path)
