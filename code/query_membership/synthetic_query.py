"""
Description: Script for generating synthetic query sets, where all genes in the target pathway are in the query set and the fdr is 10%.

"""
import numpy as np
import pandas as pd
"""
Description: 
This script generates synthetic query sets for pathway analysis. For each specified pathway, the target pathway genes are included in the query set, along with additional genes sampled randomly from the genome to achieve a false discovery rate (FDR) of 10%. 
The process is repeated for a user-defined number of iterations to generate multiple synthetic query sets. These sets can be used for further analysis or to test statistical methods under controlled conditions.
"""

import os
import argparse
from tqdm import tqdm

# Function to load pathway IDs from a file
def load_pathway_ids(file_path):
    """Load pathway IDs categorized by cancer and infection."""
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

# Function to generate synthetic membership for target genes
def synthetic_membership(genome_genes, target_genes):
    """Generate synthetic membership list for target genes."""
    genome_genes = list(set(genome_genes))  # Remove duplicates
    target_genes = list(set(target_genes))  # Remove duplicates
    
    if len(genome_genes) < 20:
        raise ValueError("Not enough genes to sample random genes. Provide a larger genome list.")
    
    # Ensure target_genes do not exceed genome_genes
    target_genes = [gene for gene in target_genes if gene in genome_genes]

    random_genes = np.random.choice(
        [gene for gene in genome_genes if gene not in target_genes], 
        size=int(0.1 * len(target_genes) / (1 - 0.1)), 
        replace=False
    )
    
    query_genes = set(target_genes).union(set(random_genes))
    query_membership = [1 if gene in query_genes else 0 for gene in genome_genes]
    
    return query_membership

# Function to save synthetic memberships to output directory
def save_synthetic_memberships(n_it, genome_path, pathway_file, pathway_ids, output_dir):
    """Generate and save synthetic memberships for each pathway."""
    genome_genes = pd.read_csv(genome_path, sep='\t')['ENSEMBL'].tolist()
    genome_genes = list(set(genome_genes))  # Remove duplicates
    print(f"Unique genome genes: {len(genome_genes)}")  # Check unique genes

    os.makedirs(output_dir, exist_ok=True)

    for pathway_name in pathway_ids:
        target_genes = pd.read_csv(pathway_file, sep='\t')
        target_genes = target_genes[target_genes['Pathway_Name'] == pathway_name]['Ensembl_ID'].tolist()
        target_genes = list(set(target_genes))  # Remove duplicates
        print(f"Number of target genes for {pathway_name}: {len(target_genes)}")  # Check target genes

        random_query_memberships_list = []
        for _ in tqdm(range(n_it), desc=f"Generating synthetic memberships for {pathway_name}"):
            membership = synthetic_membership(genome_genes, target_genes)
            if len(membership) == len(genome_genes):
                random_query_memberships_list.append(membership)
            else:
                print(f"Generated membership of unexpected length {len(membership)} for pathway {pathway_name}.")

        # Construct DataFrame only if list lengths are consistent
        if random_query_memberships_list:
            random_query_memberships = pd.DataFrame(
                data=np.array(random_query_memberships_list).T,
                columns=[f"Iter_{i}_Membership" for i in range(n_it)]
            )
            
            # Check if the lengths match
            if len(random_query_memberships) == len(genome_genes):
                random_query_memberships['Ensembl_ID'] = genome_genes
            else:
                print(f"Length mismatch: {len(random_query_memberships)} vs {len(genome_genes)}")

            synthetic_membership_path = os.path.join(output_dir, f'synthetic_memberships_{pathway_name}.tsv')
            random_query_memberships.to_csv(synthetic_membership_path, sep='\t', index=False)
            print(f"Synthetic memberships for {pathway_name} saved to {synthetic_membership_path}")
        else:
            print(f"No valid memberships generated for pathway {pathway_name}.")

# Main function to parse command-line arguments and execute the script
def synthetic_memberships_main():
    """Main function to handle command-line arguments and run the synthetic membership generation."""
    parser = argparse.ArgumentParser(description="Generate synthetic query memberships for pathway genes.")
    parser.add_argument('--genome_path', type=str, required=True, help='Path to the genome gene file (e.g., ENSEMBL IDs).')
    parser.add_argument('--pathway_file', type=str, required=True, help='Path to the pathway file containing target genes.')
    parser.add_argument('--pathway_groups', type=str, required=True, help='Path to the file containing pathway categories (e.g., cancer, infection).')
    parser.add_argument('--category', type=str, required=True, choices=['cancer', 'infection'], help='Category of pathways to generate memberships for (e.g., "cancer", "infection").')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the generated synthetic memberships.')
    parser.add_argument('--iterations', type=int, default=1, help='Number of iterations to generate synthetic memberships.')

    args = parser.parse_args()

    # Load pathway IDs from file and use the specified category
    pathway_ids = load_pathway_ids(args.pathway_groups)[args.category]

    # Run membership generation for each pathway ID
    save_synthetic_memberships(args.iterations, args.genome_path, args.pathway_file, pathway_ids, args.output_dir)

if __name__ == "__main__":
    synthetic_memberships_main()
