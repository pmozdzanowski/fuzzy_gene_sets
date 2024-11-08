'''
Description: Script for precomputing probabilities from pathway membership distributions using dynamic programming. These probabilities can then be used to run ORA with a fuzzy pathway set without permutations.

'''

import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm
from scipy.special import comb
from decimal import Decimal
import time
import argparse

def load_pathway_ids(file_path):
    '''Load pathway IDs categorized into 'cancer' and 'infection' from a file.'''
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

def count_combinations(k):
    '''Count the combinations of genes based on membership values and counts.'''
    # Calculate the maximum possible sum of membership values (max_s) and the total count of genes (Q)
    max_s = sum(m * c for m, c in k)  # Maximum sum of memberships
    Q = sum(c for _, c in k)           # Total count of genes in all categories

    # Initialize a matrix N to store the number of ways to achieve different sums for different gene counts
    N = np.zeros((max_s + 1, Q + 1), dtype=np.object_)  # Initialize the DP matrix
    N[0, 0] = 1  # There's one way to achieve a sum of 0 with 0 genes

    # Find the smallest membership value to optimize the DP process
    min_membership = min(m for m, _ in k)

    # Dynamic programming to count the ways to achieve each sum with varying gene counts
    for membership_value, c in k:
        for s in range(max_s, membership_value - 1, -1):  # Iterate over possible sums in reverse order
            # Calculate the minimum and maximum number of genes that can contribute to this sum
            min_genes = max(0, (s + membership_value - 1) // membership_value)  # Minimum genes needed
            max_genes = min(Q, s // min_membership)  # Maximum genes that could be used
            
            # Iterate over the possible number of genes contributing to the sum
            for q in range(min_genes, max_genes + 1):
                # Iterate over the possible number of selected genes from the current group
                for b in range(1, min(c, q) + 1):
                    # Check if the sum s can be achieved by using 'b' genes with the current membership value
                    if s >= b * membership_value:
                        # Update the DP table by adding the number of ways to achieve the remaining sum
                        N[s, q] += comb(c, b, exact=True) * N[s - b * membership_value, q - b]

    # Convert the row indices to a list of possible sums (max_s + 1 possible sums)
    row_index = list(range(max_s + 1))

    # Return the DP matrix as a DataFrame
    return pd.DataFrame(N, index=row_index, columns=range(Q + 1))


def cumulative_probabilities(N, factor=337):
    '''Calculate the cumulative probabilities from the combinations matrix.'''
    # Create an empty DataFrame to store the cumulative probabilities
    P = pd.DataFrame(index=N.index, columns=N.columns, dtype=object)
    
    # Iterate through each column (representing different gene counts)
    for c in N.columns:
        # Compute the cumulative sum of the number of ways to achieve each sum
        cumulative_counts = N[c].cumsum().astype(object)
        
        # Get the total number of ways to achieve all possible sums for this gene count
        total_ways = Decimal(cumulative_counts.iloc[-1])
        
        # Initialize an array to store the probability of each sum for the given gene count
        prob_array = np.zeros_like(cumulative_counts, dtype=object)
        
        # Calculate the probability for each sum, which is the fraction of ways to achieve the sum
        for s in range(len(prob_array)):
            if total_ways > 0:
                # The probability for sum 's' is the remaining probability after previous sums
                prob_array[s] = (total_ways - Decimal(cumulative_counts.iloc[s-1])) / total_ways if s > 0 else Decimal(1)
            else:
                # If there are no ways to achieve this sum, set the probability to 0
                prob_array[s] = Decimal(0)
        
        # Store the computed probabilities for this gene count in the DataFrame
        P[c] = prob_array
    
    # Scale the index (sums) by the given factor (to represent probabilities on a different scale)
    P.index = [i / factor for i in P.index]
    
    # Return the DataFrame with cumulative probabilities
    return P


def create_membership_tuples(df, factor=337, membership_type='Overlap_Membership'):
    '''Create tuples of scaled membership values for each pathway.'''
    pathway_membership_dict = {}
    for pathway, group in df.groupby('Pathway_Name'):
        group['Scaled_Membership'] = (group[membership_type] * factor).round().astype(int)
        membership_counts = group.groupby('Scaled_Membership')['Ensembl_ID'].count().reset_index()
        membership_tuples = list(membership_counts.itertuples(index=False, name=None))
        membership_tuples.sort(key=lambda x: x[0])
        pathway_membership_dict[pathway] = membership_tuples
    return pathway_membership_dict

def dp_per_pathway(pathway, k, output_folder, factor):
    '''Calculate and save DP probabilities for a single pathway.'''
    N = count_combinations(k)
    probabilities_df = cumulative_probabilities(N, factor)
    output_file_path = os.path.join(output_folder, f'{pathway}_dp_probabilities.tsv')
    probabilities_df.to_csv(output_file_path, sep='\t')
    return f"Saved probabilities for pathway: {pathway} to {output_file_path}"

def dp_precompute_probabilities(pathway_file, pathway_membership_type, factor, output_folder, selected_pathways=None):
    '''Precompute DP probabilities for all pathways in the provided file.'''
    df = pd.read_csv(pathway_file, sep='\t')
    pathway_memberships = create_membership_tuples(df, factor, pathway_membership_type)
    if selected_pathways is not None:
        pathway_memberships = {pathway: k for pathway, k in pathway_memberships.items() if pathway in selected_pathways}
    
    membership_output_folder = os.path.join(output_folder, pathway_membership_type)
    os.makedirs(membership_output_folder, exist_ok=True)

    start_time = time.time()
    
    Parallel(n_jobs=cpu_count()-1)(
        delayed(dp_per_pathway)(pathway, k, membership_output_folder, factor)
        for pathway, k in tqdm(pathway_memberships.items(), desc="Processing pathways")
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / len(pathway_memberships) if pathway_memberships else 0
    print(f"Total runtime: {total_time:.2f} seconds")
    print(f"Average processing time per pathway: {avg_time:.2f} seconds")

# Command-line interface
def dp_precompute_probabilities_main():
    '''Handle command-line arguments and initiate probability precomputation.'''
    parser = argparse.ArgumentParser(description="Precompute DP probabilities for pathways")
    parser.add_argument('--pathway_file', type=str, required=True, help="Path to the pathway file (e.g., KEGG_2022_pathway_memberships.tsv)")
    parser.add_argument('--output_folder', type=str, required=True, help="Output folder to save results")
    parser.add_argument('--factor', type=int, default=337, help="Scaling factor for membership values (default: 337)")
    parser.add_argument('--membership_type', type=str, default='Overlap_Membership', help="Type of pathway membership to use (default: 'Overlap_Membership')")
    parser.add_argument('--selected_pathways', type=str, nargs='*', default=None, help="List of specific pathways to process (optional)")

    args = parser.parse_args()

    dp_precompute_probabilities(
        pathway_file=args.pathway_file,
        pathway_membership_type=args.membership_type,
        factor=args.factor,
        output_folder=args.output_folder,
        selected_pathways=args.selected_pathways
    )

if __name__ == "__main__":
    dp_precompute_probabilities_main()
