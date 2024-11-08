'''
Description: Script for precomputing probabilities from a single, shared pathway membership distributions using dynamic programming. These probabilities can then be used to run ORA with a fuzzy pathway set without permutations.

'''

import os
import numpy as np
import pandas as pd
from scipy.special import comb
from decimal import Decimal
from tqdm import tqdm
import argparse

def load_unique_memberships(df, factor=337, membership_type='Overlap_Membership'):
    """
    Collect unique genes with membership values across all pathways.
    Scale memberships and return as a tuple list, ignoring NaN and zero memberships.
    """
    df = df[df[membership_type].notna() & (df[membership_type] != 0)]
    df['Scaled_Membership'] = (df[membership_type] * factor).round().astype(int)
    unique_genes = df.drop_duplicates(subset=['Ensembl_ID'])
    membership_counts = unique_genes['Scaled_Membership'].value_counts().sort_index().reset_index()
    membership_counts.columns = ['Membership_Value', 'Count']
    k = list(membership_counts.itertuples(index=False, name=None))
    return k

def count_combinations(k, max_genes):
    """
    Count combinations using a general k across all pathways, with progress tracking.
    """
    max_membership_value = max(m for m, _ in k)
    max_s = max_genes * max_membership_value
    N = np.zeros((max_s + 1, max_genes + 1), dtype=np.object_)
    N[0, 0] = 1  

    for membership_value, count in tqdm(k, desc="Processing memberships"):
        for s in range(max_s, membership_value - 1, -1):
            for q in range(1, max_genes + 1):
                for b in range(1, min(count, q) + 1):
                    if s >= b * membership_value:
                        N[s, q] += comb(count, b, exact=True) * N[s - b * membership_value, q - b]
                        
    return pd.DataFrame(N, index=range(max_s + 1), columns=range(max_genes + 1))

def cumulative_probabilities(N, factor=337):
    """
    Calculate cumulative probabilities across all possible membership values with progress tracking.
    """
    P = pd.DataFrame(index=N.index, columns=N.columns, dtype=object)
    for c in tqdm(N.columns, desc="Calculating cumulative probabilities"):
        cumulative_counts = N[c].cumsum().astype(object)
        total_ways = Decimal(cumulative_counts.iloc[-1])
        prob_array = np.zeros_like(cumulative_counts, dtype=object)
        
        for s in range(len(prob_array)):
            if total_ways > 0:
                prob_array[s] = (total_ways - Decimal(cumulative_counts.iloc[s-1])) / total_ways if s > 0 else Decimal(1)
            else:
                prob_array[s] = Decimal(0)
        P[c] = prob_array
    
    P.index = [i / factor for i in P.index]
    return P

def dp_precompute_probabilities(pathway_file, pathway_membership_type, factor, output_path):
    """
    Precompute probabilities with a general k for all pathways, saving to a single file.
    """
    df = pd.read_csv(pathway_file, sep='\t')
    k = load_unique_memberships(df, factor, pathway_membership_type)
    max_genes = 384
    N = count_combinations(k, max_genes)
    probabilities_df = cumulative_probabilities(N, factor)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    probabilities_df.to_csv(output_path, sep='\t')
    print(f"Saved combined probabilities to {output_path}")

def parse_args():
    """
    Parse command-line arguments for the script.
    """
    parser = argparse.ArgumentParser(description="Precompute dynamic programming probabilities for pathway memberships.")
    parser.add_argument('--pathway_file', required=True, help='Path to the pathway membership file')
    parser.add_argument('--output_path', required=True, help='Path to save the output probabilities file')
    parser.add_argument('--factor', type=int, default=337, help='Scaling factor for memberships')
    parser.add_argument('--membership_type', type=str, default='Strict_Overlap_Membership', help='Membership type to use')

    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    
    # Run the probability precomputation
    dp_precompute_probabilities(
        pathway_file=args.pathway_file,
        pathway_membership_type=args.membership_type,
        factor=args.factor,
        output_path=args.output_path
    )

