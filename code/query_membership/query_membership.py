'''
Description: Script for deriving query membership values for genes from q-values and fold changes. 

'''

import pandas as pd
import numpy as np
import math
import argparse

def compute_query_membership(q):
    """
    Computes fuzzy membership based on q-value.
    """
    fuzzy_membership = 1 - 1 / (1 + np.exp(-300 * (q - 0.05)))  # Fuzzy logic based on q-value
    return fuzzy_membership

def compute_logfc_membership(logfc, logfc_threshold):
    """
    Computes fuzzy membership based on absolute logFC value.
    """
    fuzzy_membership_logfc = 1 / (1 + np.exp(-300 * (abs(logfc) - logfc_threshold)))  # Fuzzy logic based on logFC
    return fuzzy_membership_logfc

def process_query_set(query_path, output_path, logfc_threshold=1.0):
    """
    Processes the query set, calculates memberships, and saves the results.
    """
    query_df = pd.read_csv(query_path)  # Load query data

    # Compute Crisp Membership based on q-value and logFC threshold
    query_df['Crisp_Membership'] = ((query_df['q'] <= 0.05) & 
                                      (abs(query_df['logfc']) >= logfc_threshold)).astype(int)

    # Compute fuzzy memberships for q-value and logFC
    query_df['Fuzzy_Membership_Q'] = query_df['q'].apply(compute_query_membership)
    query_df['Fuzzy_Membership_LogFC'] = query_df['logfc'].apply(compute_logfc_membership, logfc_threshold=logfc_threshold)

    # Take the minimum of the two fuzzy memberships
    query_df['Fuzzy_Membership'] = query_df[['Fuzzy_Membership_Q', 'Fuzzy_Membership_LogFC']].min(axis=1)
    
    # Save results to output path
    query_df.to_csv(output_path, sep='\t', index=False)
    print(f"Results saved successfully to {output_path}.")

    # Calculate number and percentage of genes with Crisp Membership of 1
    num_crisp_membership_1 = query_df['Crisp_Membership'].sum()
    total_genes = len(query_df)
    percentage_crisp_membership_1 = (num_crisp_membership_1 / total_genes) * 100 if total_genes > 0 else 0

    # Print results
    print(f"Number of genes with Crisp Membership of 1: {num_crisp_membership_1}")
    print(f"Percentage of genes with Crisp Membership of 1: {percentage_crisp_membership_1:.2f}%")

    return query_df

def process_query_set_main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Process query dataset and compute memberships.")
    parser.add_argument("query_path", type=str, help="Path to the query dataset (CSV file).")
    parser.add_argument("output_path", type=str, help="Path to save the output (CSV file).")
    parser.add_argument("--logfc_threshold", type=float, default=math.log(1.1), help="LogFC threshold for fuzzy membership (default is log(1.1)).")
    
    args = parser.parse_args()

    # Process the query set with the specified logFC threshold
    process_query_set(args.query_path, args.output_path, logfc_threshold=args.logfc_threshold)

if __name__ == "__main__":
    process_query_set_main()

