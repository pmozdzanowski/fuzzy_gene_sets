'''
Description: Script for performing fuzzy Over-Representation Analysis (ORA) using gene label permutations.

'''

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count
import matplotlib.pyplot as plt
import os
import argparse
import tqdm
from typing import List, Optional

def load_query(query_file: str, query_membership_type: str) -> pd.DataFrame:
    '''Load and clean query data from a file.'''
    # Check if the query file exists
    if not os.path.isfile(query_file):
        raise FileNotFoundError(f"Query file '{query_file}' does not exist.")
    
    # Read the query data from the specified file
    query_df = pd.read_csv(query_file, sep='\t')
    
    # Check if required columns are present
    if 'Ensembl_ID' not in query_df.columns or query_membership_type not in query_df.columns:
        raise ValueError(f"Query file must contain 'Ensembl_ID' and '{query_membership_type}' columns.")
    
    # Clean the DataFrame by dropping NaN values and renaming columns
    query_df = (query_df
                 .dropna()
                 .rename(columns={query_membership_type: 'Query_Membership'}))
    
    return query_df  # Return the cleaned query DataFrame

def load_pathways(pathway_file: str, pathway_membership_type: str) -> pd.DataFrame:
    '''Load and aggregate pathway data from a file.'''
    # Check if the pathway file exists
    if not os.path.isfile(pathway_file):
        raise FileNotFoundError(f"Pathway file '{pathway_file}' does not exist.")
    
    # Read the pathway data from the specified file, focusing on necessary columns
    pathway_df = pd.read_csv(
        pathway_file,
        sep='\t',
        usecols=['Pathway_Name', 'Description', 'Ensembl_ID', pathway_membership_type],
        dtype={'Ensembl_ID': str, pathway_membership_type: float}
    )

    # Check if required columns are present
    if 'Pathway_Name' not in pathway_df.columns or 'Description' not in pathway_df.columns or pathway_membership_type not in pathway_df.columns:
        raise ValueError(f"Pathway file must contain 'Pathway_Name', 'Description', and '{pathway_membership_type}' columns.")
    
    # Clean the DataFrame, group by Pathway_Name, and aggregate data
    pathway_df = (pathway_df
                   .dropna()
                   .rename(columns={pathway_membership_type: 'Pathway_Membership'})
                   .groupby('Pathway_Name')
                   .agg({
                       'Description': 'first',  # Take the first description per pathway
                       'Ensembl_ID': list,  # Store Ensembl IDs as lists
                       'Pathway_Membership': list  # Store memberships as lists
                   })
                   .reset_index())
    
    return pathway_df  # Return the cleaned pathway DataFrame

def ora_fuzzy_intersection(query_memberships, pathway_memberships):
    '''Calculate the fuzzy intersection size between query and pathway memberships.'''
    intersection = np.multiply(query_memberships, pathway_memberships)  # Element-wise product
    intersection_size = np.sum(intersection)  # Sum of the product values
    return intersection_size  # Return the intersection size

def ora_permutation(query_memberships, pathway_memberships):
    '''Generate a random intersection size by permuting query memberships.'''
    random_intersection_size = ora_fuzzy_intersection(np.random.permutation(query_memberships), pathway_memberships)
    return random_intersection_size  # Return the random intersection size

def ora_null_distribution(query_memberships, pathway_memberships, n_permutations=1000, n_jobs=cpu_count()-1):
    '''Compute a null distribution of intersection sizes from permutations.'''
    null_dist = Parallel(n_jobs=n_jobs)(delayed(ora_permutation)(query_memberships, pathway_memberships) for _ in range(n_permutations))
    return null_dist  # Return the null distribution

def ora_p_value(observed_intersection, null_distribution):
    '''Calculate the p-value based on the observed intersection and null distribution.'''
    p_value = np.mean(null_distribution >= observed_intersection)  # Compute the proportion of null values greater than or equal to observed
    return p_value  # Return the computed p-value

def fuzzy_ora_compute_stats(pathway, query_df, n_permutations, n_jobs, plots=False):
    '''Compute statistics for fuzzy ORA for a given pathway and query DataFrame.'''
    pathway_df = pd.DataFrame({
        'Ensembl_ID': pathway['Ensembl_ID'],
        'Pathway_Membership': pathway['Pathway_Membership']
    })

    # Perform a left join to keep all query genes and include only matching pathway genes
    merged_df = pd.merge(query_df, pathway_df, on='Ensembl_ID', how='left').fillna(0)

    # Set pathway memberships to zero for genes not found in the pathway
    merged_df['Pathway_Membership'] = merged_df['Pathway_Membership'].fillna(0)

    # Get membership arrays for both query and pathway
    query_memberships = merged_df['Query_Membership'].values
    pathway_memberships = merged_df['Pathway_Membership'].values

    # Calculate observed intersection
    observed_intersection = ora_fuzzy_intersection(query_memberships, pathway_memberships)
    
    if plots:
        null_distribution = ora_null_distribution(query_memberships, pathway_memberships, n_permutations, n_jobs)
        p_value = ora_p_value(observed_intersection, null_distribution)
        return observed_intersection, null_distribution, p_value  # Return all three values if plots are requested
    else:
        null_distribution = ora_null_distribution(query_memberships, pathway_memberships, n_permutations // 10, n_jobs)
        p_value = ora_p_value(observed_intersection, null_distribution)
        return observed_intersection, p_value  # Return only observed intersection and p-value

def ora_plot_null_distribution(pathway, observed_score, null_distribution, p_value, plot_path, query_membership_type, pathway_membership_type, dataset_name):
    '''Plot the null distribution of intersection sizes.'''
    plt.figure(figsize=(8, 6))
    plt.hist(null_distribution, bins=30, alpha=0.7, color='gray', edgecolor='black')
    plt.axvline(observed_score, color='red', linestyle='--', linewidth=2, label=f'Observed Score = {observed_score:.2f}')
    plt.title(f'Null Distribution for {pathway}\n(Query: {query_membership_type}, Pathway: {pathway_membership_type})')
    plt.xlabel('Fuzzy Intersection Size')
    plt.ylabel('Frequency')
    plt.annotate(f'P-value = {p_value:.4f}', xy=(0.7, 0.9), xycoords='axes fraction', fontsize=12, color='red')
    plt.legend()
    plt.tight_layout()
    
    folder_name = f"{query_membership_type}_{pathway_membership_type}"
    full_plot_path = os.path.join(plot_path, 'null_distributions_product', folder_name)
    os.makedirs(full_plot_path, exist_ok=True)
    
    file_name = f"{pathway}_{dataset_name}_{query_membership_type}_{pathway_membership_type}.png"
    plt.savefig(os.path.join(full_plot_path, file_name))  # Save the plot to the specified directory
    plt.close()  # Close the plot

def fuzzy_ora(
    query_file: str,
    pathway_file: str,
    query_membership_type: str = 'Crisp_Membership',
    pathway_membership_type: str = 'Crisp_Membership',
    n_permutations: int = 1000,
    n_jobs: int = cpu_count() - 1,
    output_path: Optional[str] = None,
    plots: bool = False,
    dataset_name: str = '',
    pathway_ids: Optional[List[str]] = None
) -> pd.DataFrame:
    '''Perform fuzzy Over-Representation Analysis (ORA) with optional plotting.'''
    
    # Load query and pathway data
    query_df = load_query(query_file, query_membership_type)
    pathway_df = load_pathways(pathway_file, pathway_membership_type)

    # Filter pathways by IDs if provided
    if pathway_ids:
        pathway_df = pathway_df[pathway_df['Pathway_Name'].isin(pathway_ids)]

    results = []  # Initialize results list
    num_pathways = len(pathway_df)  # Total number of pathways

    # Iterate over each pathway and compute stats
    for idx, pathway in tqdm.tqdm(pathway_df.iterrows(), total=num_pathways, desc="Processing Pathways"):
        if plots:
            observed_intersection, null_distribution, p_value = fuzzy_ora_compute_stats(
                pathway, query_df, n_permutations, n_jobs, plots=True
            )
            plot_path = os.path.join(output_path, dataset_name) if output_path else os.path.join('.', dataset_name)
            ora_plot_null_distribution(
                pathway['Pathway_Name'], observed_intersection, null_distribution, p_value, 
                plot_path, query_membership_type, pathway_membership_type, dataset_name
            )
        else:
            observed_intersection, p_value = fuzzy_ora_compute_stats(
                pathway, query_df, n_permutations, n_jobs, plots=False
            )

        # Append results to the results list
        results.append({
            'Pathway_Name': pathway['Pathway_Name'],
            'Description': pathway['Description'],
            'Observed_Intersection': observed_intersection,
            'p_value': p_value
        })

    results_df = pd.DataFrame(results)  # Create a DataFrame from the results
    results_df.sort_values('p_value', ascending=True, inplace=True)  # Sort results by p-value
    
    # Save results to a CSV file if output path is provided
    if output_path:
        output_file = os.path.join(output_path, f'{dataset_name}_{query_membership_type}_{pathway_membership_type}_results.csv')
        results_df.to_csv(output_file, index=False)  # Save DataFrame to CSV
    
    return results_df  # Return the results DataFrame


def fuzzy_ora_main():
    """Parse command-line arguments and execute the fuzzy_ora function."""
    parser = argparse.ArgumentParser(description="Run fuzzy ORA analysis.")
    parser.add_argument('-q', '--query_file', required=True, help="Path to the query file.")
    parser.add_argument('-p', '--pathway_file', required=True, help="Path to the pathway file.")
    parser.add_argument('-q_name', '--query_membership_type', default='Crisp_Membership', help="Query membership type.")
    parser.add_argument('-p_name', '--pathway_membership_type', default='Crisp_Membership', help="Pathway membership type.")
    parser.add_argument('-o', '--output_path', default=None, help="Output directory path.")
    parser.add_argument('-d', '--dataset_name', default='dataset', help="Dataset name for output files.")
    parser.add_argument('-n', '--n_permutations', type=int, default=1000, help="Number of permutations.")
    parser.add_argument('-j', '--n_jobs', type=int, default=cpu_count()-1, help="Number of parallel jobs.")
    parser.add_argument('--plots', action='store_true', help="Generate null distribution plots.")
    
    args = parser.parse_args()
    
    fuzzy_ora(
        query_file=args.query_file,
        pathway_file=args.pathway_file,
        query_membership_type=args.query_membership_type,
        pathway_membership_type=args.pathway_membership_type,
        n_permutations=args.n_permutations,
        n_jobs=args.n_jobs,
        output_path=args.output_path,
        plots=args.plots,
        dataset_name=args.dataset_name
    )

if __name__ == "__main__":
    fuzzy_ora_main()