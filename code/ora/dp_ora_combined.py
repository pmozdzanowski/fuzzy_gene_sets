'''
Description: Script for performing Over-Representation Analysis (ORA) with a fuzzy pathway set using probabilities precomputed through dynamci programming.

'''

import numpy as np
import pandas as pd
import os
import argparse
import tqdm
import sys
from scipy.stats import hypergeom  
from typing import List, Optional


# Add the project root to sys.path for relative imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

from preprocessing.qvalue import qvalues


def load_pathway_ids(file_path):
    """
    Load pathway IDs from a file, categorized into 'cancer' and 'infection'.
    """
    ids = {'cancer': [], 'infection': []}
    with open(file_path, 'r') as file:
        current_category = None
        for line in file:
            line = line.strip()
            if line.startswith('#'):  # Identifies categories (cancer, infection)
                if 'cancer' in line:
                    current_category = 'cancer'
                elif 'infection' in line:
                    current_category = 'infection'
            elif line:
                ids[current_category].append(line)
    return ids

def load_query(query_file: str, query_membership_type: str) -> pd.DataFrame:
    """
    Load and clean query data from a file.
    Ensures required columns ('Ensembl_ID', membership type) are present.
    """
    if not os.path.isfile(query_file):  # Check if the query file exists
        raise FileNotFoundError(f"Query file '{query_file}' does not exist.")
    
    query_df = pd.read_csv(query_file, sep='\t')
    
    # Validate columns in the query file
    if 'Ensembl_ID' not in query_df.columns or query_membership_type not in query_df.columns:
        raise ValueError(f"Query file must contain 'Ensembl_ID' and '{query_membership_type}' columns.")
    
    query_df = query_df.dropna().rename(columns={query_membership_type: 'Query_Membership'})
    
    return query_df

def load_pathways(pathway_file: str, pathway_membership_type: str) -> pd.DataFrame:
    """
    Load and aggregate pathway data from a file.
    Excludes certain pathways and handles missing values.
    """
    if not os.path.isfile(pathway_file):  # Check if the pathway file exists
        raise FileNotFoundError(f"Pathway file '{pathway_file}' does not exist.")
    
    # Load pathway data from a tab-separated file
    pathway_df = pd.read_csv(
        pathway_file,
        sep='\t',
        usecols=['Pathway_Name', 'Description', 'Ensembl_ID', pathway_membership_type],
        dtype={'Ensembl_ID': str, pathway_membership_type: float}
    )
    
    # Validate required columns
    if 'Pathway_Name' not in pathway_df.columns or 'Description' not in pathway_df.columns or pathway_membership_type not in pathway_df.columns:
        raise ValueError(f"Pathway file must contain 'Pathway_Name', 'Description', and '{pathway_membership_type}' columns.")
    
    # Exclude pathways that should not be included
    exclude_pathways = {'hsa01100', 'hsa01200', 'hsa04740', 'hsa05168'}
    pathway_df = pathway_df[~pathway_df['Pathway_Name'].isin(exclude_pathways)]

    # Aggregate data by pathway, grouping by 'Pathway_Name'
    pathway_df = (pathway_df
                   .dropna()
                   .rename(columns={pathway_membership_type: 'Pathway_Membership'})
                   .groupby('Pathway_Name')
                   .agg({
                       'Description': 'first',  # Retain the first description for each pathway
                       'Ensembl_ID': list,      # List of unique Ensembl IDs for the pathway
                       'Pathway_Membership': list  # Membership values as lists
                   })
                   .reset_index())

    # Exclude pathways with too many genes (greater than 100)
    pathway_df['Gene_Count'] = pathway_df['Ensembl_ID'].apply(lambda x: len(set(x)))  # Count unique genes
    pathway_df = pathway_df[pathway_df['Gene_Count'] <= 150]  # Limit to pathways with 150 or fewer genes
    pathway_df = pathway_df.drop(columns=['Gene_Count'])  # Drop the gene count column used for filtering

    return pathway_df

def load_probabilities(probability_file: str) -> pd.DataFrame:
    """
    Load precomputed probabilities from a file.
    """
    if not os.path.isfile(probability_file):  # Check if the probability file exists
        raise FileNotFoundError(f"Probability file '{probability_file}' does not exist.")
    
    probabilities_df = pd.read_csv(probability_file, sep='\t', index_col=0)
    return probabilities_df

def ora_fuzzy_intersection(query_memberships, pathway_memberships):
    """
    Calculate the fuzzy intersection size between query and pathway memberships.
    This is the element-wise product of the memberships.
    """
    intersection = np.multiply(query_memberships, pathway_memberships)
    intersection_size = np.sum(intersection)
    return intersection_size

def dp_p_value(observed_intersection, probabilities_df, universe_size, pathway_size, query_size):
    """
    Calculate the p-value based on the observed intersection using precomputed probabilities.
    Uses a hypergeometric distribution for the p-value calculation.
    """
    # Ensure the pathway size is within the range of available probabilities
    if pathway_size > probabilities_df.shape[1]:
        print(f"Skipping pathway as its size {pathway_size} exceeds available columns in probabilities_df.")
        return None
    
    # Find the closest match in probabilities for the observed intersection
    try:
        closest_score = probabilities_df.index[probabilities_df.index.get_loc(observed_intersection)]  # Exact match
    except KeyError:
        # Find the nearest index if an exact match is not found
        closest_score = probabilities_df.index[(np.abs(probabilities_df.index - observed_intersection)).argmin()]
    
    score_row = probabilities_df.loc[closest_score, :]
    
    # Filter columns to those within the pathway size
    valid_columns = score_row.iloc[:pathway_size]
    
    # Raise an error if no valid columns are found
    if valid_columns.empty:
        raise ValueError(f"No valid columns found for pathway size {pathway_size} in probabilities_df.")
    
    # Calculate the p-value
    p_value = 0.0
    for index, value in valid_columns.items():
        try:
            index = int(index)  # Ensure the index is an integer
        except ValueError:
            raise ValueError(f"Index {index} could not be converted to an integer.")
        
        # Compute the probability for the intersection size and update the p-value
        prob = value * hypergeom.pmf(index, universe_size, pathway_size, query_size)
        p_value += prob
    
    return p_value

def dp_ora_compute_stats(pathway, query_df, probability_file, plots=False):
    """
    Compute statistics for fuzzy ORA for a given pathway and query DataFrame.
    This includes calculating observed intersection and p-values.
    """
    # Load the precomputed probabilities
    probabilities_df = load_probabilities(probability_file)

    pathway_df = pd.DataFrame({
        'Ensembl_ID': pathway['Ensembl_ID'],
        'Pathway_Membership': pathway['Pathway_Membership']
    })

    # Merge query data with pathway data
    merged_df = pd.merge(query_df, pathway_df, on='Ensembl_ID', how='left').fillna(0)
    
    query_memberships = merged_df['Query_Membership'].values
    pathway_memberships = merged_df['Pathway_Membership'].values
    
    # Calculate the observed intersection
    observed_intersection = ora_fuzzy_intersection(query_memberships, pathway_memberships)

    # Calculate pathway size
    pathway_size = merged_df[merged_df['Pathway_Membership'] > 0].shape[0]

    # Universe size is the number of rows in the merged dataframe
    universe_size = len(merged_df)
    # Query size is the number of genes with a Query_Membership value of 1
    query_size = len(merged_df[merged_df['Query_Membership'] == 1])
    
    # Calculate the p-value using the probabilities
    p_value = dp_p_value(observed_intersection, probabilities_df, universe_size, pathway_size, query_size)
    
    return observed_intersection, p_value

def dp_ora_combined(
    query_file: str,
    pathway_file: str,
    probability_folder: str,
    query_membership_type: str = 'Crisp_Membership',
    pathway_membership_type: str = 'Overlap_Membership',
    output_path: Optional[str] = None,
    dataset_name: str = '',
    pathway_ids: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Perform fuzzy Over-Representation Analysis (ORA) using precomputed probabilities.
    Computes the observed intersection and p-value for each pathway.
    """
    
    # Load query and pathway data
    query_df = load_query(query_file, query_membership_type)
    pathway_df = load_pathways(pathway_file, pathway_membership_type)

    # Filter pathways by the provided list of IDs, if any
    if pathway_ids:
        pathway_df = pathway_df[pathway_df['Pathway_Name'].isin(pathway_ids)]

    results = []
    
    # Set the path for the probabilities file
    probability_file = os.path.join(probability_folder, "combined_dp_probabilities.tsv")
    
    # Iterate over pathways and compute stats
    for idx, pathway in tqdm.tqdm(pathway_df.iterrows(), total=pathway_df.shape[0], desc="Processing pathways"):
        pathway = pathway[1]  # Get the row as a Series
        observed_intersection, p_value = dp_ora_compute_stats(pathway, query_df, probability_file)
        
        # Append results
        results.append({
            'Pathway_Name': pathway['Pathway_Name'],
            'Observed_Intersection': observed_intersection,
            'P_Value': p_value
        })
    
    # Convert results into DataFrame
    results_df = pd.DataFrame(results)

    # Save output if requested
    if output_path:
        results_df.to_csv(output_path, index=False)
    
    return results_df

def dp_ora_main():
    """Parse command-line arguments and execute the dp_ora function."""
    parser = argparse.ArgumentParser(description="Run fuzzy ORA analysis.")
    parser.add_argument('-q', '--query_file', required=True, help="Path to the query file.")
    parser.add_argument('-p', '--pathway_file', required=True, help="Path to the pathway file.")
    parser.add_argument('-prob_folder', '--probability_folder', required=True, help="Path to the folder containing pathway-specific probability files.")
    parser.add_argument('-q_name', '--query_membership_type', default='Crisp_Membership', help="Query membership type.")
    parser.add_argument('-p_name', '--pathway_membership_type', default='Crisp_Membership', help="Pathway membership type.")
    parser.add_argument('-o', '--output_path', default=None, help="Output directory path.")
    parser.add_argument('-d', '--dataset_name', default='dataset', help="Dataset name for output files.")
    parser.add_argument('-p_ids', '--pathway_ids', nargs='*', help="Specific pathway IDs to analyze.")

    args = parser.parse_args()
    
    dp_ora_combined(
        query_file=args.query_file,
        pathway_file=args.pathway_file,
        probability_folder=args.probability_folder,
        query_membership_type=args.query_membership_type,
        pathway_membership_type=args.pathway_membership_type,
        output_path=args.output_path,
        dataset_name=args.dataset_name,
        pathway_ids=args.pathway_ids
    )


if __name__ == "__main__":
    dp_ora_main()