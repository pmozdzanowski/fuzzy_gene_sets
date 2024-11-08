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


# Add the project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

from preprocessing.qvalue import qvalues


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


def load_query(query_file: str, query_membership_type: str) -> pd.DataFrame:
    """Load and clean query data from a file, ensuring necessary columns."""
    if not os.path.isfile(query_file):
        raise FileNotFoundError(f"Query file '{query_file}' does not exist.")
    
    query_df = pd.read_csv(query_file, sep='\t')
    
    if 'Ensembl_ID' not in query_df.columns or query_membership_type not in query_df.columns:
        raise ValueError(f"Query file must contain 'Ensembl_ID' and '{query_membership_type}' columns.")
    
    query_df = query_df.dropna().rename(columns={query_membership_type: 'Query_Membership'})
    
    return query_df


def load_pathways(pathway_file: str, pathway_membership_type: str) -> pd.DataFrame:
    """Load and aggregate pathway data, filtering unnecessary pathways."""
    if not os.path.isfile(pathway_file):
        raise FileNotFoundError(f"Pathway file '{pathway_file}' does not exist.")
    
    pathway_df = pd.read_csv(
        pathway_file,
        sep='\t',
        usecols=['Pathway_Name', 'Description', 'Ensembl_ID', pathway_membership_type],
        dtype={'Ensembl_ID': str, pathway_membership_type: float}
    )
    
    if 'Pathway_Name' not in pathway_df.columns or 'Description' not in pathway_df.columns or pathway_membership_type not in pathway_df.columns:
        raise ValueError(f"Pathway file must contain 'Pathway_Name', 'Description', and '{pathway_membership_type}' columns.")
    
    exclude_pathways = {'hsa01100', 'hsa01200', 'hsa04740', 'hsa05168'}
    pathway_df = pathway_df[~pathway_df['Pathway_Name'].isin(exclude_pathways)]

    pathway_df = (pathway_df
                   .dropna()
                   .rename(columns={pathway_membership_type: 'Pathway_Membership'})
                   .groupby('Pathway_Name')
                   .agg({
                       'Description': 'first',  
                       'Ensembl_ID': list,      
                       'Pathway_Membership': list  
                   })
                   .reset_index())
    
    pathway_df['Gene_Count'] = pathway_df['Ensembl_ID'].apply(lambda x: len(set(x)))  
    pathway_df = pathway_df[pathway_df['Gene_Count'] <= 150]
    pathway_df = pathway_df.drop(columns=['Gene_Count'])
    
    return pathway_df


def load_probabilities(probability_file: str) -> pd.DataFrame:
    """Load precomputed pathway probabilities."""
    if not os.path.isfile(probability_file):
        raise FileNotFoundError(f"Probability file '{probability_file}' does not exist.")
    
    probabilities_df = pd.read_csv(probability_file, sep='\t', index_col=0)
    return probabilities_df


def ora_fuzzy_intersection(query_memberships, pathway_memberships):
    """Calculate the fuzzy intersection size between query and pathway memberships."""
    intersection = np.multiply(query_memberships, pathway_memberships)
    intersection_size = np.sum(intersection)
    return intersection_size


def dp_p_value(observed_intersection, probabilities_df, universe_size, pathway_size, query_size):
    """Calculate the p-value for the observed intersection using precomputed probabilities."""
    try:
        closest_score = probabilities_df.index[probabilities_df.index.get_loc(observed_intersection)]  
    except KeyError:
        closest_score = probabilities_df.index[(np.abs(probabilities_df.index - observed_intersection)).argmin()]

    score_row = probabilities_df.loc[closest_score, :]  
    
    if score_row.empty:
        raise ValueError(f"No entry found for observed intersection size: {observed_intersection}")

    p_value = 0
    for index, value in score_row.items():  
        try:
            index = int(index)
        except ValueError:
            raise ValueError(f"Index {index} could not be converted to an integer.")

        prob = value * hypergeom.pmf(index, universe_size, pathway_size, query_size)
        p_value += prob

    return p_value 


def dp_ora_compute_stats(pathway, query_df, probability_file, plots=False):
    """Compute statistics for fuzzy ORA for a given pathway and query DataFrame."""
    probabilities_df = load_probabilities(probability_file)

    pathway_df = pd.DataFrame({
        'Ensembl_ID': pathway['Ensembl_ID'],
        'Pathway_Membership': pathway['Pathway_Membership']
    })

    merged_df = pd.merge(query_df, pathway_df, on='Ensembl_ID', how='left').fillna(0)
    
    query_memberships = merged_df['Query_Membership'].values
    pathway_memberships = merged_df['Pathway_Membership'].values
    
    observed_intersection = ora_fuzzy_intersection(query_memberships, pathway_memberships)

    pathway_size = merged_df[merged_df['Pathway_Membership'] > 0].shape[0]

    universe_size = len(merged_df)
    query_size = len(merged_df[merged_df['Query_Membership'] == 1])
    
    p_value = dp_p_value(observed_intersection, probabilities_df, universe_size, pathway_size, query_size)
    
    return observed_intersection, p_value


def dp_ora(
    query_file: str,
    pathway_file: str,
    probability_folder: str,
    query_membership_type: str = 'Crisp_Membership',
    pathway_membership_type: str = 'Overlap_Membership',
    output_path: Optional[str] = None,
    dataset_name: str = '',
    pathway_ids: Optional[List[str]] = None
) -> pd.DataFrame:
    """Perform fuzzy Over-Representation Analysis (ORA) using precomputed probabilities for each pathway."""
    
    query_df = load_query(query_file, query_membership_type)
    pathway_df = load_pathways(pathway_file, pathway_membership_type)

    if pathway_ids:
        pathway_df = pathway_df[pathway_df['Pathway_Name'].isin(pathway_ids)]

    results = []
    
    for idx, pathway in tqdm.tqdm(pathway_df.iterrows(), total=len(pathway_df), desc="Processing Pathways"):
        pathway_name = pathway['Pathway_Name']

        probability_file = os.path.join(probability_folder, f"{pathway_name}_dp_probabilities.tsv")
        
        if not os.path.exists(probability_file):
            print(f"Warning: Probability file for pathway '{pathway_name}' not found: {probability_file}. Skipping this pathway.")
            continue

        observed_intersection, p_value = dp_ora_compute_stats(pathway, query_df, probability_file)

        results.append({
            'Pathway_Name': pathway_name,
            'Description': pathway['Description'],
            'Observed_Intersection': observed_intersection,
            'p_value': p_value
        })

    results_df = pd.DataFrame(results).sort_values('p_value').reset_index(drop=True)
    results_df['Rank'] = results_df['p_value'].rank(method='min').astype(int)
    results_df['p_value'] = results_df['p_value'].astype(float)  
    
    results_df = qvalues(results_df, p_col="p_value", q_col="q_value")

    results_df['p_value'] = results_df['p_value'].apply(lambda x: f"{x:.4e}")
    results_df['q_value'] = results_df['q_value'].apply(lambda x: f"{x:.4e}")

    if output_path:
        results_folder = os.path.join(output_path, dataset_name)
        os.makedirs(results_folder, exist_ok=True)
        output_file = os.path.join(results_folder, f'{dataset_name}_{query_membership_type}_{pathway_membership_type}_dp_results.csv')
        results_df.to_csv(output_file, index=False)
    
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
    parser.add_argument('-d', '--dataset_name', default='default_dataset', help="Dataset name.")
    parser.add_argument('-ids', '--pathway_ids', nargs='*', default=None, help="Specific pathway IDs to process.")
    
    args = parser.parse_args()
    
    dp_ora(
        query_file=args.query_file,
        pathway_file=args.pathway_file,
        probability_folder=args.probability_folder,
        query_membership_type=args.query_membership_type,
        pathway_membership_type=args.pathway_membership_type,
        output_path=args.output_path,
        dataset_name=args.dataset_name,
        pathway_ids=args.pathway_ids
    )


if __name__ == '__main__':
    dp_ora_main()
