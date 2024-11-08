'''
Description: Script for deriving pathway membership values for genes from pathway overlap, topology and STRING association scores. 

'''
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed

def crisp_membership(pathways_df):
    """Add a Crisp Membership column with a value of 1."""
    pathways_df['Crisp_Membership'] = 1
    return pathways_df

def overlap_membership(pathways_df):
    """Calculate Overlap Membership based on gene counts in pathways."""
    gene_counts = pathways_df['Ensembl_ID'].value_counts()
    max_count = gene_counts.max()  # Assign max count from gene counts
    total_pathways = pathways_df['Pathway_Name'].nunique()
    
    # Print max_count and total_pathways to the console
    print(f"Max count of gene occurrences across pathways: {max_count}")
    print(f"Total number of unique pathways: {total_pathways}")
    
    pathways_df['Overlap_Membership'] = (1 + total_pathways - pathways_df['Ensembl_ID'].map(gene_counts)) / total_pathways
    pathways_df['Strict_Overlap_Membership'] = (1 + max_count - pathways_df['Ensembl_ID'].map(gene_counts)) / max_count
    return pathways_df


def string_membership_single_pathway(pathway_name, pathway_data, string_scores):
    """Process a single pathway to determine expansion membership based on STRING scores."""
    pathway_genes = set(pathway_data['Ensembl_ID'])
    
    # Filter scores for Gene2 in pathway genes
    scores = string_scores[string_scores['Gene2'].isin(pathway_genes)]
    
    # Filter Gene1 to exclude pathway genes
    scores = scores[~scores['Gene1'].isin(pathway_genes)]
    
    if scores.empty:
        return None  # Return None if no scores are found
    
    # Average the scores for Gene1
    gene1_avg_scores = scores.groupby('Gene1')['string_score'].mean() / 1000
    
    # Filter for scores >= 0.4
    gene1_avg_scores = gene1_avg_scores[gene1_avg_scores >= 0.4]
    
    # Scale scores to range from 0 to 0.8
    if not gene1_avg_scores.empty:
        scaled_scores = ((gene1_avg_scores - 0.4) / (1 - 0.4)) * 0.8
        
        # Collect results for non-pathway genes
        return [
            {
                'Pathway_Name': pathway_name,
                'Ensembl_ID': gene,
                'Expansion_Membership': score,
                'Description': pathway_data['Description'].iloc[0]
            }
            for gene, score in scaled_scores.items()
        ]
    return None  # Return None if no valid genes are found

def string_membership(pathways_df, string_path):
    """Determine STRING membership for each pathway and return updated pathways DataFrame."""
    string_scores = pd.read_csv(string_path, sep="\t")
    
    # Parallel processing of pathways
    results = Parallel(n_jobs=-1)(
        delayed(string_membership_single_pathway)(pathway_name, pathway_data, string_scores)
        for pathway_name, pathway_data in tqdm(pathways_df.groupby('Pathway_Name'), desc="Processing Pathways")
    )

    # Flatten the results while filtering out None values
    string_genes = [gene for sublist in results if sublist for gene in sublist]

    # Create DataFrame from collected results
    string_genes_df = pd.DataFrame(string_genes)

    # If there are no new genes, return the original pathways_df
    if string_genes_df.empty:
        return pathways_df
    
    # Ensure all original columns are kept
    for col in pathways_df.columns:
        if col not in string_genes_df.columns:
            string_genes_df[col] = pd.NA

    # Concatenate the original pathways_df with the new string_genes_df
    pathway_df = pd.concat([pathways_df, string_genes_df], ignore_index=True)
    
    return pathway_df

def topology_membership(pathways_df, topology_path):
    """Calculate topology memberships from centrality data and return updated pathways DataFrame."""
    topology_df = pd.read_csv(topology_path, sep="\t")
    
    # Identify centrality columns, excluding specific identifier columns
    centrality_columns = [col for col in topology_df.columns if col not in ['Pathway_Name', 'Description', 'Ensembl_ID']]
    
    # Create a new DataFrame to store topology membership results
    topology_memberships = pd.DataFrame(columns=['Pathway_Name', 'Description', 'Ensembl_ID'])

    for centrality in centrality_columns:
        # Extract unique non-zero values for the current centrality
        non_zero_values = topology_df[centrality][topology_df[centrality] > 0].unique()
        
        # Check if there are at least two distinct non-zero values
        if len(non_zero_values) < 2:
            print(f"Not enough distinct non-zero values for {centrality}. Skipping.")
            topology_memberships[f"{centrality}_Membership"] = np.nan  # Assign NaN if insufficient data
            continue

        # Sort the unique non-zero values and get the two smallest
        sorted_non_zero_values = np.sort(non_zero_values)
        lowest = sorted_non_zero_values[:2]  # Get the two smallest distinct non-zero values
        offset = lowest[1] - lowest[0]  # Calculate the offset (e.g., 11 - 10 = 1)
        
        # Adjust the centrality values and normalize
        adjusted_values = topology_df[centrality] + offset
        max_value = adjusted_values.max()
        
        # Calculate membership values, avoiding division by zero
        if max_value > 0:
            membership_values = adjusted_values / max_value
        else:
            membership_values = np.zeros(len(adjusted_values))
        
        # Add the new membership column to results
        topology_memberships[f"{centrality}_Membership"] = membership_values
    
    # Keep common identifier columns in topology_memberships
    topology_memberships[['Pathway_Name', 'Description', 'Ensembl_ID']] = topology_df[['Pathway_Name', 'Description', 'Ensembl_ID']]
    
    # Merge topology membership results with pathways_df
    merged_df = pd.merge(pathways_df, topology_memberships, on=['Pathway_Name', 'Description', 'Ensembl_ID'], how='left')
    
    return merged_df

def pathway_memberships(pathway_file, output_path, membership_types=["crisp", "overlap", "string", "topology"], string_path=None, topology_path=None):
    """Main function to process pathway memberships based on selected types."""
    pathways_df = pd.read_csv(pathway_file, sep='\t')

    if 'crisp' in membership_types:
        pathways_df = crisp_membership(pathways_df)
    if 'overlap' in membership_types:
        pathways_df = overlap_membership(pathways_df)
    if 'string' in membership_types and string_path:
        pathways_df = string_membership(pathways_df, string_path)
    if 'topology' in membership_types and topology_path:
        pathways_df = topology_membership(pathways_df, topology_path)

    # Construct output file name based on pathway_file
    if output_path is not None:
        # Extract the base name without the '_parsed_ensembl' part
        base_name = pathway_file.split('/')[-1].replace('_parsed_ensembl.tsv', '')
        output_file_name = f"{base_name}_pathway_memberships.tsv"
        full_output_path = f"{output_path}/{output_file_name}"

        pathways_df.to_csv(full_output_path, sep="\t", index=False)

    return pathways_df


def pathway_membership_main():
    """Parse command-line arguments and execute the pathway_memberships function."""
    parser = argparse.ArgumentParser(description="Process pathway memberships.")
    parser.add_argument('-p', '--pathway_file', required=True, help="Path to the pathway file.")
    parser.add_argument('-o', '--output_path', default=None, help="Path to save the output file.")
    parser.add_argument('-s', '--string_path', default=None, help="Path to the STRING scores file.")
    parser.add_argument('-t', '--topology_path', default=None, help="Path to the topology centrality data file.")
    parser.add_argument('-m', '--membership_types', nargs='+', default=["crisp", "overlap"], help="Types of memberships to calculate (default: crisp overlap string topology).")

    args = parser.parse_args()

    # Execute the pathway_memberships function with the parsed arguments
    pathway_memberships(
        pathway_file=args.pathway_file,
        output_path=args.output_path,
        string_path=args.string_path,
        topology_path=args.topology_path,
        membership_types=args.membership_types
    )

if __name__ == "__main__":
    pathway_membership_main()

