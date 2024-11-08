''' 
Description: Script for preprocessing single-cell transcriptomics data and performing ANOVA and log fold change calculations.

'''
import scanpy as sc  
import os  
import pandas as pd  
import numpy as np 
from scipy.stats import f_oneway 
from joblib import Parallel, delayed, cpu_count  
from qvalue import qvalues  
from tqdm import tqdm 
import argparse

def display_possible_labels(metadata):
    """Display possible disease labels in the metadata."""
    unique_labels = metadata['Factor Value[disease]'].unique()
    print("Possible disease labels:")
    for label in unique_labels:
        print(label)

def anova_proc_scipy(data_groupby, gene_name):
    """Perform ANOVA test on gene expression data for a specific gene."""
    f, p = f_oneway(*data_groupby[gene_name].apply(list))  # Perform ANOVA test
    return (gene_name, p, f)  # Return gene name, p-value, and F-statistic

def anova_genes_scipy(data, label, n_jobs):
    """Perform ANOVA across all genes, parallelized."""
    data['label'] = label  # Add label column to data
    data_gb = data.groupby('label', observed=False)  # Group data by label

    # Run ANOVA for each gene in parallel
    result = Parallel(n_jobs=n_jobs)(
        delayed(anova_proc_scipy)(data_gb, gene) for gene in tqdm(data.columns[:-1], desc="Running ANOVA", unit="gene")
    )

    # Store results in DataFrame and set gene ID as index
    result_df = pd.DataFrame(data=result, columns=['Ensembl_ID', 'p', 'f']).set_index('Ensembl_ID')
    return result_df

def log_fold_change_proc(data_groupby, gene_name, order, positive_label):
    """Calculate log fold change for a specific gene."""
    unpacked = data_groupby[gene_name].apply(list)  # Extract gene expression data
    # Determine positive and negative groups based on label order
    if order[0] == positive_label:
        mean1 = np.mean(unpacked[order[0]])  # Positive label
        mean2 = np.mean(unpacked[order[1]])  # Negative label
    else:
        mean1 = np.mean(unpacked[order[1]])  # Positive label
        mean2 = np.mean(unpacked[order[0]])  # Negative label
    fc = mean1 - mean2  # Calculate log fold change
    return (gene_name, fc)  # Return gene name and log fold change

def log_fold_change(data, label, n_jobs, positive_label):
    """Calculate log fold change for all genes, parallelized."""
    data['label'] = label  # Add label column to data
    data_gb = data.groupby('label', observed=False)  # Group data by label
    order = label.cat.categories  # Get order of label categories

    # Run log fold change calculation for each gene in parallel
    result = Parallel(n_jobs=n_jobs)(
        delayed(log_fold_change_proc)(data_gb, gene, order, positive_label) for gene in tqdm(data.columns[:-1], desc="Calculating Log Fold Change", unit="gene")
    )

    # Store results in DataFrame and set gene ID as index
    result_df = pd.DataFrame(data=result, columns=['Ensembl_ID', 'logfc']).set_index('Ensembl_ID')
    return result_df

def sc_exp_data_preprocessing(study, negative_label, positive_label, condition, output_path):
    """Process expression data, run ANOVA, and calculate log fold change."""
    # Load single-cell data from Expression Atlas
    adata = sc.datasets.ebi_expression_atlas(study)

    # Basic filtering of cells and genes
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # Normalize data for library size and log transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Extract metadata and gene expression data as DataFrames
    metadata = adata.obs
    adata_df = adata.to_df()

    # Ensure the disease label is categorical
    metadata = metadata.dropna(subset=['Factor Value[disease]'])
    metadata['Factor Value[disease]'] = pd.Categorical(metadata['Factor Value[disease]'], categories=[negative_label, positive_label], ordered=True)

    # Display possible labels before filtering
    display_possible_labels(metadata)

    print("Unique labels in metadata:")
    print(metadata['Factor Value[disease]'].unique())
    
    # Keep only the specified positive and negative labels
    labels_of_interest = [negative_label, positive_label]
    metadata_filtered = metadata[metadata['Factor Value[disease]'].isin(labels_of_interest)]
    
    adata_filtered = adata_df.loc[metadata_filtered.index]
    label_filtered = metadata_filtered['Factor Value[disease]']

    # Perform ANOVA across all genes
    n_jobs = cpu_count() - 1  # Set number of jobs for parallel processing
    
    # After filtering
    print("Filtered metadata:")
    print(metadata_filtered['Factor Value[disease]'].value_counts())

    results = anova_genes_scipy(adata_filtered, label_filtered, n_jobs)

    # Apply q-value correction for multiple testing
    results = qvalues(results)

    # Perform log fold change calculation
    logfc = log_fold_change(adata_filtered, label_filtered, n_jobs, positive_label)
    results['logfc'] = logfc['logfc']  # Add log fold change to results

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Save results to CSV
    output_file = os.path.join(output_path, f'{condition}_{study}.csv')
    results.to_csv(output_file, index=True)
    print(f"Results saved to {output_file}")

def sc_exp_data_preprocessing_main():
    """CLI for preprocessing single-cell data and performing ANOVA and log fold change calculations."""
    parser = argparse.ArgumentParser(description="Process single-cell data and perform ANOVA and log fold change calculations.")
    parser.add_argument('--study', type=str, required=True, help="Study ID to load from Expression Atlas (e.g., E-GEOD-111727)")
    parser.add_argument('--negative_label', type=str, required=True, help="Label for negative group (e.g., control)")
    parser.add_argument('--positive_label', type=str, required=True, help="Label for positive group (e.g., disease)")
    parser.add_argument('--condition', type=str, required=True, help="Short disease name for output files (e.g., HIV)")
    parser.add_argument('--output_path', type=str, default="../../data/single_cell", help="Path to save results")

    args = parser.parse_args()
    sc_exp_data_preprocessing(args.study, args.negative_label, args.positive_label, args.condition, args.output_path)

if __name__ == "__main__":
    sc_exp_data_preprocessing_main()

