'''
Description: Script for performing GSEA with phenotype permutations for fuzzy pathway memberships.

'''
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm

def shuffle_phenotype_labels(df):
    """Randomly shuffles phenotype labels among 'control' and 'disease' columns."""
    shuffled_df = df.copy()
    
    # Ensure 'Ensembl_ID' is a column
    if shuffled_df.index.name == 'Ensembl_ID':
        shuffled_df.reset_index(inplace=True)
    
    phenotype_cols = [col for col in shuffled_df.columns if col.startswith(('control', 'disease'))]
    shuffled_col_names = np.random.permutation(phenotype_cols)
    shuffled_df = shuffled_df.rename(columns=dict(zip(phenotype_cols, shuffled_col_names)))
    
    return shuffled_df

def calculate_statistics_from_expression_dataframe(df):
    """Calculates T-statistics for each gene based on expression data."""
    
    # Ensure 'Ensembl_ID' is a column and set as index for easier manipulation
    if 'Ensembl_ID' in df.columns:
        df.set_index('Ensembl_ID', inplace=True)
    
    # Extract column names corresponding to control and disease conditions
    control_cols = [col for col in df.columns if col.startswith('control')]
    disease_cols = [col for col in df.columns if col.startswith('disease')]
    
    # Convert relevant columns to numeric and drop rows with NaNs in these columns
    df[control_cols + disease_cols] = df[control_cols + disease_cols].apply(pd.to_numeric, errors='coerce')
    
    # Extract data as NumPy arrays
    control_data = df[control_cols].values
    disease_data = df[disease_cols].values
    
    # Perform t-test across rows
    t_stats, _ = ttest_ind(control_data, disease_data, axis=1, equal_var=False, nan_policy='omit')
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Ensembl_ID': df.index,
        't': t_stats
    })
    
    # Sort results by T-statistic in descending order
    results_df = results_df.sort_values(by='t', ascending=False).reset_index(drop=True)
    
    return results_df

def fuzzy_gsea_score(ranked_list, pathways_dict):
    """Calculates enrichment scores for pathways using a ranked list and a dataset of pathways."""
    enrichment_scores = {}
    plotting_values_all = {}
    N = len(ranked_list)
    ranked_gene_list = list(ranked_list.keys())

    for pathway_name, pathway_info in pathways_dict.items():
        gene_ids = set(pathway_info['genes'].keys())
        correlation = {gene_id: ranked_list.get(gene_id, 0) for gene_id in gene_ids}
        N_r = sum(abs(correlation.get(gene_id, 0)) * pathway_info['genes'].get(gene_id, 1) for gene_id in gene_ids)
        N_misses = N - len(gene_ids)
        P_hit = 0
        P_miss = 1
        counter = 0
        enrichment_score = 0.0
        plotting_values = []

        for idx, gene_id in enumerate(ranked_gene_list):
            if gene_id in gene_ids:
                membership_value = pathway_info['genes'].get(gene_id, 1)
                P_hit += abs(correlation.get(gene_id, 0) * membership_value) / N_r
                counter += 1
            P_miss = ((idx - counter) + 1) / N_misses

            # Update enrichment score if the current score is higher
            if abs(P_hit - P_miss) > abs(enrichment_score):
                enrichment_score = P_hit - P_miss

            # Track the enrichment score for plotting
            plotting_values.append(P_hit - P_miss)

        enrichment_scores[pathway_name] = enrichment_score
        plotting_values_all[pathway_name] = plotting_values

    return enrichment_scores, plotting_values_all

def permute_and_calculate_null_distribution(expression_df, pathways_dict, n_permutations=1000, n_jobs=cpu_count() - 1):
    """Generates a null distribution of enrichment scores by permuting column labels."""
    null_distributions = {pathway: [] for pathway in pathways_dict.keys()}

    def process_permutation(_):
        """Perform a single permutation and return the enrichment scores."""
        shuffled_df = shuffle_phenotype_labels(expression_df)
        ranked_df = calculate_statistics_from_expression_dataframe(shuffled_df)
        ranked_list = ranked_df.set_index('Ensembl_ID')['t'].to_dict()
        enrichment_scores, _ = fuzzy_gsea_score(ranked_list, pathways_dict)
        return enrichment_scores

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_permutation)(i) for i in tqdm(range(n_permutations), desc="Permutations")
    )

    for enrichment_scores in results:
        for pathway, score in enrichment_scores.items():
            null_distributions[pathway].append(score)

    return null_distributions

def calculate_p_value(observed_score, null_distribution):
    """Calculates the two-sided p-value for an observed score given a null distribution."""
    count = sum(1 for score in null_distribution if abs(score) >= abs(observed_score))
    p_value = count / len(null_distribution)
    return p_value

def save_enrichment_plots(plotting_values_all, observed_scores, pathways_dict, plot_path, membership):
    """Saves enrichment plots for each pathway with titles including the pathway description."""
    membership_path = os.path.join(plot_path, membership, 'enrichment_plots')
    os.makedirs(membership_path, exist_ok=True)

    for pathway, plotting_values in plotting_values_all.items():
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(plotting_values)), plotting_values, label='Enrichment Score')
        final_score = observed_scores[pathway]
        plt.axhline(y=final_score, color='r', linestyle='--', label=f'Final Enrichment Score: {final_score:.2f}')
        pathway_description = pathways_dict[pathway]['description']
        plt.title(f'GSEA Plot for {pathway}\nDescription: {pathway_description}\nMembership: {membership}')
        plt.xlabel('Rank')
        plt.ylabel('Enrichment Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(membership_path, f'{pathway}_{membership}_gsea_plot.png'))
        plt.close()

def save_null_distribution_plots(null_distributions, observed_scores, pathways_dict, plot_path, membership):
    """Saves histograms of the null distributions with the observed enrichment score, p-value, and pathway description."""
    null_dist_path = os.path.join(plot_path, membership, 'null_distributions')
    os.makedirs(null_dist_path, exist_ok=True)

    for pathway, null_distribution in null_distributions.items():
        observed_score = observed_scores[pathway]
        p_value = calculate_p_value(observed_score, null_distribution)
        pathway_description = pathways_dict[pathway]['description']
        plt.figure(figsize=(10, 6))
        plt.hist(null_distribution, bins=30, alpha=0.75, color='blue', label='Null Distribution')
        plt.axvline(observed_score, color='red', linestyle='--', linewidth=2, label=f'Observed Score: {observed_score:.2f}')
        plt.title(f'Null Distribution for {pathway}\nDescription: {pathway_description}\nMembership: {membership}')
        plt.xlabel('Enrichment Score')
        plt.ylabel('Frequency')
        plt.text(0.95, 0.9, f'p-value: {p_value:.4f}', transform=plt.gca().transAxes, 
                 fontsize=12, verticalalignment='top', horizontalalignment='right')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(null_dist_path, f'{pathway}_{membership}_null_distribution.png'))
        plt.close()

def main(expression_path, pathway_file_path, output_path, plot_path=None, membership='Default_Membership', n_permutations=1000):
    expression_df = pd.read_csv(expression_path)                     
    ranked_df = calculate_statistics_from_expression_dataframe(expression_df)
    ranked_list = ranked_df.set_index('Ensembl_ID')['t'].to_dict()
    
    pathways_df = pd.read_csv(pathway_file_path, sep='\t')
    pathways_df = pathways_df.dropna(subset=[membership])
    pathways_dict = {}

    for _, row in pathways_df.iterrows():
        pathway_name = row['Pathway_Name']
        gene_id = row['Ensembl_ID']
        membership_value = row[membership]
        pathway_description = row.get('Description', '')
        if pathway_name not in pathways_dict:
            pathways_dict[pathway_name] = {'genes': {}, 'description': pathway_description}
        pathways_dict[pathway_name]['genes'][gene_id] = membership_value

    observed_scores, plotting_values_all = fuzzy_gsea_score(ranked_list, pathways_dict)
    null_distributions = permute_and_calculate_null_distribution(expression_df, pathways_dict, n_permutations)
    
    results = []

    for pathway, observed_score in observed_scores.items():
        null_distribution = null_distributions[pathway]
        p_value = calculate_p_value(observed_score, null_distribution)
        pathway_description = pathways_dict[pathway]['description']
        results.append({'Pathway_Name': pathway, 'Observed_Score': observed_score, 'p-value': p_value, 'Description': pathway_description})

    results_df = pd.DataFrame(results)
    results_df['Rank'] = results_df['p-value'].rank(method='first')  # Add rank column
    results_df = results_df.sort_values(by='p-value').reset_index(drop=True)  # Sort by p-value
    results_df.to_csv(os.path.join(output_path, f'gsea_{membership}_results.csv'), index=False)

    if plot_path:
        save_enrichment_plots(plotting_values_all, observed_scores, pathways_dict, plot_path, membership)
        save_null_distribution_plots(null_distributions, observed_scores, pathways_dict, plot_path, membership)

if __name__ == "__main__":
    main()