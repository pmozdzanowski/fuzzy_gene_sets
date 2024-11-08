'''
Description: The purpose of this script is to generate histograms and cumulative distribution functions (ECDFs) for null distributions of p-values

'''

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse

# Function to plot histograms and CDFs
def plot_histogram(p_values, output_dir, filename):
    """Plots and saves a histogram of p-values."""
    plt.figure(figsize=(10, 6))
    plt.hist(p_values, bins=100, alpha=0.7, color='blue')
    plt.title('Histogram of p-values')
    plt.xlabel('p-value')
    plt.ylabel('Frequency')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_ecdf(p_values, output_dir, filename):
    """Plots and saves an ECDF of p-values."""
    sorted_p_values = np.sort(p_values)
    ecdf = np.arange(1, len(sorted_p_values) + 1) / len(sorted_p_values)

    plt.figure(figsize=(10, 6))
    plt.step(sorted_p_values, ecdf, where='post', color='green', label='Empirical CDF')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Expected CDF (Uniform)')
    plt.title('ECDF of p-values')
    plt.xlabel('p-value')
    plt.ylabel('Cumulative Probability')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# Function to process and generate plots for p-values
def generate_plots(null_distribution_path, output_dir_hist, output_dir_cdf):
    """Generates histograms and ECDFs for p-values in the given null distribution."""
    
    # Step 1: Load the data
    results = pd.read_csv(null_distribution_path, sep='\t', index_col=0)
    
    # Step 2: Extract p-value columns
    p_value_columns = [col for col in results.columns if '_p_value' in col]

    # Ensure output directories exist for saving plots
    os.makedirs(output_dir_hist, exist_ok=True)
    os.makedirs(output_dir_cdf, exist_ok=True)

    # Step 3: Create and save an overall histogram of all p-values
    all_p_values = results[p_value_columns].values.flatten()
    all_p_values = all_p_values[~np.isnan(all_p_values)]  # Remove NaN values
    plot_histogram(all_p_values, output_dir_hist, 'overall_p_value_histogram.png')

    # Step 4: Create and save an overall ECDF of all p-values
    plot_ecdf(all_p_values, output_dir_cdf, 'overall_ecdf_p_values.png')

    # Step 5: Create Histograms and ECDFs per Pathway
    pathway_names = results.index
    for pathway in pathway_names:
        # Extract p-values for the current pathway, skipping NaNs
        pathway_p_values = results.loc[pathway, p_value_columns].values.flatten()
        pathway_p_values = pathway_p_values[~np.isnan(pathway_p_values)]  # Remove NaN values
        
        # Skip pathway if all values are NaN
        if len(pathway_p_values) == 0:
            continue
        
        # Step 6: Plot Histogram for the current pathway
        plot_histogram(pathway_p_values, output_dir_hist, f'{pathway}_histogram.png')

        # Step 7: Plot ECDF for the current pathway
        plot_ecdf(pathway_p_values, output_dir_cdf, f'{pathway}_ecdf_p_values.png')

# Main function to parse arguments and call processing function
def main():
    # Step 1: Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate histograms and ECDFs from null distribution data")
    parser.add_argument("null_distribution_path", help="Path to the null distribution CSV file")
    parser.add_argument("output_dir_hist", help="Directory to save histogram plots")
    parser.add_argument("output_dir_cdf", help="Directory to save ECDF plots")

    args = parser.parse_args()

    # Step 2: Generate and save plots
    generate_plots(args.null_distribution_path, args.output_dir_hist, args.output_dir_cdf)

# Run the main function if the script is executed
if __name__ == "__main__":
    main()

