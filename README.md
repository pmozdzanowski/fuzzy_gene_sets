# Improving on traditional gene set enrichment analysis methods using fuzzy set theory

## General Info
Pathway enrichment analysis is a widely used method for interpreting high-throughput omics data by identifying biological pathways that are significantly associated with differences in gene expression between two conditions. Common techniques for pathway analysis include Over-Representation Analysis (ORA) and Gene Set Enrichment Analysis (GSEA). While effective, these methods have limitations, particularly in their failure to account for pathway topology, gene overlap, and the inherent incompleteness of pathway annotations.

To overcome these limitations, we apply fuzzy set theory to enhance the functionality of ORA and GSEA. In fuzzy set theory, rather than classifying elements as strictly belonging to a set or not, each element receives a degree of membership between 0 and 1. This allows for a more nuanced representation of gene involvement in pathways. We show how to transform traditional pathway gene sets into fuzzy sets by calculating gene membership values based on multiple factors, including pathway overlap, topology, and functional association scores.

Additionally, we address the limitation of ORA's reliance on a fixed significance threshold by converting the query set of differentially expressed genes into a fuzzy set. Membership values are derived from statistical measures such as p-values and fold changes, allowing for a more flexible and dynamic representation of gene sets. This fuzzy approach enables a more customized pathway analysis, enabling researchers to select membership functions tailored to their specific context.

## Project Overview
This repository provides a pipeline for performing fuzzy pathway analysis. It contains the following modules:

1. **preprocessing: Preprocessing and Differential Expression Analysis for Single-Cell Expression Atlas Data**:
    - Code for preprocessing single-cell RNA sequencing data and performing differential expression analysis to identify genes with significant changes in expression between conditions.

2. **query_membership: Computing Query Memberships from Differential Expression Analysis Results**:
    - Code for converting the results of differential expression analysis into a fuzzy query set by calculating membership values based on statistical significance (e.g., p-values) and fold changes.

3. **pathway_membership: Pathway Memberships from Overlap, Topology, and STRING Interaction Scores**:
    - Code for deriving pathway memberships from multiple sources of information, including gene set overlap, pathway topology, and protein-protein interaction data from STRING.

4. **ora and gsea: ORA and GSEA for Fuzzy Sets**:
    - Implementations of Over-Representation Analysis (ORA) and Gene Set Enrichment Analysis (GSEA) adapted for fuzzy sets.

5. **dynamic_programming: Dynamic Programming for Exact p-Value Computation**:
    - To avoid the computational burden of permutations, a dynamic programming approach is implemented to compute exact p-values when the query set is crisp and the pathway set is fuzzy.

6. **bias_under_null: Assessing Bias Under the Null Hypothesis**:
    - Code for assessing whether pathway enrichment methods are biased toward certain pathways by generating p-value distributions for randomly sampled query sets.
