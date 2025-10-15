# SpaMCI-DL: A hybrid deep learning framework for integrated identification of domains and spatially variable genes in spatial transcriptomics

<img width="2144" height="1150" alt="image" src="https://github.com/user-attachments/assets/be35c3db-7160-42c1-8f2c-e47fc915372e" />

## Overview
SpaMCI-DL is a comprehensive computational framework that integrates spatial domains with SVGs detection tasks, as shown in Fig. For spatial domain identification, SpaMCI- DL designs a Multi-Constrained Graph Convolutional Au- toencoder (MCGCA) to learn spots representations (Fig. A). Particularly, this component includes a binary constraint to accentuate gene expression variability, and a graph-structured constraint to model spatial dependencies. For SVGs detection, SpaMCI-DL introduces an interpretable deep learning strategy to build Integrated Gradient Interpreters (IGI) that systemat- ically quantify the contribution of each gene (Fig. B). IGI utilizes a MLP classifier to model the mapping between gene expression and spatial domain, and applies integrated gradients to evaluate gene importance. To further enhance detection resolution, IGI introduces multi-scale constraints that capture gene contributions across different spatial resolutions.

## Example

SpaMCI-DL is a two-stage framework. Taking the HBC dataset as an example, HBC_domains.py needs to be run first, followed by HBC_svgs. All relevant codes have been demonstrated in the project.

## NOTE
All the datasets used in the text are public datasets, and the specific information of the datasets has been disclosed in the main text.


## Citation
The article has been accepted by the 2025 IEEE International Conference on Bioinformatics and Biomedicine (IEEE BIBM 2025). The specific page numbers will be given after the conference.
