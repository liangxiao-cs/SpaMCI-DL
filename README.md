# SpaMCI-DL: A hybrid deep learning framework for integrated identification of domains and spatially variable genes in spatial transcriptomics

<img width="2144" height="1150" alt="image" src="https://github.com/user-attachments/assets/be35c3db-7160-42c1-8f2c-e47fc915372e" />

## Overview
SpaMCI-DL is a comprehensive computational framework that integrates spatial domains with SVGs detection tasks, as shown in Fig. For spatial domain identification, SpaMCI- DL designs a Multi-Constrained Graph Convolutional Au- toencoder (MCGCA) to learn spots representations (Fig. A). Particularly, this component includes a binary constraint to accentuate gene expression variability, and a graph-structured constraint to model spatial dependencies. For SVGs detection, SpaMCI-DL introduces an interpretable deep learning strategy to build Integrated Gradient Interpreters (IGI) that systemat- ically quantify the contribution of each gene (Fig. B). IGI utilizes a MLP classifier to model the mapping between gene expression and spatial domain, and applies integrated gradients to evaluate gene importance. To further enhance detection resolution, IGI introduces multi-scale constraints that capture gene contributions across different spatial resolutions.


## Example

For training spaMMCL model, run

'python spaMMCL.py'

## NOTE
It is recommended to give priority to using the version with images (MML.py). If there are no images, use the version without images (MML_without_img.py).


## Citation
Liang et al. A multi-modality and multi-granularity collaborative learning framework for identifying spatial domains and spatially variable genes. Bioinformatics, 2024, 40(10): btae607
