# DeepFalcon-Evaluation-Tasks

## Graph Transformers for Fast Detector Simulation

This repository contains the notebooks and models for the tasks I have completed for the project "Graph Transformers for Fast Detector Simulation" for GSoC 2025

## Tasks I Completed

I've structured the repository based on the tasks I completed:

1.  **Common Task 1. Auto-encoder of the quark/gluon events (`common1/`)**
    * Implemented and trained a VAE using CNNs to learn compressed representations of the 3-channel jet images.
    * My focus was on reconstructing the input images from the learned latent space.
    * Generated visualizations showing side-by-side comparisons of original and reconstructed jet events.
    * Analyzed the VAE's latent space using PCA for dimensionality reduction and visualization, coloring points by their quark/gluon labels.

2.  **Common Task 2. Jets as graphs (`common2/`)**
    * I developed a pipeline to convert jet images into graph representations:
        * I first converted images to point clouds by selecting pixels above an intensity threshold.
        * For node features, I used the original 3 channel intensities plus engineered polar coordinates (radius, angle).
        * I constructed graphs using K-Nearest Neighbors (KNN) based on spatial coordinates.
        * Used the normalized distance between nodes as an edge feature.
    * Implemented and trained a Graph Attention Network (GATv2) for classifying jets as quarks or gluons.
    * Evaluated the GNN's performance using accuracy, confusion matrix, and classification reports.
    * Visualized the learned graph embeddings using UMAP to see if the classes were separable.

3.  **Specific Task 3 (if you are interested in “Graph Transformers for Fast Detector Simulation” project): (`specific/`)**
    * Implemented a generative model for jet events using an autoregressive Transformer operating on the latent space of a pre-trained Variational Autoencoder (VAE).
    * Loaded a VAE (trained in Common Task 1) with a latent dimension of 128. Used its encoder to obtain latent mean vectors for the training dataset images.
    * Discretized the 128-dimensional continuous latent space by applying K-Means clustering (`NUM_BINS=256`) independently to each dimension, using a subset of the training latent vectors for efficient fitting. This process defined bin centers for mapping continuous values to discrete integer codes.
    * Implemented an autoregressive sampling function (`generate_latent_sequence`) using the trained Transformer (with temperature/top-k options) to generate new sequences of latent codes.
    * Converted the generated discrete code sequences back into continuous latent vectors by mapping the codes to their corresponding pre-calculated bin center values.
    * Evaluated the generative performance through:
        * Qualitative visual inspection:Displaying samples of the generated jet images.
        * Quantitative comparison: Generating histograms comparing the distribution of physically meaningful observables (specifically, total per-channel energy after un-standardizing pixel values) between the generated samples and real validation samples. This comparison highlighted areas where the generated distributions matched or diverged from the real data.

## What I Learned

* How to apply VAEs for unsupervised representation learning and image reconstruction in a physics data context.
* Various techniques for converting structured image data into graph representations (point clouds, KNN graphs) and how to engineer relevant node/edge features (like polar coordinates or distance).
* How to implement and train different GNN architectures (GATv2 for classification, GCN for autoencoding) using PyTorch Geometric.
* The conceptual differences and potential trade-offs between using standard CNN-based autoencoders versus graph-based autoencoders for this type of jet data.
* Strategies for using sequence-based models like Transformers for generative tasks on non-sequential data, specifically by discretizing the continuous latent space of a pre-trained VAE (using K-Means per dimension) and training an autoregressive Transformer on the resulting sequences of discrete codes.

## Challenges I Faced & Considerations

* **Computational Resources:** Training these deep learning models was computationally demanding. Using GPUs was essential for feasible training times, and even then, I worked with subsets (15,000 samples) of the full dataset to manage runtimes.
* **Hyperparameter Tuning:** Finding the best settings (like learning rates, network sizes, `k` for KNN, VAE's beta value, dropout rates) wasn't straightforward and usually requires more extensive experimentation than I performed in these initial explorations.
* **Graph Construction Details:** Turning images into graphs involves making several choices (intensity threshold, how to define nodes/edges/features) that significantly influence the results. I also had to filter out graphs that didn't have enough nodes after thresholding.
* **Visualization:** Visualizing the high-dimensional outputs (latent spaces, embeddings) needed dimensionality reduction (PCA, UMAP). I also had to adjust plotting parameters like axis limits to get meaningful views, especially for the latent space plots.

PS: The respective directories and the notebooks for each task has detailed explanations included for your reference and evaluation.
    The models were developed in a Kaggle environment (16GB RAM + T4 GPU)
