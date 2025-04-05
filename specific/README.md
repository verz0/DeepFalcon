# Specific Task 3: Generative Transformer for Jet Simulation (`specific/`)

This notebook focuses on implementing a Transformer-based generative model for jet events, leveraging the latent space of a pre-trained Variational Autoencoder (VAE).

## What I Did

1.  **Data Loading & Preprocessing:** Loaded the jet image data (`X_jets` only) and applied standard preprocessing (resizing to 128x128, train/validation split of indices, per-channel standardization based on training set statistics) to obtain `X_jets_normalized`. Created image datasets/loaders.
2.  **VAE Loading:** Defined the VAE architecture (matching the one trained in Common Task 1 with `latent_dim=128`) and loaded its pre-trained weights from file. The VAE's encoder and decoder are used for representation learning and final image generation, respectively.
3.  **Latent Space Discretization Setup:**
    * Generated latent mean vectors (`mu`) for a *subset* of the training images using the loaded VAE encoder.
    * Fitted K-Means models (`NUM_BINS=256`) independently to each of the 128 dimensions using the subset latent vectors.
    * Stored the resulting cluster/bin centers for each dimension.
4.  **Training Discrete Code Generation:** Encoded the *entire* training dataset (`X_jets_normalized[train_indices]`) using the VAE encoder and converted the resulting continuous latent vectors into sequences of discrete integer codes (length 128) by assigning each dimension's value to its nearest pre-computed bin center index. This was done batch-wise for memory efficiency.
5.  **Transformer Model:** Implemented a decoder-only autoregressive `GenerativeTransformer` model with standard components (embedding layer for codes, positional encoding, multiple Transformer decoder layers with causal self-attention, final linear layer predicting logits over bins).
6.  **Transformer Training:**
    * Trained the `GenerativeTransformer` on the sequences of discrete codes (`train_codes_tensor`).
    * Used Cross-Entropy loss, aiming to predict the next code index (`token[i+1]`) given the preceding sequence (`token[0...i]`).
    * Employed an AdamW optimizer, gradient clipping, and `ReduceLROnPlateau` scheduler. Saved the best model based on training loss.
7.  **Generative Sampling:**
    * Implemented an autoregressive sampling function (`generate_latent_sequence`) to produce new sequences of latent codes using the trained Transformer (allowing temperature/top-k control).
    * Converted the generated discrete code sequences back into continuous latent vectors (`z`) by mapping codes to their corresponding bin center values.
    * Used the loaded VAE **decoder** to transform the generated continuous latent vectors (`z`) into synthetic 3-channel jet images.
8.  **Evaluation:**
    * Performed **qualitative** evaluation by visually inspecting the generated jet images (combined channels).
    * Performed **quantitative** evaluation by comparing distributions of physical observables (total energy per channel, calculated after **un-standardizing** pixel values) between the generated samples and real samples from the validation set using histograms.

![image](https://github.com/user-attachments/assets/b5e0d4a3-1ee5-4eae-b747-22a0be3893e4)

![image](https://github.com/user-attachments/assets/8dfb9785-a440-4556-bde5-0dd6e2c0d0b6)

