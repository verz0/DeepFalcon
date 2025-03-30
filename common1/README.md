# Common Task 1. Auto-encoder of the quark/gluon events

This notebook details my implementation of a Variational Autoencoder (VAE) for learning representations and reconstructing 3-channel (ECAL, HCAL, Tracks) jet images from the quark/gluon dataset.

## What I Did

1.  **Data Loading & Preprocessing:** I loaded the jet data from the HDF5 file, taking a subset for efficiency. I converted the images to PyTorch tensors, resized them to a standard dimension (e.g., 128x128), and normalized them per-channel using mean and standard deviation calculated from the training set.
2.  **VAE Model:** I defined a VAE architecture consisting of:
    * An **Encoder** using convolutional layers (Conv2D, BatchNorm, ReLU) to map input images to the parameters (mean mu and log-variance logvar) of a latent Gaussian distribution.
    * A **Decoder** using transposed convolutional layers (ConvTranspose2d, BatchNorm, ReLU) to reconstruct the image from a sample drawn from the latent distribution using the reparameterization trick.
3.  **Loss Function:** I used the standard VAE loss, combining:
    * Mean Squared Error (MSE) for the reconstruction loss between the original and decoded images.
    * Kullback-Leibler (KL) Divergence to regularize the latent space, encouraging it to approximate a standard normal distribution. I included a `beta` parameter to potentially weight the KLD term.
4.  **Training:** I implemented a standard PyTorch training loop with:
    * An Adam optimizer.
    * A ReduceLROnPlateau learning rate scheduler based on validation loss.
5.  **Visualization:**
    * I generated side-by-side plots comparing original input jet images and their reconstructions from the trained VAE.
    * I visualized the learned latent space (using the `mu` values) by applying Principal Component Analysis (PCA) to reduce it to 2 dimensions and creating a scatter plot, coloring the points by their true quark/gluon label. I adjusted plot limits for better visibility.

![image](https://github.com/user-attachments/assets/6e8938d8-a970-40e1-94b3-fbc0d217d02c)

![image](https://github.com/user-attachments/assets/ee1c87c2-4d2c-43bf-ad88-151dba1fa76e)
