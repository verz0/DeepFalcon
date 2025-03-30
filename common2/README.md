# Common Task 2. Jets as graphs 

This notebook covers my approach to classifying quark vs. gluon jets using a graph-based Graph Neural Network (GNN).

## What I Did

1.  **Data Loading & Preprocessing:** I loaded the HDF5 jet data, resized images (e.g., to 64x64 for efficiency and to avoid memory issues), and performed per-channel min-max normalization. I split the data into training and validation sets.
2.  **Image-to-Graph Conversion:** I developed a process to represent each jet image as a graph:
    * **Point Cloud Generation:** I selected pixels with intensity above a certain threshold across the 3 channels to form the nodes of the graph.
    * **Node Features:** For each node (pixel), I used its 3 channel intensities plus two engineered features: normalized radial distance from the image center and normalized polar angle. This resulted in 5 node features per node.
    * **Graph Construction:** I used K-Nearest Neighbors (KNN) based on the spatial (y, x) coordinates of the nodes to define the graph edges.
    * **Edge Features:** I used the normalized Euclidean distance between connected nodes as a 1-dimensional edge feature.
    * **PyG Dataset:** I created a custom PyTorch Geometric `Dataset` class to handle this conversion process and serve `Data` objects containing node features (`x`), edge indices (`edge_index`), edge features (`edge_attr`), and graph labels (`y`). I filtered out graphs with fewer than 2 nodes.
3.  **GNN Model:** I implemented a `JetGNN` model using `GATv2Conv` layers from PyTorch Geometric.
    * The model consists of multiple GATv2 layers incorporating edge features, followed by `GraphNorm` and ELU activation.
    * I used global mean and max pooling (`global_mean_pool`, `global_max_pool`) to aggregate node embeddings into a fixed-size graph embedding.
    * A Multi-Layer Perceptron (MLP) head was added to classify the graph embedding into Quark (0) or Gluon (1).
4.  **Training:** I implemented a `Trainer` class encapsulating the training loop:
    * Used AdamW optimizer and `CrossEntropyLoss`.
    * Included gradient clipping for stability.
    * Used `ReduceLROnPlateau` scheduler based on validation accuracy.
    * Implemented early stopping based on validation accuracy to prevent overfitting.
      
5.  **Evaluation & Visualization:**
    * Evaluated the best model on the validation set, reporting final loss and accuracy.
    * Plotted the training/validation loss and validation accuracy curves over epochs.
    * I generated and displayed a confusion matrix and a classification report (precision, recall, F1-score).
    * Extracted the graph embeddings from the GNN for the validation set and visualized them in 2D using UMAP, coloring points by their true label.


![image](https://github.com/user-attachments/assets/736c042f-d6f4-4b6b-82a5-ec1e43330886)

![image](https://github.com/user-attachments/assets/2e97fe1d-06ff-440f-af3c-df9350bd9bc5)

![image](https://github.com/user-attachments/assets/00305878-f62a-45d1-b2b8-a7b3ec917af1)


