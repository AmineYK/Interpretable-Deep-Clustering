import math

import torch
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


def random_binary_mask(
    size, device, type_mask="INPUT", zero_ratio=0.9, mean=0, std=1e-2
):
    if type_mask == "INPUT":
        samples_rnd = torch.rand(size).to(device)
        masks = torch.ones(size).to(device).float()
        masks[samples_rnd < 0.9] = 0
        return masks

    return torch.normal(mean=mean, std=std, size=size, device=device)


def cosine_scheduler(current_epoch, total_epochs, min_val=0, max_val=10):
    return min_val + 0.5 * (max_val - min_val) * (
        1.0 + np.cos(current_epoch * math.pi / total_epochs)
    )


def get_synthetic_dataset():
    num_samples_per_cluster = 800
    num_clusters = 4
    num_informative_features = 3
    num_nuisance_features = 10
    cluster_std = 0.5
    nuisance_std = 0.1

    # Generate isotropic Gaussian blobs with custom cluster centers
    centers = [[0, 1, 1], [0, 1, 5], [4, 0, 4], [4, 5, 4]]
    X, y = make_blobs(
        n_samples=num_samples_per_cluster * num_clusters,
        n_features=num_informative_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=42,
    )

    # Add nuisance background features
    nuisance_features = np.random.normal(
        0, nuisance_std, size=(X.shape[0], num_nuisance_features)
    )
    X = np.hstack([X, nuisance_features])

    return torch.tensor(X).float(), torch.tensor(y)


def plot_synthetic_dataset(X, y):
    # Visualize the clusters in the first two dimensions {x[1], x[2]}
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolors="k", marker="o")
    plt.title("Clusters in the space X1 and X2")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

    # Visualize the clusters in the dimensions pair {x[1], x[3]}
    plt.scatter(X[:, 0], X[:, 2], c=y, cmap="viridis", edgecolors="k", marker="o")
    plt.title("Clusters in the space X1 and X3")
    plt.xlabel("X1")
    plt.ylabel("X3")
    plt.show()
