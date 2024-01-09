import torch
import numpy as np

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def random_binary_mask(N, D, zero_ratio=0.9):
    masks = torch.ones(N, D)

    n_zeros = int(D * zero_ratio)

    perm = np.arange(D)
    idx = np.array([perm] * N)
    np.apply_along_axis(np.random.shuffle, 1, idx)

    indices = torch.tensor(idx[:, :n_zeros])
    masks.scatter_(1, indices, 0)

    return masks


def cosine_scheduler(x, max_x, lmbd_init=1):
    return lmbd_init * 0.5 * (1 + np.cos(np.pi * x / max_x))


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
