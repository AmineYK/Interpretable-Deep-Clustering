import math

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

from itertools import combinations

def feature_importance_accuracy_drop(idc, X, Z, feature_importances):
    feature_importances = feature_importances / feature_importances.max()

    h = idc.ae.encoder(X * Z)
    clust_logits, _, _ = idc.clusterNN(X * Z, h)
    yhat = clust_logits.argmax(dim=1)
    
    feature_indices = torch.argsort(feature_importances, descending=True)
    
    performances = []
    
    for idx in feature_indices:
        Z[:, idx] = 0

        h = idc.ae.encoder(X * Z)
        clust_logits, _, _ = idc.clusterNN(X * Z, h)
        ypred = clust_logits.argmax(dim=1)

        performance = clustering_accuracy(yhat.cpu().detach().numpy(), ypred.cpu().detach().numpy())
        performances.append([feature_importances[idx].cpu().detach(), performance])

    return np.array(performances) 

def jaccard_similarity(z1, z2):
    non_zero_indices_z1 = (z1!=0)
    non_zero_indices_z2 = (z2!=0)
    intersection = (non_zero_indices_z1 & non_zero_indices_z2).sum()
    
    union = (non_zero_indices_z1 | non_zero_indices_z2).sum()
    jaccard_sim = intersection / union if union != 0 else 0

    return jaccard_sim


def diversity(global_z):
    K = global_z.size(0)
    sum_jaccard_similarities = 0
    for i, j in combinations(range(K), 2):
        sum_jaccard_similarities += jaccard_similarity(global_z[i], global_z[j])
    return 1 - sum_jaccard_similarities / (K * (K - 1) / 2)



def uniqueness(X, Z, epsilon=2):
    n_samples = X.shape[0]
    uniqueness_scores = []
    
    for i in range(n_samples):
        for k in range(i+1, n_samples):
            distance_x = np.linalg.norm(X[i] - X[k])
            
            if distance_x <= epsilon:
                distance_w = np.linalg.norm(Z[i] - Z[k])
                uniqueness_score = distance_w / distance_x
                uniqueness_scores.append(uniqueness_score)
    
    if len(uniqueness_scores) == 0:
        return 0
    
    return np.min(uniqueness_scores)


def clustering_accuracy(labels_true, labels_pred):
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics.cluster import _supervised

    labels_true, labels_pred = _supervised.check_clusterings(labels_true, labels_pred)
    value = _supervised.contingency_matrix(labels_true, labels_pred)
    [r, c] = linear_sum_assignment(-value)
    return value[r, c].sum() / len(labels_true)


def random_binary_mask(
    size, device, type_mask="INPUT", zero_ratio=0.9, mean=1, std=1e-2
):
    if type_mask == "INPUT":
        samples_rnd = torch.rand(size).to(device)
        masks = torch.ones(size).to(device).float()
        masks[samples_rnd < zero_ratio] = 0
        return masks

    return torch.normal(mean=mean, std=std, size=size, device=device)


def cosine_scheduler(current_epoch, total_epochs, min_val=0, max_val=1):
    return min_val + 0.5 * (max_val - min_val) * (
        1.0 + np.cos(current_epoch * math.pi / total_epochs)
    )

def gumble_softmax(logits, tau):
    logps = F.log_softmax(logits, dim=-1)
    gumble = torch.rand_like(logps).log().mul(-1).log().mul(-1)
    logits = logps + gumble
    return (logits / tau).softmax(dim=-1)


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
