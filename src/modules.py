import torch
import torch.nn as nn
from utils import random_binary_mask
import torch.nn.Funtionnal as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HardThersholding(nn.Module):
    def __init__(self, mean=0, std=0.5):
        super(HardThersholding, self).__init__()

        self.mean = mean
        self.std = std

    def forward(self, X):
        noises = torch.normal(
            mean=self.mean, std=self.std, size=X.size(), requires_grad=False
        ).to(device)

        return torch.maximum(
            torch.tensor([0]).to(device),
            torch.minimum(torch.tensor([1]).to(device), 0.5 + X + noises),
        )


class GatingNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GatingNN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        )

        self.hard_thersholding = HardThersholding(mean=0, std=0.5)

    def forward(self, X):
        logits = self.network(X)
        local_gates = self.hard_thersholding(logits)
        return X * local_gates, local_gates, logits

    def get_local_gates(self, X):
        """
        Get the final local gates (interpretability) of the network for a X

        :param X :
        """
        return self.hard_thersholding(self.network(X))


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

    def forward(self, X):
        raise NotImplementedError("TODO : ")


class MLPAutoEncoder(AutoEncoder):
    def __init__(
        self,
        input_dim,
        input_denoising=True,
        random_rate_denoising=0.9,
        latent_denoising=True,
        sigma_denoising=1,
    ):
        super(MLPAutoEncoder, self).__init__()

        self.input_denoising = input_denoising
        self.latent_denoising = latent_denoising
        self.random_rate_denoising = random_rate_denoising
        self.sigma_denoising = sigma_denoising

        self.input_dim = input_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            # TODO : A tester
            # nn.BatchNorm1d(input_dim),
            # nn.ReLU(),
            # nn.Sigmoid(),
        )

    def forward(self, X):
        N, D = X.size()

        if self.input_denoising:
            X = X * random_binary_mask(N, D, self.random_rate_denoising).to(device)

        hidden_emb = self.encoder(X)

        if self.latent_denoising:
            hidden_emb = (
                hidden_emb
                * torch.normal(mean=1, std=self.sigma_denoising, size=(1,)).item()
            )

        return self.decoder(hidden_emb), hidden_emb


class ClusteringNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, nb_classes, tau=1e-1):
        super(ClusteringNN, self).__init__()

        self.cluster_head = ClusterHead(nb_classes, hidden_dim, tau)
        self.aux_classifier = AuxClassifier(input_dim, hidden_dim, nb_classes)
        self.global_gates = torch.randint(0, 2, (nb_classes, input_dim))

    def forward(self, X, h):
        # X --> X*Z
        clust_logits = self.cluster_head(h)
        yhat = clust_logits.argmax(dim=1)

        aux_input = X * self.global_gates[yhat]
        aux_logits = self.aux_classifier(aux_input)

        return clust_logits, aux_logits, self.global_gates


class AuxClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, nb_classes):
        super(AuxClassifier, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Linear(hidden_dim, nb_classes)
        )

    def forward(self, X):
        return self.network(X)


class ClusterHead(nn.Module):
    def __init__(self, nb_classes, hidden_dim, tau=1e-1):
        super(ClusterHead, self).__init__()

        self.tau = tau
        self.network = nn.Sequential(
            nn.Linear(2048, hidden_dim), nn.Linear(hidden_dim, nb_classes)
        )

    def forward(self, X):
        logits = self.network(X)

        return F.gumbel_softmax(logits, tau=self.tau)
