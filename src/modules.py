import torch
import torch.nn as nn
import torch.nn.functional as F


class HardThresholding(nn.Module):
    def __init__(self, mean=0, std=0.5):
        super(HardThresholding, self).__init__()

        self.mean = mean
        self.std = std

    def forward(self, X):
        device = X.device

        # Adding noise only in training mode
        noises = (
            torch.normal(
                mean=self.mean, std=self.std, size=X.size(), requires_grad=False
            ).to(device)
            * 0.5  # Decrease the noises
            * self.training
        )

        return torch.maximum(
            torch.tensor([0]).to(device),
            torch.minimum(torch.tensor([1]).to(device), 0.5 + X + noises),
        )


class GatingNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GatingNN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh(),
        )

        self.hard_thresholding = HardThresholding(mean=0, std=0.5)

    def forward(self, X):
        logits = self.network(X)
        local_gates = self.hard_thresholding(logits)
        return X * local_gates, local_gates, logits


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

    def forward(self, X):
        raise NotImplementedError("TODO : ")


class MLPAutoEncoder(AutoEncoder):
    def __init__(self, layer_dims):
        super(MLPAutoEncoder, self).__init__()

        encoder_layers = []
        decoder_layers = []

        previous_layer_dim = layer_dims[0]
        for dim in layer_dims[1:-1]:
            encoder_layers.append(nn.Linear(previous_layer_dim, dim))
            encoder_layers.append(nn.BatchNorm1d(dim))
            encoder_layers.append(nn.ReLU())
            previous_layer_dim = dim

        encoder_layers.append(nn.Linear(previous_layer_dim, layer_dims[-1]))
        self.encoder = nn.Sequential(*encoder_layers)

        reversed_layer_dims = list(reversed(layer_dims))
        previous_layer_dim = reversed_layer_dims[0]
        for dim in reversed_layer_dims[1:-1]:
            decoder_layers.append(nn.Linear(previous_layer_dim, dim))
            decoder_layers.append(nn.BatchNorm1d(dim))
            decoder_layers.append(nn.ReLU())
            previous_layer_dim = dim

        decoder_layers.append(nn.Linear(previous_layer_dim, reversed_layer_dims[-1]))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, X):
        return self.decoder(self.encoder(X))


class ClusteringNN(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, nb_classes, tau=1e-1):
        super(ClusteringNN, self).__init__()

        self.cluster_head = ClusterHead(latent_dim, hidden_dim, nb_classes, tau)
        self.aux_classifier = AuxClassifier(input_dim, hidden_dim, nb_classes)
        self.global_gates = nn.Embedding(nb_classes, input_dim)

        self.hard_thresholding = HardThresholding(mean=0, std=0.5)

    def forward(self, X, h):
        # X --> X*Z
        clust_logits = self.cluster_head(h)
        yhat = clust_logits.argmax(dim=1)

        aux_input = X * self.hard_thresholding(self.global_gates(yhat))
        aux_logits = self.aux_classifier(aux_input)

        # return the logits of the clustering, the logits of the auxiliary classifier and the global gates (mu not thresholded) for the batch
        return clust_logits, aux_logits, self.global_gates(yhat)


class AuxClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, nb_classes):
        super(AuxClassifier, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nb_classes),
        )

    def forward(self, X):
        return self.network(X)


class ClusterHead(nn.Module):
    def __init__(self, latent_dim, hidden_dim, nb_classes, tau=1e-1):
        super(ClusterHead, self).__init__()

        self.tau = tau
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nb_classes),
        )

    def forward(self, X):
        logits = self.network(X)
        logits = F.log_softmax(logits, dim=1)
        return F.gumbel_softmax(logits, tau=self.tau)
