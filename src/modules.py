import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights_normal(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, std=0.001)

        if "bias" in vars(m).keys():
            m.bias.data.fill_(0.0)


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

        self.network.apply(init_weights_normal)

    def forward(self, X):
        logits = self.network(X)
        local_gates = self.hard_thresholding(logits)
        return X * local_gates, local_gates, logits


class MLPAutoEncoder(nn.Module):
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

        self.encoder.apply(init_weights_normal)
        self.decoder.apply(init_weights_normal)

    def forward(self, X):
        return self.decoder(self.encoder(X))


class ClusteringNN(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, nb_classes, tau=1e-1):
        super(ClusteringNN, self).__init__()

        self.cluster_head = ClusterHead(latent_dim, hidden_dim, nb_classes, tau)
        self.aux_classifier = AuxClassifier(input_dim, hidden_dim, nb_classes)

        embedding = nn.Embedding(nb_classes, input_dim)
        torch.nn.init.normal_(embedding.weight, std=0.01)

        self.global_gates = nn.Sequential(embedding, nn.Tanh())

        self.hard_thresholding = HardThresholding(mean=0, std=0.5)

    def forward(self, X, h):
        # X --> X*Z
        clust_logits = self.cluster_head(h)
        yhat = clust_logits.argmax(dim=1)

        u_zg = self.global_gates(yhat)

        aux_input = X * self.hard_thresholding(u_zg)
        aux_logits = self.aux_classifier(aux_input)

        # return the logits of the clustering, the logits of the auxiliary classifier and the global gates (mu not thresholded) for the batch
        return clust_logits, aux_logits, u_zg


class AuxClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, nb_classes):
        super(AuxClassifier, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nb_classes),
        )

        self.network.apply(init_weights_normal)

    def forward(self, X):
        return self.network(X)


def gumble_softmax(logits, tau):
    logps = F.log_softmax(logits, dim=-1)
    gumble = torch.rand_like(logps).log().mul(-1).log().mul(-1)
    logits = logps + gumble
    return (logits / tau).softmax(dim=-1)


class ClusterHead(nn.Module):
    def __init__(self, latent_dim, hidden_dim, nb_classes, tau=100):
        super(ClusterHead, self).__init__()

        self.tau = tau
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nb_classes),
        )

        self.network.apply(init_weights_normal)

    def forward(self, X):
        logits = self.network(X)
        return F.softmax(logits, dim=-1)


class IDC(nn.Module):
    def __init__(
        self,
        data_input_dim,
        ae_layer_dims,
        gnn_hidden_dim,
        cluster_hidden_dim,
        nb_classes,
    ):
        super(IDC, self).__init__()

        self.nb_classes = nb_classes
        self.ae = MLPAutoEncoder(ae_layer_dims)
        self.gnn = GatingNN(data_input_dim, gnn_hidden_dim)

        # Dimension of latent space H
        ae_latent_dim = ae_layer_dims[-1]
        self.clusterNN = ClusteringNN(
            data_input_dim, ae_latent_dim, cluster_hidden_dim, nb_classes
        )
