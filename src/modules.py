import torch
import torch.nn as nn


class HardThersholding(nn.Module):
    def __init__(self, mean=0, std=0.5):
        super(HardThersholding, self).__init__()

        self.mean = mean
        self.std = std

    def forward(self, X):
        noises = torch.normal(
            mean=self.mean, std=self.std, size=X.size(), requires_grad=False
        )
        return torch.maximum(
            torch.tensor([0]), torch.minimum(torch.tensor([1]), 0.5 + X + noises)
        )


class GatingNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GatingNN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        )

        self.hard_thersholding = HardThersholding()
        self.local_gates = None

    def forward(self, X):
        self.local_gates = self.hard_thersholding(self.network(X))
        return X * self.local_gates


#  self.input_denoising = input_denoising
#  self.latent_denoising = latent_denoising
#  self.random_rate_denoising = random_rate_denoising


class AutoEncoder(nn.Module):
    def __init__(self):
        pass


class ClusteringNN(nn.Module):
    def __init__(self):
        pass


class AuxClassifier(nn.Module):
    def __init__(self):
        pass


class ClusterHead(nn.Module):
    def __init__(self):
        pass
