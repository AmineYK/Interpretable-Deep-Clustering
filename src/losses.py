import torch
import torch.nn as nn
import numpy as np

from utils import cosine_scheduler
from modules import device


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, y, **kwargs):
        raise NotImplementedError("TODO : ")


class ReconstructionLoss(Loss):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()

    def forward(self, y, yhat):
        return torch.norm(yhat - y, p=1)


class GTCRLoss(Loss):
    def __init__(self):
        super(GTCRLoss, self).__init__()

    def forward(self, z, epsilon=1e-3):
        B = z.size(0)

        # Image's case
        if z.ndim > 2:
            z = z.view(B, -1)

        B, D = z.size()

        lmbd = D * (B * epsilon)

        z_1 = z.view(B, D, 1)
        z_2 = z_1.permute(0, 2, 1)

        return -torch.mean(
            torch.logdet(((torch.bmm(z_1, z_2) * lmbd) + torch.eye(D).to(device))) * 0.5
        )


class RegLoss(Loss):
    def __init__(self, sigma=0.5):
        super(RegLoss, self).__init__()

        self.sigma = sigma

    def forward(self, u):
        return (
            0.5 - 0.5 * torch.erf(-(u + 0.5) / (np.sqrt(2) * self.sigma))
        ).sum() / u.size(0)


class SparseLoss(Loss):
    def __init__(self, epsilon=1e-3):
        super(SparseLoss, self).__init__()

        self.epsilon = epsilon

    def forward(self, X, X_hat, z, u, lmbd):
        recons = ReconstructionLoss()
        gtcr = GTCRLoss()
        reg = RegLoss(sigma=0.5)

        # a = recons(X, X_hat)
        # b = gtcr(z, self.epsilon)
        # c = reg(u)
        # d = c

        # print("GTCR :", b.item(), end=" ")
        # print("Recons :", a.item(), end=" ")
        # print("Reg :", d.item(), "(lambda = ", lmbd, "reg_u = ", c.item(), ")")
        # print("\n==========================================================\n")
        # return a + b + d

        return recons(X, X_hat) + gtcr(z, self.epsilon) + lmbd * reg(u)


class HeadLoss(Loss):
    def __init__(self, nb_classes):
        super(HeadLoss, self).__init__()
        self.nb_classes = nb_classes

    def forward(self, h, yhat):
        loss = 0
        for k in range(self.nb_classes):
            h_k = h[torch.where(yhat == k)]
            B, D = h_k.size()

            h_1 = h_k.view(B, D, 1)
            h_2 = h_1.permute(0, 2, 1)

            loss += (
                torch.logdet(
                    (torch.mean(torch.bmm(h_1, h_2), dim=0) * 0.5 + torch.eye(D))
                )
                * 0.5
            ).item()

        return loss


class ClusterLoss(Loss):
    def __init__(self):
        super(ClusterLoss, self).__init__()

    def forward(self, h, yhat, yg, zg, lmbd):
        head = HeadLoss()
        ce = nn.functional.cross_entropy()
        reg = RegLoss(sigma=0.5)

        yhat_argmax = yhat.argmax(dim=1)

        return head(h, yhat_argmax) + ce(yhat, yg) + lmbd * reg(zg)
