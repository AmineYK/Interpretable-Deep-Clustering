import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils import cosine_scheduler, gumble_softmax


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, y, **kwargs):
        raise NotImplementedError("TODO : ")


class ReconstructionLoss(Loss):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()

    def forward(self, y, yhat):
        return F.l1_loss(y, yhat)


class GTCRLoss(Loss):
    def __init__(self, epsilon=1e-3):
        super(GTCRLoss, self).__init__()

        self.epsilon = epsilon

    def forward(self, z, scale_factor):
        B, D = z.size()
        lmbd = D / (B * self.epsilon)
        I = torch.eye(D, device=z.device)

        logdet = torch.logdet(I + lmbd * z.T.matmul(z))
        return - (logdet / 2) / scale_factor


class RegLoss(Loss):
    def __init__(self, sigma=0.5):
        super(RegLoss, self).__init__()
        self.sigma = sigma

    def forward(self, u):
        return torch.mean(0.5 - 0.5 * torch.erf((-1 / 2 - u) / (self.sigma * math.sqrt(2))))

class SparseLoss(Loss):
    def __init__(self, epsilon=1e-3, pretrain=True):
        super(SparseLoss, self).__init__()

        self.epsilon = epsilon
        self.pretrain = pretrain

    def forward(
        self,
        X,
        X_hat,
        X_input_noised_hat,
        X_latent_noised_hat,
        X_z_hat=None,
        z=None,
        mu=None,
        lmbd=None,
        local_gates_lmbd=100,
    ):
        input_recons = ReconstructionLoss()
        input_denoising_recons = ReconstructionLoss()
        latent_denoising_recons = ReconstructionLoss()
        gate_recons = ReconstructionLoss()

        if self.pretrain:
            gtcr_reg = 0
            X_z_hat = X
        else:
            gtcr = GTCRLoss(self.epsilon)
            reg = RegLoss(sigma=0.5)
            gtcr_reg = gtcr(z, X.size(0)) + lmbd * reg(mu)

        return (
            input_recons(X, X_hat)
            + local_gates_lmbd * gate_recons(X, X_z_hat)
            + local_gates_lmbd * input_denoising_recons(X, X_input_noised_hat)
            + latent_denoising_recons(X, X_latent_noised_hat)
            + gtcr_reg
        )


class HeadLoss(Loss):
    def __init__(self, nb_classes):
        super(HeadLoss, self).__init__()
        self.nb_classes = nb_classes

    def forward(self, h, yhat):
        h = h.T
        D, B = h.size()

        I = torch.eye(D, device=h.device).expand((self.nb_classes, D, D))

        assign = yhat.T.reshape((self.nb_classes, 1, -1))
        intense_clust = assign.sum(2) + 1e-8
        scale = (D / (intense_clust * 1e-1)).view(self.nb_classes, 1, 1)
        h = h.view((1, D, B))

        log_det = torch.logdet(I + scale * h.mul(assign).matmul(h.transpose(1, 2)))
        compress_loss = (intense_clust.squeeze() * log_det / (2 * B)).sum()
        return compress_loss / self.nb_classes

class AuxLoss(Loss):
    
    def __init__(self):
        super(AuxLoss, self).__init__()
        self.reg = RegLoss(sigma=0.5)
        
    def forward(self, yhat, yg, u_zg, lmbd):
        reg_loss = 0
        aux_loss = 0
        
        for y in yhat.unique():
            u_zg_c = u_zg[yhat == y]
            reg_loss = reg_loss + self.reg(u_zg_c)
            
            yg_c = yg[yhat == y]
            aux_loss = aux_loss + F.cross_entropy(yg_c, y.reshape(1).repeat(u_zg_c.size(0)))
        
        aux_loss = aux_loss / len(yhat.unique())
        reg_loss = reg_loss / len(yhat.unique())
        
        return aux_loss + lmbd * reg_loss

class ClusterLoss(Loss):
    def __init__(self, nb_classes, epsilon=1e-3, tau=100, pretrain=True):
        super(ClusterLoss, self).__init__()

        self.nb_classes = nb_classes
        self.pretrain = pretrain
        self.epsilon = epsilon
        self.tau = tau
        
        self.head = HeadLoss(self.nb_classes)
        self.gtcr = GTCRLoss(epsilon)
        self.aux = AuxLoss()
        
    def forward(self, h, clust_logits, yg, u_zg, lmbd, gamma):

        prob = gumble_softmax(clust_logits, self.tau)
        h = F.normalize(h)
        
        if self.pretrain:
            return gamma * self.head(h, prob) + self.gtcr(h, self.nb_classes)
        
        yhat = clust_logits.argmax(dim=-1)

        return gamma * self.head(h, prob) + self.gtcr(h, self.nb_classes), self.aux(yhat, yg, u_zg, lmbd)
