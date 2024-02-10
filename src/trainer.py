import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn

from losses import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def idc_trainer(idc, X, ae_gnn_config_train, clust_config_train):

    idc = idc.to(device)

    # AE + GNN
    stage_one_dataset = TensorDataset(X)
    stage_one_dataloader = DataLoader(
        stage_one_dataset, batch_size=ae_gnn_config_train["batch_size"], shuffle=True
    )

    stage_one_criterion = SparseLoss(epsilon=ae_gnn_config_train["eps"], pretrain=True)
    stage_one_optimizer = optim.Adam(
        list(idc.gnn.parameters()) + list(idc.ae.parameters()),
        lr=ae_gnn_config_train["lr"],
    )

    ae_sparse_losses = []
    ae_gnn_sparse_losses = []

    print("--------------------------------")
    print("Stage One Starting :")

    for epoch in tqdm(range(ae_gnn_config_train["epochs"])):
        epoch_sparse_loss = 0.0

        lmbd = None
        if epoch > ae_gnn_config_train["end_pretrain_epoch"]:
            stage_one_criterion.pretrain = False

            lmbd = cosine_scheduler(
                epoch - ae_gnn_config_train["end_pretrain_epoch"],
                ae_gnn_config_train["epochs"]
                - ae_gnn_config_train["end_pretrain_epoch"],
                max_val=ae_gnn_config_train["reg_lmbd"],
            )

        for (x,) in stage_one_dataloader:
            x = x.to(device)

            x_hat = idc.ae(x)

            x_z_hat, z, u = None, None, None
            if epoch > ae_gnn_config_train["end_pretrain_epoch"]:
                x_z, z, u = idc.gnn(x)
                x_z_hat = idc.ae(x_z)

            input_noise_mask = random_binary_mask(x.size(), x.device, type_mask="INPUT")
            x_input_noised_hat = idc.ae(x * input_noise_mask)

            h = idc.ae.encoder(x)
            h_noise = random_binary_mask(h.size(), h.device, type_mask="LATENT")
            h = h * h_noise
            x_latent_noised_hat = idc.ae.decoder(h)

            sparse_loss = stage_one_criterion(
                x,
                x_hat,
                x_input_noised_hat,
                x_latent_noised_hat,
                x_z_hat,
                z,
                u,
                lmbd,
                local_gates_lmbd=ae_gnn_config_train["local_gates_lmbd"],
            )

            stage_one_optimizer.zero_grad()
            sparse_loss.backward()
            stage_one_optimizer.step()

            epoch_sparse_loss += sparse_loss.item()

        avg_epoch_sparse_loss = epoch_sparse_loss / len(stage_one_dataloader)

        if epoch > ae_gnn_config_train["end_pretrain_epoch"]:
            ae_gnn_sparse_losses.append(avg_epoch_sparse_loss)
        else:
            ae_sparse_losses.append(avg_epoch_sparse_loss)

    print("Stage One Finishing")
    print("--------------------------------")

    # Cluster Head + Aux Classifier
    X_Z, _, _ = idc.gnn(X.to(device))

    # encode from X_Z
    H_from_X_Z = idc.ae.encoder(X_Z)
    stage_two_dataset = TensorDataset(X_Z, H_from_X_Z)

    stage_two_dataloader = DataLoader(
        stage_two_dataset, batch_size=clust_config_train["batch_size"], shuffle=True
    )

    stage_two_criterion = ClusterLoss(
        idc.nb_classes, pretrain=True, epsilon=ae_gnn_config_train["eps"]
    )

    cluster_head_optimizer = optim.Adam(
        idc.clusterNN.cluster_head.parameters(),
        lr=clust_config_train["lr_cluster_head"],
    )

    aux_optimizer = optim.SGD(
        idc.clusterNN.aux_classifier.parameters(), lr=clust_config_train["lr_aux"]
    )

    zg_optimizer = optim.SGD(
        idc.clusterNN.global_gates.parameters(), lr=clust_config_train["lr_zg"]
    )

    clust_head_pretrain_losses = []
    clust_head_finetune_losses = []
    aux_losses = []

    print("\n--------------------------------")
    print("Stage Two Starting :")
    for epoch in tqdm(range(clust_config_train["epochs"])):

        if epoch > clust_config_train["end_pretrain_epoch"]:
            stage_two_criterion.pretrain = False

            lmbd = cosine_scheduler(
                epoch - clust_config_train["end_pretrain_epoch"],
                clust_config_train["epochs"] - clust_config_train["end_pretrain_epoch"],
                max_val=clust_config_train["global_gates_lmbd"],
            )

        else:
            lmbd = cosine_scheduler(
                epoch,
                clust_config_train["end_pretrain_epoch"],
                max_val=clust_config_train["global_gates_lmbd"],
            )

        epoch_loss_head = 0.0
        epoch_loss_aux = 0.0
        for x, h in stage_two_dataloader:

            x = x.to(device)
            h = h.to(device)

            clust_logits, aux_logits, u_zg = idc.clusterNN(x, h)

            if epoch > clust_config_train["end_pretrain_epoch"]:
                loss_head, loss_aux = stage_two_criterion(
                    h,
                    clust_logits,
                    aux_logits,
                    u_zg,
                    lmbd,
                    gamma=clust_config_train["gamma"],
                )
            else:
                loss_head = stage_two_criterion(
                    h,
                    clust_logits,
                    aux_logits,
                    u_zg,
                    lmbd,
                    gamma=clust_config_train["gamma"],
                )

            cluster_head_optimizer.zero_grad()
            loss_head.backward(retain_graph=True)
            cluster_head_optimizer.step()

            epoch_loss_head += loss_head.item()

            if epoch > clust_config_train["end_pretrain_epoch"]:
                aux_optimizer.zero_grad()
                zg_optimizer.zero_grad()
                loss_aux.backward(retain_graph=True)
                aux_optimizer.step()
                zg_optimizer.step()

                epoch_loss_aux += loss_aux.item()

        if epoch > clust_config_train["end_pretrain_epoch"]:
            avg_epoch_loss_head = epoch_loss_head / len(stage_two_dataloader)
            clust_head_finetune_losses.append(avg_epoch_loss_head)

            avg_epoch_loss_aux = epoch_loss_aux / len(stage_two_dataloader)
            aux_losses.append(avg_epoch_loss_aux)
        else:
            avg_epoch_loss_head = epoch_loss_head / len(stage_two_dataloader)
            clust_head_pretrain_losses.append(avg_epoch_loss_head)

    print("Stage Two Finishing")
    print("--------------------------------")

    return {
        "stage_one": {
            "ae_sparse_losses": ae_sparse_losses,
            "ae_gnn_sparse_losses": ae_gnn_sparse_losses,
        },
        "stage_two": {
            "clust_head_pretrain_losses": clust_head_pretrain_losses,
            "clust_head_finetune_losses": clust_head_finetune_losses,
            "aux_losses": aux_losses,
        },
    }
