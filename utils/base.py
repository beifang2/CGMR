#!/usr/bin/env python
# -*- coding: utf-8 -*-
from copy import deepcopy

import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score

types_pretty = {'train': 'training', 'valid': 'validation', 'valid': 'valid'}

def metrics(predictions, gts, label_values=["Building", "Non-building"]):

    # 自动获取标签编号
    labels = list(range(len(label_values)))

    # 防止报错：若 gts 中类别不足，默认填充 cm
    try:
        cm = confusion_matrix(gts, predictions, labels=labels)
    except ValueError:
        cm = np.zeros((len(labels), len(labels)), dtype=int)

    # OA
    oa = accuracy_score(gts, predictions)

    # Kappa
    kappa = cohen_kappa_score(gts, predictions)

    return oa, kappa



def metrics_iou(predictions, gts, label_values=["Building", "Non-building"]):
    # 自动获取标签编号
    labels = list(range(len(label_values)))

    # 防止报错：若 gts 中类别不足，默认填充 cm
    try:
        cm = confusion_matrix(gts, predictions, labels=labels)
    except ValueError:
        cm = np.zeros((len(labels), len(labels)), dtype=int)

    # OA
    oa = accuracy_score(gts, predictions)

    # mIoU
    ious = np.diag(cm) / (cm.sum(1) + cm.sum(0) - np.diag(cm) + 1e-10)
    miou = np.nanmean(ious)

    return oa, miou

def weighted_average_updates(w, n_k, weights, client_modalities):
    total_clients = len(w)
    w_avg = deepcopy(w[0])

    def get_modality(key):
        if key.startswith("CNN_Encoder1"):
            return "hsi"
        elif key.startswith("CNN_Encoder2"):
            return "lidar"
        elif key.startswith("CNN_Encoder3"):
            return "rgb"
        else:
            return "shared"  # 公共部分或分类头等通用层

    for key in w_avg.keys():
        modality = get_modality(key)

        # 初始化 w_avg[key] 为 float 类型，避免类型冲突
        w_avg[key] = torch.zeros_like(w_avg[key], dtype=torch.float32)

        total_weight = 0.0

        for i in range(total_clients):
            # 如果该客户端不包含该模态，跳过
            if modality != "shared" and modality not in client_modalities[i]:
                continue

            weight = weights[i] * len(w) * n_k[i]

            # 累加加权参数
            w_avg[key] += w[i][key].float() * weight
            total_weight += weight

        if total_weight > 0:
            w_avg[key] /= total_weight
        # 否则保留为 0 或保持原始参数（可以根据需要自定义行为）

    return w_avg



class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)

            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at
        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def inference_fedavg(model, loader, device):
    if loader is None:
        return None, None

    # criterion = BCELoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    # criterion = FocalLoss().cuda()
    loss, total, correct = 0., 0, 0
    model.eval()

    with torch.no_grad():
        all_preds = []
        all_gts = []
        for batch, (examples, targets) in enumerate(loader):
            examples, targets = examples.to(device), targets.to(device)
            outputs, con_loss = model(examples)
            loss = (1 - 0.8) * criterion(outputs, targets) + 0.8 * con_loss

            out_seg = outputs
            _, pred = torch.max(out_seg, dim=1, keepdim=True)
            pred = pred.detach().cpu().numpy().astype(np.float32)
            targets = targets.detach().cpu().numpy().astype(np.float32)

            all_preds.append(pred)
            all_gts.append(targets)
            total += targets.shape[0]

    kappa, oa = metrics(np.concatenate([p.ravel() for p in all_preds]),
                           np.concatenate([p.ravel() for p in all_gts]).ravel())
    print(kappa, oa)
    loss /= total

    return kappa, loss

def inference_dbe(model, loader, device):
    if loader is None:
        return None, None

    criterion = nn.CrossEntropyLoss().to(device)
    loss, total, correct = 0., 0, 0
    model.eval()

    with torch.no_grad():
        all_preds = []
        all_gts = []
        for batch, (examples, targets) in enumerate(loader):
            examples, targets = examples.to(device), targets.to(device)
            outputs, con_loss, _ = model(examples)
            loss = (1 - 0.8) * criterion(outputs, targets) + 0.8 * con_loss

            out_seg = outputs
            _, pred = torch.max(out_seg, dim=1, keepdim=True)
            pred = pred.detach().cpu().numpy().astype(np.float32)
            targets = targets.detach().cpu().numpy().astype(np.float32)

            all_preds.append(pred)
            all_gts.append(targets)
            total += targets.shape[0]

    kappa, oa = metrics(np.concatenate([p.ravel() for p in all_preds]),
                           np.concatenate([p.ravel() for p in all_gts]).ravel())
    print(kappa, oa)
    # accuracy = correct/total
    loss /= total

    return kappa, loss

def inference_fedmuti(model, loader, device):
    if loader is None:
        return None, None

    # criterion = BCELoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    # criterion = FocalLoss().cuda()
    loss, total, correct = 0., 0, 0
    model.eval()

    with torch.no_grad():
        all_preds = []
        all_gts = []
        for batch, (examples, targets) in enumerate(loader):
            examples, targets = examples.to(device), targets.to(device)
            outputs, con_loss, _, _, _ = model(examples)
            loss = (1 - 0.8) * criterion(outputs, targets) + 0.8 * con_loss

            out_seg = outputs
            _, pred = torch.max(out_seg, dim=1, keepdim=True)
            pred = pred.detach().cpu().numpy().astype(np.float32)
            targets = targets.detach().cpu().numpy().astype(np.float32)

            all_preds.append(pred)
            all_gts.append(targets)
            total += targets.shape[0]

    kappa, oa = metrics(np.concatenate([p.ravel() for p in all_preds]),
                           np.concatenate([p.ravel() for p in all_gts]).ravel())
    print(kappa, oa)
    loss /= total

    return kappa, loss

def inference_fedmuti_iou(model, loader, device):
    if loader is None:
        return 0.0, 0.0  # Return 0 if loader is None

    criterion = nn.CrossEntropyLoss().to(device)
    loss, total, correct = 0., 0, 0
    model.eval()

    with torch.no_grad():
        all_preds = []
        all_gts = []
        for batch, (examples, targets) in enumerate(loader):
            examples, targets = examples.to(device), targets.to(device)
            outputs, con_loss, _, _ = model(examples)
            loss = (1 - 0.8) * criterion(outputs, targets) + 0.8 * con_loss

            out_seg = outputs
            _, pred = torch.max(out_seg, dim=1, keepdim=True)
            pred = pred.detach().cpu().numpy().astype(np.float32)
            targets = targets.detach().cpu().numpy().astype(np.float32)

            all_preds.append(pred)
            all_gts.append(targets)
            total += targets.shape[0]

    # Check if predictions or ground truths are empty
    if len(all_preds) == 0 or len(all_gts) == 0:
        print("Warning: No data collected, returning 0 for IoU and loss.")
        return 0.0, 0.0  # Return 0 if no data is collected

    # If data is available, calculate IoU and loss
    oa, iou = metrics_iou(np.concatenate([p.ravel() for p in all_preds]),
                           np.concatenate([p.ravel() for p in all_gts]).ravel())
    print(oa, iou)

    loss /= total

    return iou, loss


def get_acc_avg(acc_types, clients, model, device):
    acc_avg = {}
    # acc_types = ["valid"]
    for type in acc_types:
        acc_avg[type] = 0.
        num_examples = 0
        for client_id in range(len(clients)):

            # IoU, BSELoss
            acc_client, _ = clients[client_id].inference(model, type="valid", device=device)
            if acc_client is not None:
                acc_avg[type] += acc_client * len(clients[client_id].loaders[type].dataset)
                num_examples += len(clients[client_id].loaders[type].dataset)
        acc_avg[type] = acc_avg[type] / num_examples if num_examples != 0 else None
        # acc_avg[type] = acc_avg[type] /
    return acc_avg



def printlog_stats(quiet, logger, loss_avg, acc_avg, acc_types, lr, round, iter, iters):
    if not quiet:
        print(f'        Iteration: {iter}', end='')
        if iters is not None: print(f'/{iters}', end='')
        print()
        print(f'        Learning rate: {lr}')
        print(f'        Average running loss: {loss_avg:.6f}')
        for type in acc_types:
            print(f'        Average {types_pretty[type]} accuracy: {acc_avg[type]:.3%}')

    if logger is not None:
        logger.add_scalar('Learning rate (Round)', lr, round)
        logger.add_scalar('Learning rate (Iteration)', lr, iter)
        logger.add_scalar('Average running loss (Round)', loss_avg, round)
        logger.add_scalar('Average running loss (Iteration)', loss_avg, iter)
        for type in acc_types:
            logger.add_scalars('Average accuracy (Round)', {types_pretty[type].capitalize(): acc_avg[type]}, round)
            logger.add_scalars('Average accuracy (Iteration)', {types_pretty[type].capitalize(): acc_avg[type]}, iter)
        logger.flush()

def get_weights(local_protos):
    global_protos = torch.stack(local_protos)
    global_protos = torch.mean(global_protos, dim=0)
    cosine_similarities = []

    for tensor in local_protos:
        cosine_similarity = F.cosine_similarity(tensor.flatten(), global_protos.flatten(), dim=0)
        cosine_similarities.append(cosine_similarity.item())

    sum_of_similarities = sum(cosine_similarities)
    weights_list = [similarity / sum_of_similarities for similarity in cosine_similarities]

    return weights_list

def set_clients(args, Client, client_modalities):
    clients = []

    clients_list = [f"_{i+1}" for i in range(args.num_clients)]
    dataset_path = "../dataset/data-GLM/"

    if args.algorithm == "CTO":
        clients_list = ["CTO"]
    if args.algorithm == "LTO":
        clients_list = clients_list[:1]

    if args.mode == 'hsi':
        modality_order = ['hsi', 'lidar', 'rgb']
    # 定义模态顺序
    if args.mode == 'rgb':
        modality_order = ['rgb', 'lidar']

    for i, client_id in enumerate(clients_list):
        if i >= len(client_modalities):
            raise IndexError(f"client_modalities 列表不足，缺少 Client {i} 的模态信息")

        client_modal = client_modalities[i]
        modality_flags = [mod in client_modal for mod in modality_order]

        clients.append(Client(
            args=args,
            path_root=dataset_path,
            client_name=client_id,
            modality_flags=modality_flags
        ))

    return clients



def aggregate_parameters(args,model,updates, num_examples, weights_list, client_modalities):
    update_avg = weighted_average_updates(updates, num_examples, weights_list, client_modalities)
    v = deepcopy(update_avg)
    for key in model.state_dict():
        if model.state_dict()[key].type() == v[key].type():
            model.state_dict()[key] -= v[key] * args.server_lr


class BaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()

        self.base = base
        self.head = head

    def forward(self, x):
        out, x = self.base(x)
        out = self.head(out)

        return out, x