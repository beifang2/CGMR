import os
import random
import time
from collections import defaultdict
from copy import deepcopy
from datetime import timedelta
from os import environ

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.base import get_acc_avg, printlog_stats, set_clients , aggregate_parameters
import torch.optim as toptim
from client.client_cgmr import Client
from client.client_cgmr_rgb import Client_RGB
import ast

class CGMR():
    def __init__(self, args, model):
        self.start_time = time.time()
        self.args = args
        self.acc_types = ['train', 'valid']
        self.p_clients = None
        self.m = max(int(self.args.frac_clients * self.args.num_clients), 1)
        self.last_round = -1
        self.iter = 0
        self.global_protos = []
        self.global_cov_matrices = {}

        self.optimizer = toptim.SGD(model.parameters(), **self.args.optim_args)
        self.model = model

    def train(self):
        _Client = None
        if self.args.mode == 'hsi':
            input_size = (3,) + tuple([19, 11, 11])
            _Client = Client
        if self.args.mode == 'rgb':
            input_size = (3,) + tuple([4, 512, 512])
            _Client = Client_RGB

        with open(rf'./dataset/{self.args.dataset}/client_modalities_train.txt', 'r', encoding='utf-8') as f:
            content = f.read()

        client_modalities = ast.literal_eval(content.split('=', 1)[1].strip())

        clients = set_clients(self.args, _Client, client_modalities)
        torch.use_deterministic_algorithms(False)

        logger = SummaryWriter(f'./runs/{self.args.name}')



        fake_input = torch.zeros(input_size).to(self.args.device)
        logger.add_graph(self.model, fake_input)

        acc_avg = get_acc_avg(self.acc_types, clients, self.model, self.args.device)
        acc_avg_best = acc_avg[self.acc_types[1]]

        # Print and log initial stats
        if not self.args.quiet:
            print('Training:')
        loss_avg, lr = torch.nan, torch.nan

        init_end_time = time.time()

        for round in range(self.last_round + 1, self.args.rounds):
            if not self.args.quiet:
                print(f'    Round: {round + 1}' + (f'/{self.args.rounds}' if self.args.iters is None else ''))

            client_ids = np.random.choice(range(self.args.num_clients), self.m, replace=False, p=self.p_clients)

            updates, num_examples, max_iters, loss_tot = [], [], 0, 0.
            client_examples_list = []
            client_cov_matrices_list = []
            for i, client_id in enumerate(client_ids):
                if not self.args.quiet:
                    print(f'        Client: {client_id} ({i + 1}/{self.m})')

                client_model = deepcopy(self.model)
                self.optimizer.__setstate__({'state': defaultdict(dict)})
                self.optimizer.param_groups[0]['params'] = list(client_model.parameters())

                client_update, client_num_examples, client_num_iters, client_loss, epoch_cov_matrices = clients[client_id].train(
                    model=client_model, optim=self.optimizer, device=self.args.device, cov_matrices=self.global_cov_matrices)
                client_examples_list.append(client_num_examples)
                client_cov_matrices_list.append(epoch_cov_matrices)

                if client_num_iters > max_iters: max_iters = client_num_iters

                if client_update is not None:
                    updates.append(deepcopy(client_update))
                    loss_tot += client_loss * client_num_examples
                    num_examples.append(client_num_examples)

            self.iter += max_iters
            lr = self.optimizer.param_groups[0]['lr']

            weights_list = [client_examples / sum(client_examples_list) for client_examples in client_examples_list]
            print(weights_list)
            if len(updates) > 0:
                aggregate_parameters(self.args,self.model, updates, num_examples, weights_list, client_modalities)
                self.global_cov_matrices = aggregate_cov_matrices(client_cov_matrices_list, weights_list, client_modalities, self.global_cov_matrices)
                if round % self.args.server_stats_every == 0:
                    loss_avg = loss_tot / sum(num_examples)
                    acc_avg = get_acc_avg(self.acc_types, clients, self.model, self.args.device)
                    if acc_avg[self.acc_types[1]] > acc_avg_best:
                        acc_avg_best = acc_avg[self.acc_types[1]]
            if round % self.args.server_stats_every == 0:
                printlog_stats(self.args.quiet, logger, loss_avg, acc_avg, self.acc_types, lr, round + 1, self.iter, self.args.iters)

            if self.args.iters is not None and self.iter >= self.args.iters: break

        train_end_time = time.time()
        os.makedirs('./snapshot', exist_ok=True)
        torch.save(self.model.state_dict(), "./snapshot/" + self.args.name + ".pth")

        test_end_time = time.time()

        print(f'    Train time: {timedelta(seconds=int(train_end_time - init_end_time))}')
        print(f'    Total time: {timedelta(seconds=int(time.time() - self.start_time))}')

        if logger is not None:
            logger.close()


def aggregate_cov_matrices(all_client_covs, weights, client_modalities, previous_cov_matrices=None):
    aggregated = {}

    all_modalities = set()
    for client_cov in all_client_covs:
        all_modalities.update(client_cov.keys())
    if previous_cov_matrices:
        all_modalities.update(previous_cov_matrices.keys())

    for modality in all_modalities:
        aggregated[modality] = {}
        class_indices = set()

        for client_cov in all_client_covs:
            class_indices.update(client_cov.get(modality, {}).keys())
        if previous_cov_matrices and modality in previous_cov_matrices:
            class_indices.update(previous_cov_matrices[modality].keys())

        for class_idx in class_indices:
            agg_cov, agg_mean = None, None
            total_weight = 0.0
            pca_model = None

            for client_idx, client_cov in enumerate(all_client_covs):
                if modality.lower() not in client_modalities[client_idx]:
                    continue
                if class_idx not in client_cov.get(modality, {}):
                    continue

                entry = client_cov[modality][class_idx]
                cov = entry['cov']
                mean = entry['mean']
                weight = weights[client_idx]

                if agg_cov is None:
                    agg_cov = cov * weight
                    agg_mean = mean * weight
                    # pca_model = entry['pca_model']
                else:
                    agg_cov += cov * weight
                    agg_mean += mean * weight

                total_weight += weight

            if total_weight > 0:
                aggregated[modality][class_idx] = {
                    'cov': agg_cov / total_weight,
                    'mean': agg_mean / total_weight,
                }
            elif previous_cov_matrices and modality in previous_cov_matrices and class_idx in previous_cov_matrices[modality]:
                aggregated[modality][class_idx] = previous_cov_matrices[modality][class_idx]
                print(f"[保持旧值] 模态='{modality}' 类别={class_idx} 本轮缺失，保留上一轮协方差")

    return aggregated
