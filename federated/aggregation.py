"""Aggregate models in FL."""
import copy
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from advertorch.attacks import LinfPGDAttack
from collections import defaultdict


class ModelAccumulator(object):
    """Accumulate models.
    If local_bn is True, a dict of bn layers will be kept for all users.

    Concepts:
        running_model: The model used to train. This is not persistent storage. Load by call
            `load_model` at practice.
        server_state_dict: The current state_dict in server.
        accum_state_dict: The accumulated state_dict which will accumulate the trained results
            from running model and update to server_state_dict when fulfilled.

    Args:
        running_model: Model to init state_dict shape and bn layers.
        n_accum: Number of models to accumulate per round. If retrieve before this value,
            an error will raise.
        num_model: Total number of models. Used if local_bn is True.
        local_bn: Whether to keep local bn for all users.
        raise_err_on_early_accum: Raise error if update model when not all users are accumulated.
    """
    def __init__(self, running_model: nn.Module, n_accum, num_model, local_bn=False,
                 raise_err_on_early_accum=True):
        """
        TODO set local_bn to be True for FedRBN, FedBN
        """
        self.n_accum = n_accum
        self._cnt = 0
        self.local_bn = local_bn
        self._weight_sum = 0
        self.raise_err_on_early_accum = raise_err_on_early_accum
        with torch.no_grad():
            self.server_state_dict = {
                k: copy.deepcopy(v) for k, v in running_model.state_dict().items()
            }
            self._accum_state_dict = {
                k: torch.zeros_like(v) for k, v in running_model.state_dict().items()
            }
            if local_bn:
                self.local_state_dict = [{
                    k: copy.deepcopy(v) for k, v in running_model.state_dict().items() if 'bn' in k
                } for _ in range(num_model)]
            else:
                self.local_state_dict = []

    def state_dict(self):
        return {
            'server': self.server_state_dict,
            'clients': self.local_state_dict,
        }

    def load_state_dict(self, state_dict: dict):
        self.server_state_dict = state_dict['server']
        local_state_dict = state_dict['clients']
        if self.local_bn:
            assert len(local_state_dict) > 0, "Not found local state dict when local_bn is set."
            # num_model
            assert len(local_state_dict) == len(self.local_state_dict), \
                f"Load {len(local_state_dict)} local states while expected" \
                f" {len(self.local_state_dict)}"
        else:
            assert len(local_state_dict) == 0, "Found local bn state when local_bn is not set."
        self.local_state_dict = local_state_dict

    def add(self, model_idx, model, weight):
        """Use weight = 1/n_accum to average.
        """
        if self._cnt >= self.n_accum:  # note cnt starts from 0
            raise RuntimeError(f"Try to accumulate {self._cnt}, while only {self.n_accum} models"
                               f" are allowed. Did you forget to reset after accumulated?")
        with torch.no_grad():
            for key in self._accum_state_dict:
                if self.local_bn:
                    if 'bn' in key:
                        self.local_state_dict[model_idx][key].data.copy_(model.state_dict()[key])
                    else:
                        temp = weight * model.state_dict()[key]
                        self._accum_state_dict[key].data.add_(temp)
                else:
                    if 'num_batches_tracked' in key:
                        # if self._cnt == 0:
                        # num_batches_tracked is a non trainable LongTensor and
                        # num_batches_tracked are the same for all clients for the given datasets
                        self._accum_state_dict[key].data.copy_(model.state_dict()[key])
                    else:
                        temp = weight * model.state_dict()[key]
                        self._accum_state_dict[key].data.add_(temp)
        self._cnt += 1  # DO THIS at the END such that start from 0.
        self._weight_sum += weight

    @property
    def accumulated_count(self):
        return self._cnt

    @property
    def accum_state_dict(self):
        self.check_full_accum()
        return self._accum_state_dict

    def load_model(self, running_model: nn.Module, model_idx: int, ignore_local_bn=False):
        """Load server model and local BN states into the given running_model."""
        running_model.load_state_dict(self.server_state_dict)
        if self.local_bn and not ignore_local_bn:
            with torch.no_grad():
                for k, v in self.local_state_dict[model_idx].items():
                    running_model.state_dict()[k].data.copy_(v)

    def update_server_and_reset(self, beta=0.):
        """Load accumulated state_dict to server_model and
        reset accumulated values but not local bn."""
        self.check_full_accum()
        weight_norm = 1. / self._weight_sum
        with torch.no_grad():
            # update server
            for k in self.server_state_dict:
                if beta > 0 and 'num_batches_tracked' not in k:
                    self.server_state_dict[k].data.mul_(beta).add_(
                        (1-beta) * self._accum_state_dict[k].data * weight_norm)
                else:
                    self.server_state_dict[k].data.copy_(self._accum_state_dict[k].data * weight_norm)

            # reset
            self._cnt = 0
            self._weight_sum = 0
            for k in self._accum_state_dict:
                self._accum_state_dict[k].data.zero_()

    def check_full_accum(self):
        """Check if the number of accumulated models reaches the expected value (n_accum)."""
        if self.raise_err_on_early_accum:
            assert self._cnt == self.n_accum, "Retrieve before all models are accumulated."

    def copy_dual_noise_bn(self, noised_src_idx, dst_idx, diff_coef=0.):
        """Copy the noise BN in dual BN."""
        assert self.local_bn
        copy_dual_noise_bn(self.local_state_dict[noised_src_idx],
                           self.local_state_dict[dst_idx],
                           diff_coef=diff_coef)

    def copy_multi_dual_noise_bn(self, noised_src_idxs, dst_idx, diff_coef=0., src_weight_mode='transBN'):
        assert self.local_bn
        copy_multi_dual_noise_bn(
            [self.local_state_dict[i] for i in noised_src_idxs],
            self.local_state_dict[dst_idx],
            diff_coef=diff_coef, src_weight_mode=src_weight_mode)

    def duplicate_dual_clean_bn(self, idx):
        duplicate_dual_clean_bn(self.local_state_dict[idx])

    def aggregate_local_bn(self):
        """Will average all local variables into the global model.
        NOTE only valid for equal client sample size.
        """
        with torch.no_grad():
            is_init = True
            n_client = len(self.local_state_dict)
            for model_idx in range(n_client):
                for k, v in self.local_state_dict[model_idx].items():
                    if 'num_batches_tracked' in k: continue
                    if is_init:
                        self.server_state_dict[k].data.zero_()
                    self.server_state_dict[k].data.add_(v / float(n_client))
                is_init = False

# NOTE: this class will cause unknown state because we may load to same running_model multiple times.
# class InstantModel(object):
#     """Load state dict to model on need, like a list. Read only."""
#     def __init__(self, running_model: nn.Module, model_accum: ModelAccumulator):
#         self.running_model = running_model
#         self.model_accum = model_accum
#
#     def __getitem__(self, index: int):
#         self.model_accum.load_model(self.running_model, index)
#         return self.running_model


def duplicate_dual_clean_bn(model):
    """Copy the in-layer clean bn to in-layer noise bn for each dual BN."""
    found_noise_bn = False
    if not isinstance(model, dict):
        model_state_dict = model.state_dict()
    else:
        model_state_dict = model
    for key in model_state_dict:
        if 'noise_bn' in key:
            found_noise_bn = True
            clean = model_state_dict[key.replace('noise_bn', 'clean_bn')].data
            model_state_dict[key].data.copy_(clean)

    if not found_noise_bn:
        raise ValueError(f"Not found noise BN. Please make suer you are using dual BN "
                         f"in your model.")

def copy_dual_noise_bn(noised_src_model, dst_model, diff_coef=0.):
    """Copy the noise BN in dual BN."""
    found_noise_bn = False
    eps = 1e-10
    if not isinstance(noised_src_model, dict) or not isinstance(dst_model, dict):
        dst_state_dict = dst_model.state_dict()
        noised_src_state_dict = noised_src_model.state_dict()
    else:
        dst_state_dict = dst_model
        noised_src_state_dict = noised_src_model
    for clean_key in dst_state_dict:
        if 'clean_bn' in clean_key:  # only copy noise bn
            found_noise_bn = True
            if 'running_mean' in clean_key or 'running_var' in clean_key:
                noise_key = clean_key.replace('clean_bn', 'noise_bn')
                src_clean = noised_src_state_dict[clean_key].data
                src_noise = noised_src_state_dict[noise_key].data
                dst_clean = dst_state_dict[clean_key].data
                temp = src_noise
                if diff_coef > 0.:
                    if 'var' in clean_key:
                        diff = dst_clean / (src_clean + eps)
                        temp = temp * (diff ** diff_coef)
                    elif 'mean' in clean_key:
                        diff = dst_clean - src_clean
                        temp = temp + diff * diff_coef
                    else:
                        raise RuntimeError()
                dst_state_dict[noise_key].data.copy_(temp)

    if not found_noise_bn:
        raise ValueError(f"Not found clean BN or dual BN. Please make suer you are using dual BN "
                         f"in your model.")

def copy_multi_dual_noise_bn(noised_src_models, dst_model, diff_coef=0., src_weight_mode='transBN'):
    """Copy the noise BN in dual BN."""
    if len(noised_src_models) <= 1:
        return copy_dual_noise_bn(noised_src_models[0], dst_model, diff_coef=diff_coef)

    if not isinstance(noised_src_models[0], dict) or not isinstance(dst_model, dict):
        dst_state_dict = dst_model.state_dict()
        noised_src_state_dicts = [m.state_dict() for m in noised_src_models]
    else:
        dst_state_dict = dst_model
        noised_src_state_dicts = noised_src_models

    found_noise_bn = False
    eps = 1e-10

    # Compute diff weight
    src_weights = defaultdict(list)
    candidates = defaultdict(list)
    for src_state_dict in noised_src_state_dicts:
        for clean_key in dst_state_dict:
            if 'clean_bn' in clean_key:  # only copy noise bn
                found_noise_bn = True
                if 'running_mean' in clean_key or 'running_var' in clean_key:
                    noise_key = clean_key.replace('clean_bn', 'noise_bn')
                    src_clean = src_state_dict[clean_key].data
                    src_noise = src_state_dict[noise_key].data
                    dst_clean = dst_state_dict[clean_key].data
                    temp = src_noise
                    if diff_coef > 0.:
                        if 'var' in clean_key:
                            diff = dst_clean / (src_clean + eps)
                            temp = temp * (diff ** diff_coef)
                        elif 'mean' in clean_key:
                            diff = dst_clean - src_clean
                            temp = temp + diff * diff_coef
                        else:
                            raise RuntimeError()
                    candidates[noise_key].append(temp)

                    # Compute weights
                    if 'running_mean' in clean_key:
                        clean_key_mean = clean_key
                        clean_key_var = clean_key.replace('mean', 'var')
                    elif 'running_var' in clean_key:
                        clean_key_mean = clean_key.replace('var', 'mean')
                        clean_key_var = clean_key
                    else:
                        raise ValueError()
                    if src_weight_mode.lower() == 'cos':  # cosine similarity
                        a = F.cosine_similarity(src_state_dict[clean_key_mean], dst_state_dict[clean_key_mean], 0, 1e-8) \
                            + F.cosine_similarity(src_state_dict[clean_key_var], dst_state_dict[clean_key_var], 0, 1e-8)
                        a = a * 1e2 / 2.  # T=0.01
                        src_weights[noise_key].append(a)
                    elif src_weight_mode.lower() == 'rbf':  # cosine similarity
                        a = - 1e5 * torch.sum((src_state_dict[clean_key_mean] - dst_state_dict[clean_key_mean]) ** 2) / len(dst_state_dict[clean_key_mean])
                        src_weights[noise_key].append(a)

    if not found_noise_bn:
        raise ValueError(f"Not found clean BN or dual BN. Please make suer you are using dual BN "
                         f"in your model.")
    if src_weight_mode != 'eq':
        for k in src_weights:
            src_weights[k] = torch.stack(src_weights[k], dim=0)  # shape: [n_user, weight dim]
    if src_weight_mode in ['rbf', 'cos']:
        src_weights = torch.cat([src_weights[k].unsqueeze(1) for k in src_weights], dim=1)
        src_weights = torch.mean(src_weights, dim=1)  # marginalize layers

        src_weights = torch.softmax(src_weights, 0)
        # norm_factor = torch.sum(src_weights, dim=0)
        # print(f"### src {src_weight_mode} weights", src_weights.cpu().numpy())

    for noise_key in dst_state_dict:
        if 'noise_bn' in noise_key:  # only copy noise bn
            if 'running_mean' in noise_key or 'running_var' in noise_key:
                temp = torch.zeros_like(dst_state_dict[noise_key])
                if src_weight_mode.lower() in ['rbf', 'cos']:
                    # norm_factor = torch.sum(src_weights, dim=0)
                    for v, a in zip(candidates[noise_key], src_weights):
                        # print("## a / norm_factor", a / norm_factor)
                        temp = temp + v * a
                elif src_weight_mode.lower() == 'eq':
                    norm_factor = float(len(candidates[noise_key]))
                    for v in candidates[noise_key]:
                        temp = temp + v / norm_factor
                else:
                    raise ValueError(f"src_weight_mode: {src_weight_mode}")
                dst_state_dict[noise_key].data.copy_(temp)


def copy_noise_bn(noised_src_model, dst_model, diff_coef=0.):
    """Copy the noise BN in dual BN."""
    assert diff_coef == 0, "Not support non-zero diff_coef since no clean ref is available."
    found_bn = False
    eps = 1e-10
    for key in dst_model.state_dict():
        if 'bn' in key:  # only copy noise bn
            found_bn = True
            if 'running_mean' in key or 'running_var' in key:
                src_noise = noised_src_model.state_dict()[key].data
                dst_model.state_dict()[key].data.copy_(src_noise)

    if not found_bn:
        raise ValueError(f"Not found BN. Please make suer you are using BN "
                         f"in your model.")
