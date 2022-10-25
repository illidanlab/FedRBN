from dis import dis
import sys
import numpy as np

import torch
from torch import nn
from advertorch.context import ctx_noparamgrad_and_eval
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from federated.core import ModelAccumulator
from federated.baselines import train_free
from utils.attack import FreeAttack, AffineAttack
from utils.utils import AverageMeter


def train_simple(model, data_loader, optimizer, loss_fun, device,
                 start_iter=0, max_iter=np.inf):

    model.train()
    loss_all = 0
    total = 0
    correct = 0
    max_iter = len(data_loader) if max_iter == np.inf else max_iter
    data_iterator = iter(data_loader)

    # ordinary training.
    for step in tqdm(range(start_iter, max_iter), file=sys.stdout):
    # for data, target in tqdm(data_loader, file=sys.stdout):
        try:
            data, target = next(data_iterator)
        except StopIteration:
            data_iterator = iter(data_loader)
            data, target = next(data_iterator)
        optimizer.zero_grad()

        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = loss_fun(output, target)

        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss.backward()
        optimizer.step()
    return loss_all / len(data_loader), correct / total


def train(model, data_loader, optimizer, loss_fun, device, adversary=None, adv_lmbd=0.5,
          start_iter=0, max_iter=np.inf, att_BNn=False, pnc_coef=0.,):
    if isinstance(adversary, FreeAttack):
        adv_lmbd = 1.  # FIXME this may be misleading.
        assert max_iter == np.inf or max_iter == len(data_loader)
        return train_free(model, data_loader, optimizer, loss_fun, device,
                          adversary=adversary, adv_lmbd=adv_lmbd, att_BNn=att_BNn)
    if isinstance(adversary, AffineAttack):
        adv_lmbd = 1.  # no clean loss

    model.train()
    loss_all = 0
    total = 0
    correct = 0
    max_iter = len(data_loader) if max_iter == np.inf else max_iter
    data_iterator = iter(data_loader)
    if adversary is None:
        # ordinary training.
        model.set_bn_mode(False)  # set clean mode
        for step in tqdm(range(start_iter, max_iter), file=sys.stdout, disable=True):
            # for data, target in tqdm(data_loader, file=sys.stdout):
            try:
                data, target = next(data_iterator)
            except StopIteration:
                data_iterator = iter(data_loader)
                data, target = next(data_iterator)
            optimizer.zero_grad()

            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = loss_fun(output, target)

            if pnc_coef > 0.:
                model.set_bn_mode(True)
                model.set_dbn_n_train(False)
                logits_noise = model(data)  # robust output
                loss_noise = loss_fun(logits_noise, target)  # adapt robust model
                model.set_dbn_n_train(True)  # reset
                model.set_bn_mode(False)

                loss_noise = pnc_coef * loss_noise
                loss_noise.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 10.)
                loss = (1. - pnc_coef) * loss

            loss_all += loss.item()
            total += target.size(0)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()

            loss.backward()
            optimizer.step()
    else:
        # Use adversary to perturb data.
        for step in tqdm(range(start_iter, max_iter), file=sys.stdout, disable=True):
            try:
                data, target = next(data_iterator)
            except StopIteration:
                data_iterator = iter(data_loader)
                data, target = next(data_iterator)
            optimizer.zero_grad()

            # clean data
            data = data.to(device)
            target = target.to(device)
            model.set_bn_mode(False)  # set clean mode
            logits = model(data)
            clf_loss_clean = loss_fun(logits, target)

            # noise data
            # ---- use adv ----
            if att_BNn:
                model.set_bn_mode(True)  # set noise mode
            with ctx_noparamgrad_and_eval(model):
                noise_data = adversary.perturb(data, target)
            noise_target = target
            # -----------------

            model.set_bn_mode(True)  # set noise mode
            logits_noise = model(noise_data)
            clf_loss_noise = loss_fun(logits_noise, noise_target)

            loss = (1 - adv_lmbd) * clf_loss_clean + adv_lmbd * clf_loss_noise

            loss_all += loss.item()

            output = torch.cat([logits, logits_noise], dim=0)
            target = torch.cat([target, noise_target], dim=0)
            total += target.size(0)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()

            loss.backward()
            optimizer.step()
    return loss_all / len(data_loader), correct / total


def train_prox(model, server_model, data_loader, optimizer, loss_fun, device, adversary=None,
               adv_lmbd=0.5, start_iter=0, max_iter=np.inf, mu=1e-3):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    max_iter = len(data_loader) if max_iter == np.inf else max_iter
    data_iterator = iter(data_loader)
    if adversary is None:
        model.set_bn_mode(False)  # set clean mode
        for step in tqdm(range(start_iter, max_iter), file=sys.stdout):
        # for data, target in tqdm(data_loader, file=sys.stdout):
            try:
                data, target = next(data_iterator)
            except StopIteration:
                data_iterator = iter(data_loader)
                data, target = next(data_iterator)
            optimizer.zero_grad()

            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = loss_fun(output, target)
            if step > 0:
                w_diff = torch.tensor(0., device=device)
                if isinstance(server_model, dict):  # state dict
                    for w_name, w_t in model.named_parameters():
                        w_diff += torch.pow(torch.norm(server_model[w_name] - w_t), 2)
                else:
                    for w, w_t in zip(server_model.parameters(), model.parameters()):
                        w_diff += torch.pow(torch.norm(w - w_t), 2)

                w_diff = torch.sqrt(w_diff)
                loss += mu / 2. * w_diff

            loss.backward()
            optimizer.step()

            loss_all += loss.item()
            total += target.size(0)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()
    else:
        for step in tqdm(range(start_iter, max_iter), file=sys.stdout, disable=True):
        # for data, target in tqdm(data_loader, file=sys.stdout):
            try:
                data, target = next(data_iterator)
            except StopIteration:
                data_iterator = iter(data_loader)
                data, target = next(data_iterator)
            optimizer.zero_grad()

            data = data.to(device)
            target = target.to(device)
            logits = model(data)
            clean_loss = loss_fun(logits, target)

            # noise data
            # ---- use adv ----
            with ctx_noparamgrad_and_eval(model):
                noise_data = adversary.perturb(data, target)
            noise_target = target

            model.set_bn_mode(True)  # set noise mode
            logits_noise = model(noise_data)
            clf_loss_noise = loss_fun(logits_noise, noise_target)

            loss = (1 - adv_lmbd) * clean_loss + adv_lmbd * clf_loss_noise
            # -----------------

            if step > 0:
                w_diff = torch.tensor(0., device=device)
                if isinstance(server_model, dict):  # state dict
                    for w_name, w_t in model.named_parameters():
                        w_diff += torch.pow(torch.norm(server_model[w_name] - w_t), 2)
                else:
                    for w, w_t in zip(server_model.parameters(), model.parameters()):
                        w_diff += torch.pow(torch.norm(w - w_t), 2)

                w_diff = torch.sqrt(w_diff)
                loss += mu / 2. * w_diff

            output = torch.cat([logits, logits_noise], dim=0)
            target = torch.cat([target, noise_target], dim=0)

            loss_all += loss.item()
            total += target.size(0)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()

            loss.backward()
            optimizer.step()

    return loss_all / len(data_loader), correct / total


def test_simple(model, data_loader, loss_fun, device, adversary=None):
    """Run test single model.

    Args:

    Returns:
        loss, acc
    """
    model.eval()

    noise_type = 1 if adversary else 0
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        if adversary:
            with ctx_noparamgrad_and_eval(model):  # make sure BN's are in eval mode
                data = adversary.perturb(data, target)

        with torch.no_grad():
            output = model(data)
            loss = loss_fun(output, target)

        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
    return loss_all / len(data_loader), correct / total

def test(model, data_loader, loss_fun, device,
         adversary=None, detector='gt', att_BNn=False,
         ):
    """Run test

    Args:
        detector: If None, then not predict noise.
        att_BNn: Attack the noise BN instead of clean BN.

    Returns:
        loss, acc, detect_acc. detect_acc is meaningless if `detector` is None.
    """
    model.eval()

    noise_type = 1 if adversary else 0
    loss_all = 0
    total = 0
    correct = 0
    adv_scores = AverageMeter()
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        model.set_bn_mode(is_noised=False)  # use clean mode to predict noise.
        if adversary:
            if att_BNn:
                model.set_bn_mode(True)  # set noise mode
            with ctx_noparamgrad_and_eval(model):  # make sure BN's are in eval mode
                data = adversary.perturb(data, target)


        if detector is None or detector == 'none':
            # use clean BN
            disc_pred = False
            model.set_bn_mode(is_noised=False)
        else:
            if detector == 'clean':
                disc_pred = False
            elif detector == 'noised':
                disc_pred = True
            elif detector == 'gt':
                disc_pred = noise_type > 0
            elif detector == 'rgt':
                disc_pred = noise_type <= 0
            else:
                raise ValueError(f"Invalid str detector: {detector}")
            model.set_bn_mode(is_noised=disc_pred)

        output = model(data)
        loss = loss_fun(output, target)

        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    if adv_scores.counter > 0:
        print(f"adv_scores: {adv_scores}")
    return loss_all / len(data_loader), correct / total
