import sys
import copy
import torch
from torch import nn
from tqdm import tqdm
from utils.attack import FreeAttack
from advertorch.context import ctx_noparamgrad_and_eval


def train_free(model, data_loader, optimizer, loss_fun, device, adversary=None,
               adv_lmbd=1., att_BNn=False):
    """Train with free attack

    TODO need to reduce epochs according to the inner repeat.
    TODO use adv_lmbd=1. for train free
    """
    assert isinstance(adversary, FreeAttack)
    assert adv_lmbd == 1, "Train free only can use adv_lmbd=1."

    model.train()
    loss_all = 0
    total = 0
    correct = 0
    # Use adversary to perturb data.
    for data, target in tqdm(data_loader, file=sys.stdout):
        data = data.to(device)
        target = target.to(device)
        for rep_iter in range(adversary.nb_iter):
            optimizer.zero_grad()
            model.set_bn_mode(False)  # set clean mode

            if adv_lmbd < 1.:
                # clean data
                logits = model(data)
                clf_loss_clean = loss_fun(logits, target)
            else:
                clf_loss_clean = 0.

            # noise data
            # ---- use adv ----
            if att_BNn:
                model.set_bn_mode(True)  # set noise mode
            noise_data = adversary.perturb(data)
            noise_target = target
            # -----------------

            model.set_bn_mode(True)  # set noise mode
            logits_noise = model(noise_data)
            clf_loss_noise = loss_fun(logits_noise, noise_target)

            loss = (1 - adv_lmbd) * clf_loss_clean + adv_lmbd * clf_loss_noise

            loss_all += loss.item()

            if adv_lmbd < 1.:
                output = torch.cat([logits, logits_noise], dim=0)
                _target = torch.cat([target, noise_target], dim=0)
            else:
                output = logits_noise
                _target = noise_target
            total += _target.size(0)
            pred = output.data.max(1)[1]
            correct += pred.eq(_target.view(-1)).sum().item()

            loss.backward()
            # torch.onnx.export(model, adversary.noise_batch, 'temp_free_model.onnx',
                              # input_names=['noise_batch'])
            adversary.step()  # update adversary gradients

            optimizer.step()
    return loss_all / len(data_loader), correct / total


def train_meta(model: nn.Module, data_loader, optimizer, loss_fun, device, adversary=None,
               adv_lmbd=0.5):
    """Fed MAML

    Code refer to: https://github.com/CharlieDinh/pFedMe/blob/master/FLAlgorithms/users/userperavg.py (pFedMe)
    """
    do_perturbation = adversary is not None
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    # ordinary training.
    model.set_bn_mode(False)  # set clean mode
    init_stat_dict = None
    for i, (data, target) in enumerate(tqdm(data_loader, file=sys.stdout)):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)

        if i % 2 == 0:
            with torch.no_grad():
                init_stat_dict = copy.deepcopy(model.state_dict())

            # step 1 opt
            loss, output, target = one_step_opt(adv_lmbd, adversary, data,
                                                loss_fun, model, target)
            optimizer.step()
            # print(f"### optimizer.state_dict() {optimizer.state_dict()['param_groups'][0].keys()}")
            # one_step_sgd(optimizer.param_groups[0]['params'], optimizer.param_groups[0]['lr']*0.05)
        else:
            # step 2 opt
            loss, output, target = one_step_opt(adv_lmbd, adversary, data,
                                                loss_fun, model, target)

            # restore old parameters
            with torch.no_grad():
                for pname, p in model.named_parameters():
                    p.data.copy_(init_stat_dict[pname])  # will not copy non-param, like bn stat
                init_stat_dict = None

            # optimizer.step()
            assert len(optimizer.param_groups) == 1, f"len(optimizer.param_groups) = " \
                                                     f"{len(optimizer.param_groups)}"
            one_step_sgd(optimizer.param_groups[0]['params'], optimizer.param_groups[0]['lr']*0.05)

        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
    if init_stat_dict is not None:  # drop the last non-meta step.
        with torch.no_grad():
            model.load_state_dict(init_stat_dict)
    return loss_all / len(data_loader), correct / total


def one_step_sgd(parameters, lr):
    for p in parameters:
        if p.grad is None:
            continue
        d_p = p.grad.data
        p.data.add_(-lr, d_p)


def one_step_opt(adv_lmbd, adversary, data, loss_fun, model, target):
    do_perturbation = adversary is not None

    if do_perturbation:
        model.set_bn_mode(False)
    output = model(data)
    loss = loss_fun(output, target)
    if do_perturbation:
        clf_loss_clean = loss

        # noise data
        # ---- use adv ----
        with ctx_noparamgrad_and_eval(model):
            noise_data = adversary.perturb(data, target)
        noise_target = target
        # -----------------

        model.set_bn_mode(True)  # set noise mode
        logits_noise = model(noise_data)
        clf_loss_noise = loss_fun(logits_noise, noise_target)

        loss = (1 - adv_lmbd) * clf_loss_clean + adv_lmbd * clf_loss_noise

        output = torch.cat([output, logits_noise], dim=0)
        target = torch.cat([target, noise_target], dim=0)
    loss.backward()
    return loss, output, target
