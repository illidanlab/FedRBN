import numpy as np
import math
import torch
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_seed(seed=None):
    import random
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.values = []
        self.counter = 0

    def append(self, val):
        self.values.append(val)
        self.counter += 1

    def extend(self, items):
        self.values.extend(items)
        self.counter += len(items)

    @property
    def val(self):
        return self.values[-1]

    @property
    def avg(self):
        return sum(self.values) / len(self.values)

    def __repr__(self):
        values = self.values
        return ','.join([f" {metric}: {eval(f'np.{metric}')(values)}" for metric in ['mean', 'std', 'min', 'max']])

    @property
    def last_avg(self):
        if self.counter == 0:
            return self.latest_avg
        else:
            self.latest_avg = sum(self.values[-self.counter:]) / self.counter
            self.counter = 0
            return self.latest_avg


class CosineAnnealingLR(object):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
        \begin{aligned}
            \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
            & T_{cur} \neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}

    When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators. If the learning rate is set
    solely by this scheduler, the learning rate at each step becomes:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, T_max, eta_max=1e-2, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self._cur_lr = eta_max
        # super(CosineAnnealingLR, self).__init__(optimizer, last_epoch, verbose)

    def step(self):
        self._cur_lr = self._get_lr()
        self.last_epoch += 1
        return self._cur_lr

    def _get_lr(self):
        if self.last_epoch == 0:
            return self.eta_max
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return self._cur_lr + (self.eta_max - self.eta_min) * \
                    (1 - math.cos(math.pi / self.T_max)) / 2
        return (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / \
                (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) * \
                (self._cur_lr - self.eta_min) + self.eta_min
