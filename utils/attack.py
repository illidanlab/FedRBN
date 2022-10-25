"""Adversarial attack"""
import torch
import numpy as np
from advertorch.utils import batch_multiply
from advertorch.attacks import Attack, LabelMixin


def fgsm(gradz, step_size):
    """
    https://github.com/mahyarnajibi/FreeAdversarialTraining/blob/d70774030871fa3207e09ce8528c1b84cd690603/lib/utils.py#L36
    """
    return step_size*torch.sign(gradz)


class FreeAttack(Attack):
    """Adversarial Training for Free!

    NOTE You have to manually reduce the epochs which is traded for the number of inner loop.

    Reference:
        Shafahi, A., Najibi, M., Ghiasi, A., Xu, Z., Dickerson, J., Studer, C., Davis, L. S., Taylor, G., & Goldstein, T. (2019). Adversarial Training for Free! NeurIPS. http://arxiv.org/abs/1904.12843

        https://github.com/mahyarnajibi/FreeAdversarialTraining/blob/d70774030871fa3207e09ce8528c1b84cd690603/main_free.py
    """
    def __init__(self, predict, loss_fn, batch_shape, device, data_mean, data_std,
                 clip_min=0., clip_max=1.0, fgsm_step_size=4.0, eps=4.0, nb_iter=4):
        super(FreeAttack, self).__init__(predict, loss_fn, clip_min, clip_max)
        if batch_shape[0] is None or batch_shape[0] < 0:
            batch_shape[0] = 128
        self.global_noise = torch.zeros(batch_shape, dtype=torch.float, device=device)
        self.device = device

        assert batch_shape[1] == len(data_mean), f"Invalid num of channels for mean: get {len(data_mean)} while expect {batch_shape[1]}"
        assert batch_shape[1] == len(data_std), f"Invalid num of channels for std: get {len(data_std)} while expect {batch_shape[1]}"
        self.mean = torch.Tensor(np.array(data_mean)[:, np.newaxis, np.newaxis])
        self.mean = self.mean.expand(batch_shape[1], batch_shape[2], batch_shape[3]).to(device)
        self.std = torch.Tensor(np.array(data_std)[:, np.newaxis, np.newaxis])
        self.std = self.std.expand(batch_shape[1], batch_shape[2], batch_shape[3]).to(device)

        self.noise_batch = None
        self.fgsm_step_size = fgsm_step_size
        self.eps = eps
        self.nb_iter = nb_iter

    def perturb(self, x, y=None):
        x = x.detach().clone()

        assert x.size(0) <= self.global_noise.size(0), f"Input batch size is larger than cache: {x.size(0)} (input) > {self.global_noise.size(0)} (expected max)"
        # DO NOT use .to expression here, which will create and return a non-leaf node.
        self.noise_batch = torch.tensor(self.global_noise[0:x.size(0)].detach(), requires_grad=True, device=self.device)
        out = x + self.noise_batch  # noise dasta
        out.clamp_(self.clip_min, self.clip_max)

        # To be consistent with normal preprocessing, we do not normalize.
        # out.sub_(self.mean).div_(self.std)

        return out

    def step(self):
        assert self.noise_batch is not None, "Run `perturb` first."
        assert self.noise_batch.grad is not None, "Grad is not backward yet."
        # Update the noise for the next iteration
        pert = fgsm(self.noise_batch.grad, self.fgsm_step_size)  # TODO try to use 8.0?? But is this mean the same as the steps at PGD?
        self.global_noise[0:self.noise_batch.size(0)] += pert.data
        self.global_noise.clamp_(-self.eps, self.eps)

        self.noise_batch = None


class AffineAttack(Attack, LabelMixin):
    """Affine attack used in robust federated learning.

    A(x) = \\Lambda x + \\delta
    eps = 1/lambda

    NOTE clip_min, clip_max are not used.

    Reference:
        Reisizadeh, A., Farnia, F., Pedarsani, R., & Jadbabaie, A. (2020, June 15). Robust Federated Learning: The Case of Affine Distribution Shifts. Advances in Neural Information Processing Systems. http://arxiv.org/abs/2006.08907
    """
    def __init__(self, predict, loss_fn, clip_min=0., clip_max=1.0, eps=10., nb_iter=2):
        super(AffineAttack, self).__init__(predict, loss_fn, clip_min, clip_max)
        self.nb_iter = nb_iter
        self.eps = eps
        self.delta = None
        self.Lambda = None
        self.targeted = False

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)

        x_adv, self.delta, self.Lambda = affine_perturb_iterative(
            x, y, self.predict, self.nb_iter, self.loss_fn,
            lmbd=1./self.eps, delta_init=self.delta, Lambda_init=self.Lambda
        )
        return x_adv


def affine_perturb_iterative(
        xvar, yvar, predict, nb_iter, loss_fn, lmbd=0.1,
        delta_init=None, Lambda_init=None, minimize=False):
    """Iteratively maximize the loss over the input.
    """
    step_size = 1/2. / lmbd

    batch_size, n_channel, H, W = xvar.shape
    flat_batch = xvar.view(-1, n_channel * H * W)
    if delta_init is not None:
        delta = delta_init
    else:
        delta = torch.zeros((1, n_channel, H, W), dtype=torch.float, device='cuda')

    if Lambda_init is not None:
        Lambda = Lambda_init
    else:
        Lambda = torch.zeros((n_channel * H * W, n_channel * H * W), dtype=torch.float,
                             device='cuda')

    delta.requires_grad_()
    Lambda.requires_grad_()
    for ii in range(nb_iter):
        outputs = predict(torch.mm(flat_batch, Lambda).view(xvar.shape) + delta)
        loss = loss_fn(outputs, yvar) - lmbd * torch.norm(delta) ** 2 - lmbd * torch.norm(Lambda) ** 2
        if minimize:
            loss = -loss

        loss.backward()
        grad = delta.grad.data
        delta.data = delta.data + batch_multiply(step_size, grad)
        delta.grad.data.zero_()

        grad = Lambda.grad.data
        Lambda.data = Lambda.data + batch_multiply(step_size, grad)
        Lambda.grad.data.zero_()

    x_adv = (flat_batch @ Lambda).view(xvar.shape) + delta
    return x_adv, delta, Lambda
