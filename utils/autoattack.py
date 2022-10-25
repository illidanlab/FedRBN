import torch
from advertorch.attacks import Attack, LabelMixin
from torch.nn.modules.loss import _Loss
from torch.nn import CrossEntropyLoss


class AutoAttack(Attack, LabelMixin):
    """Auto select the best attack based on the attack acc.
    Code based on https://github.com/fra31/auto-attack/blob/f551404cf7ed1d6e17d3041d5709ad23f6dbf352/autoattack/autoattack.py#L74

    Example of quickly creating auto-attack
    ```
    model = ...  # which outputs logits
    AutoBestAttack.create_auto_attack(predict, ...)
    ```
    """
    def __init__(self, predict, base_adversaries, loss_fn=None,
                 targeted=None):
        self.predict = predict
        self.base_adversaries = base_adversaries
        self.loss_fn = loss_fn
        self.targeted = targeted

        if self.loss_fn is None:
            self.loss_fn = NegAccLoss(reduction='none')
        else:
            assert self.loss_fn.reduction == "none"

        if targeted is not None:
            for adversary in self.base_adversaries:
                assert self.targeted == adversary.targeted

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)

        with torch.no_grad():
            maxloss = self.loss_fn(self.predict(x), y)
        final_adv_x = x.clone().detach()  # torch.zeros_like(x)
        for ia, adversary in enumerate(self.base_adversaries):
            # print(f"### attack with {ia+1}/{len(self.base_adversaries)} adv: {adversary.__class__.__name__}")

            # only attack those failed (correctly predicted)
            mask = torch.nonzero(maxloss < 1, as_tuple=True)[0]
            x_to_att = x[mask]
            y_to_att = y[mask]
            if len(x_to_att) <= 0:
                break

            adv_x = adversary.perturb(x_to_att, y_to_att)
            loss = self.loss_fn(self.predict(adv_x), y_to_att)
            to_use = maxloss[mask] < loss
            to_replace = mask[to_use]
            # mask[to_replace] = True
            final_adv_x[to_replace] = adv_x[to_use]
            maxloss[to_replace] = loss[to_use]

            # n_fooled = len(torch.nonzero(maxloss > 0, as_tuple=True)[0])
            # print(f"###   {torch.sum(to_use.int()).item()} new samples are fooled.")
            # print(f"###   {n_fooled}/{len(maxloss)} samples are fooled.")

        return final_adv_x

    @classmethod
    def create_auto_attack(cls, predict, epsilon, norm, version='standard', loss='acc'):
        if version == 'standard':
            names = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
        elif version == 'plus':
            names = ['apgd-ce', 'apgd-dlr', 'fab', 'square', 'apgd-t', 'fab-t']
        elif version == 'rand':
            names = ['apgd-ce', 'apgd-dlr']

        adversaries = [create_attacks(name, predict, epsilon, norm, version=version)
                       for name in names]
        if loss == 'acc':
            loss_fn = NegAccLoss(reduction='none')
        # elif loss == 'ce':  # FIXME we cannot use this since we assume the loss is 0 or 1 in perturb
        #     loss_fn = CrossEntropyLoss(reduction='none')
        else:
            raise ValueError(f"Invalid loss: {loss}")
        return cls(predict, adversaries, loss_fn=loss_fn)


class NegAccLoss(_Loss):
    """Negative Accuracy Loss
    0 means correct pred, 1 means wrong. not reduced.
    """

    def __init__(self, size_average=None, reduce=None,
                 reduction='mean'):
        super(NegAccLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, logits, target):
        acc_loss = 1. - target.eq(logits.max(1)[1]).float()  # .to(device)
        if self.reduction == 'none':
            pass
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")
        return acc_loss


def create_attacks(name, predict, epsilon, norm, seed=42, device=None, version='standard'):
    """Easy to create attack by name with preset parameters."""
    if name.startswith('apgd'): # == 'apgd-ce':
        from autoattack.autopgd_base import APGDAttack
        adv = APGDAttack(predict, n_restarts=5, n_iter=100, verbose=False,
            eps=epsilon, norm=norm, eot_iter=1, rho=.75, seed=seed, device=device)
        if name == 'apgd-ce':
            adv.loss = 'ce'
        elif name == 'apgd-dlr':
            adv.loss = 'dlr'
        elif name == 'apgd-t':
            adv.loss = 'ce'
            adv.targeted = True
        else:
            raise ValueError(f"Invalid adv name: {name}")

        if version == 'standard':
            if name == 'apgd-ce':
                if norm in ['Linf', 'L2']:
                    adv.n_restarts = 1
                elif norm in ['L1']:
                    adv.use_largereps = True
                    adv.n_restarts = 5
            elif name == 'apgd-t':
                adv.n_restarts = 1
                if norm in ['Linf', 'L2']:
                    adv.n_target_classes = 9
                elif norm in ['L1']:
                    adv.use_largereps = True
                    adv.n_target_classes = 5
        elif version == 'plus':
            if name == 'apgd-ce':
                adv.n_restarts = 5
            elif name == 'apgd-t':
                adv.n_restarts = 1
                adv.n_target_classes = 9
            if not norm in ['Linf', 'L2']:
                print('"{}" version is used with {} norm: please check'.format(
                    version, norm))
        elif version == 'rand':
            adv.n_restarts = 1
            adv.eot_iter = 20
    elif name.startswith('fab'):
        from autoattack.fab_pt import FABAttack_PT
        adv = FABAttack_PT(predict, n_restarts=5, n_iter=100, eps=epsilon, seed=seed,
            norm=norm, verbose=False, device=device)
        if name == 'fab-t':
            adv.targeted = True
        elif name == 'fab':
            adv.targeted = False
        else:
            raise ValueError(f"Invalid adv name: {name}")

        if version == 'standard':
            adv.n_restarts = 1
            adv.n_target_classes = 9
        elif version == 'plus':
            adv.n_restarts = 5
            adv.n_target_classes = 9
    elif name == 'square':
        from autoattack.square import SquareAttack
        adv = SquareAttack(predict, p_init=.8, n_queries=5000, eps=epsilon, norm=norm,
            n_restarts=1, seed=seed, verbose=False, device=device, resc_schedule=False)
        adv.n_queries = 5000

        if version == 'standard':
            adv.n_queries = 5000
        elif version == 'plus':
            adv.n_queries = 5000
    else:
        raise ValueError(f"Invalid name: {name}")
    return adv
