"""Core functions of federate learning."""
import argparse
import numpy as np
from advertorch.attacks import LinfPGDAttack
from torch import nn

from federated.aggregation import ModelAccumulator


class Federation:
    """A helper class for federated data creation.
    Use `add_argument` to setup ArgumentParser and then use parsed args to init the class.
    """
    _model_accum: ModelAccumulator

    @classmethod
    def add_argument(cls, parser: argparse.ArgumentParser):
        # data
        parser.add_argument('--percent', type=float, default=0.3,
                            help='percentage of dataset for training')
        parser.add_argument('--val_ratio', type=float, default=0.5,
                            help='ratio of train set for validation')  # NOTE this replaced the original test_val_ratio
        parser.add_argument('--batch', type=int, default=32, help='batch size')
        parser.add_argument('--test_batch', type=int, default=128, help='batch size for test')

        # federated data split
        parser.add_argument('--pd_nuser', type=int, default=10, help='#users per domain.')
        parser.add_argument('--pr_nuser', type=int, default=-1, help='#users per comm round '
                                                                     '[default: all]')
        # parser.add_argument('--pu_nclass', type=int, default=-1, help='#class per user. -1 or 0: all')
        parser.add_argument('--domain_order', choices=list(range(5)), type=int, default=0,
                            help='select the order of domains')
        parser.add_argument('--partition_mode', choices=['uni', 'dir'], type=str.lower, default='uni',
                            help='the mode when splitting domain data into users: uni - uniform '
                                 'distribution (all user have the same #samples); dir - Dirichlet'
                                 ' distribution (non-iid sample sizes)')
        # parser.add_argument('--con_test_cls', action='store_true',
        #                     help='Ensure the test classes are the same training for a client. '
        #                          'Meanwhile, make test sets are uniformly splitted for clients. '
        #                          'Mainly influence class-niid settings.')

        # adv noise
        parser.add_argument('--noise_ratio', type=float, default=1., help='per-domain noise ration')
        parser.add_argument('--n_noise_domain', type=int, default=3,  # -1,
                            help='num of noised domains, -1 for all, otherwise in [0, max]')
        parser.add_argument('--noise', type=str, default='LinfPGD', help='noise the training data')
        parser.add_argument('--adv_lmbd', type=float, default=0.5,
                            help='adversarial lambda, 0 means clean, 1 means all noise')

        parser.add_argument('--svr_momentum', type=float, default=0.,
                            help='momentum for server update')

    @classmethod
    def render_run_name(cls, args):
        if args.pr_nuser > 1 or args.pr_nuser < 0:
            args.pr_nuser = int(args.pr_nuser)

        run_name = f'__pd_nuser_{args.pd_nuser}'
        if args.percent != 0.3: run_name += f'__pct_{args.percent}'
        # if args.pu_nclass > 0: run_name += f"__pu_nclass_{args.pu_nclass}"
        if args.pr_nuser != -1: run_name += f'__pr_nuser_{args.pr_nuser}'
        if args.domain_order != 0: run_name += f'__do_{args.domain_order}'
        if args.partition_mode != 'uni': run_name += f'__ptmd_{args.partition_mode}'
        # if args.con_test_cls: run_name += '__ctc'

        if args.noise_ratio < 1.: run_name += f'__noise_r{args.noise_ratio}'
        if args.n_noise_domain > 0: run_name += f'__noise_n{args.n_noise_domain}'
        if args.noise != 'none': run_name += f'__noise_{args.noise}'

        if args.adv_lmbd != 0.5: run_name += f'__adv_lmbd_{args.adv_lmbd}'
        if args.svr_momentum > 0.: run_name += f'__svrm_{args.svr_momentum}'
        return run_name

    def __init__(self, data, args):
        self.args = args

        # Prepare Data
        num_classes = 10
        if data == 'Digits':
            from utils.data_utils import DigitsDataset
            from .data import prepare_digits_data
            prepare_data = prepare_digits_data
            DataClass = DigitsDataset
        elif data == 'DomainNet':
            from utils.data_utils import DomainNetDataset
            from .data import prepare_domainnet_data
            prepare_data = prepare_domainnet_data
            DataClass = DomainNetDataset
        elif data == 'Office':
            from utils.data_utils import OfficeDataset
            from .data import prepare_office_data
            prepare_data = prepare_office_data
            DataClass = OfficeDataset
        else:
            raise ValueError(f"Unknown dataset: {data}")
        all_domains = DataClass.resorted_domains[args.domain_order]

        if args.n_noise_domain == -1:
            args.n_noise_domain = len(all_domains)

        # TODO this args are based on Digits. Need to update for others.
        train_loaders, val_loaders, test_loaders, clients = prepare_data(
            args, domains=all_domains,
            shuffle_eval=False,
            n_user_per_domain=args.pd_nuser,
            partition_seed=args.seed + 1,
            partition_mode=args.partition_mode,
            val_ratio=args.val_ratio,
            eq_domain_train_size=args.partition_mode == 'uni',
            shuffle_train=not args.test,
            # TODO class niid
            # n_class_per_user=args.pu_nclass,
            # consistent_test_class=args.con_test_cls,
        )

        n_clean = len(all_domains) - args.n_noise_domain
        assert n_clean >= 0, f"Invalid n_noised: {args.n_noise_domain}"
        if args.noise_ratio < 1.:
            n_noised_per_domain = int(args.pd_nuser * args.noise_ratio)
            mask = []
            for i in range(args.n_noise_domain):
                mask.extend(
                    [j + i*args.pd_nuser for j in range(n_noised_per_domain)])
            clients = [ds_name + ' ' + ('noised' if i in mask else 'clean')
                       for i, ds_name in enumerate(clients)]
        else:
            clients = [ds_name + ' ' + ('noised' if i < args.n_noise_domain * args.pd_nuser else 'clean')
                       for i, ds_name in enumerate(clients)]

        self.train_loaders = train_loaders
        self.val_loaders = val_loaders
        self.test_loaders = test_loaders
        self.clients = clients
        self.num_classes = num_classes
        self.all_domains = all_domains

        # Setup fed
        self.client_num = len(self.clients)
        client_weights = [len(tl.dataset) for tl in train_loaders]
        self.client_weights = [w / sum(client_weights) for w in client_weights]

        pr_nuser = args.pr_nuser if args.pr_nuser > 0 else self.client_num
        self.args.pr_nuser = pr_nuser
        self.client_sampler = UserSampler([i for i in range(self.client_num)], pr_nuser, mode='uni')

    def get_data(self):
        return self.train_loaders, self.val_loaders, self.test_loaders

    def make_aggregator(self, running_model):
        self._model_accum = ModelAccumulator(
            running_model, self.args.pr_nuser, self.client_num,
            local_bn=self.args.mode.lower() in ['fedbn', 'fedrbn'],
            raise_err_on_early_accum=self.args.AT_iters < 0)
        return self._model_accum

    @property
    def model_accum(self):
        if not hasattr(self, '_model_accum'):
            raise RuntimeError(f"model_accum has not been set yet. Call `make_aggregator` first.")
        return self._model_accum

    def download(self, running_model, client_idx):
        """Download (personalized) global model to running_model."""
        self.model_accum.load_model(running_model, client_idx)  # , strict=strict)

    def upload(self, running_model, client_idx):
        """Upload client model."""
        self.model_accum.add(client_idx, running_model, self.client_weights[client_idx])

    def aggregate(self):
        """Aggregate received models and update global model."""
        self.model_accum.update_server_and_reset(beta=self.args.svr_momentum)


class AdversaryCreator(object):
    """A factory producing adversary.

    Args:
        attack: Name. MIA for MomentumIterativeAttack with Linf norm. LSA for LocalSearchAttack.
        eps: Constraint on the distortion norm
        steps: Number of attack steps
    """
    supported_adv = ['LinfPGD', 'jointLinfPGD', 'LinfPGD20', 'LinfPGD20_eps16', 'LinfPGD100', 'jointLinfPGD100', 'LinfPGD100_eps16',
                     'LinfPGD4_eps4', 'LinfPGD3_eps4', 'LinfPGD7_eps4',  # combined for LinfPGD7_eps8
                     'MIA', 'MIA20', 'MIA20_eps16', 'MIA100', 'MIA100_eps16', 'Free', 'LSA',
                     'LinfAA', 'LinfAA+',
                     'TrnLinfPGD',  # transfer attack
                     ]

    def __init__(self, attack: str, joint_noise_detector=None, **kwargs):
        self.attack = attack
        self.joint_noise_detector = joint_noise_detector
        # eps = 8., steps = 7
        if attack == 'Free':  # only for training.
            self.eps = kwargs.setdefault('eps', 4.)
            self.steps = kwargs.setdefault('steps', 4)
        elif attack == 'Affine':  # only for training.
            self.eps = kwargs.setdefault('eps', 0.1)  # = 1/lambda. Need to tune
            self.steps = kwargs.setdefault('steps', 2)
        else:
            if '_eps' in self.attack:
                self.attack, default_eps = self.attack.split('_eps')
                self.eps = kwargs.setdefault('eps', int(default_eps))
            else:
                self.eps = kwargs.setdefault('eps', 8.)
            if self.attack.startswith('LinfPGD') and self.attack[len('LinfPGD'):].isdigit():
                assert 'steps' not in kwargs, "The steps is set by the attack name while " \
                                              "found additional set in kwargs."
                self.steps = int(self.attack[len('LinfPGD'):])
            elif self.attack.startswith('MIA') and self.attack[len('MIA'):].isdigit():
                assert 'steps' not in kwargs, "The steps is set by the attack name while " \
                                              "found additional set in kwargs."
                self.steps = int(self.attack[len('MIA'):])
            else:
                self.steps = kwargs.setdefault('steps', 7)

    def __call__(self, model):
        if self.joint_noise_detector is not None:
            from torch.nn.modules.loss import _WeightedLoss
            from torch.nn import functional as F

            class JointCrossEntropyLoss(_WeightedLoss):
                __constants__ = ['ignore_index', 'reduction']
                ignore_index: int

                def __init__(self, weight = None, size_average=None, ignore_index: int = -100,
                             reduce=None, reduction: str = 'mean', alpha=0.5) -> None:
                    super(JointCrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
                    self.ignore_index = ignore_index
                    self.alpha = alpha

                def forward(self, input, target):
                    model_pred, dct_pred = input[:, :-2], input[:, -2:]
                    model_trg, dct_trg = target[:, 0], target[:, 1]
                    return (1 - self.alpha) * F.cross_entropy(model_pred, model_trg, weight=self.weight,
                                           ignore_index=self.ignore_index, reduction=self.reduction) \
                           + self.alpha * F.cross_entropy(dct_pred, dct_trg, weight=self.weight,
                                             ignore_index=self.ignore_index, reduction=self.reduction)

            loss_fn = JointCrossEntropyLoss(reduction="sum", alpha=self.joint_noise_detector)
        else:
            loss_fn = nn.CrossEntropyLoss(reduction="sum")
        if self.attack.startswith('LinfPGD'):
            adv = LinfPGDAttack(
                model, loss_fn=loss_fn, eps=self.eps / 255,
                nb_iter=self.steps, eps_iter=min(self.eps / 255 * 1.25, self.eps / 255 + 4. / 255) / self.steps, rand_init=True,
                clip_min=0.0, clip_max=1.0,
                targeted=False)
        elif self.attack == 'none':
            adv = None
        elif self.attack == 'Free':  # NOTE only use in train
            from utils.attack import FreeAttack
            adv = FreeAttack(
                model, loss_fn, model.input_shape, 'cuda',
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
                clip_min=0.0, clip_max=1.0,
                eps=self.eps / 255, nb_iter=self.steps, fgsm_step_size=4. / 255,
            )
        elif self.attack == "Affine":  # NOTE only use in train
            from utils.attack import AffineAttack
            adv = AffineAttack(
                model, loss_fn,
                nb_iter=self.steps, eps=self.eps,
            )
        elif self.attack.startswith("MIA"):
            from advertorch.attacks import MomentumIterativeAttack
            adv = MomentumIterativeAttack(
                model, loss_fn=loss_fn, eps=self.eps / 255,
                nb_iter=self.steps,
                eps_iter=min(self.eps / 255 * 1.25, self.eps / 255 + 4. / 255) / self.steps,
                clip_min=0.0, clip_max=1.0,
                targeted=False
            )
        elif self.attack == "LSA":
            from advertorch.attacks import LocalSearchAttack
            adv = LocalSearchAttack(
                model, loss_fn=loss_fn,
                round_ub=self.steps,  # FIXME how to set self.eps??
                clip_min=0.0, clip_max=1.0,
                targeted=False
            )
        elif self.attack.startswith('LinfAA'):
            from utils.autoattack import AutoAttack
            version = 'standard'
            if self.attack == 'LinfAA+':
                version = 'plus'
            adv = AutoAttack.create_auto_attack(
                model, epsilon=self.eps / 255, norm='Linf', version=version
            )
        else:
            raise ValueError(f"attack: {self.attack}")
        return adv


class UserSampler(object):
    def __init__(self, users, select_nuser, mode='all'):
        self.users = users
        self.total_num_user = len(users)
        self.select_nuser = select_nuser
        self.mode = mode
        if mode == 'all':
            assert select_nuser == self.total_num_user, "Conflict config: Select too few users."

    def iter(self):
        if self.mode == 'all' or self.select_nuser == self.total_num_user:
            sel = np.arange(len(self.users))
        elif self.mode == 'uni':
            sel = np.random.choice(self.total_num_user, self.select_nuser, replace=False)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        for i in sel:
            yield self.users[i]
