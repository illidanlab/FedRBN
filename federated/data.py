import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Subset

from utils.data_utils import DomainNetDataset, DigitsDataset, Partitioner, OfficeDataset

# noise_ratio=1., n_noised=1,

def prepare_domainnet_data(args, domains=['clipart', 'quickdraw'], shuffle_eval=False,
                           n_user_per_domain=1, partition_seed=42, partition_mode='uni',
                           val_ratio=0.1,  full_class_set=False, preprocess='fedbn',
                           percent=1.0, test_batch=128, 
                           eq_domain_train_size=True, shuffle_train=True):
    datasets = [f'{i}' for i in range(len(domains))]
    if preprocess.lower() == 'fedbn':
        transform_train = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30, 30)),
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
        ])
    elif preprocess.lower() == 'pytorch':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        raise TypeError(f"preprocess: {preprocess}")

    train_sets =[
        DomainNetDataset(None, domain, transform=transform_train, noise='none', full_set=full_class_set)
        for domain in domains
    ]
    test_sets = [
        DomainNetDataset(None, domain, transform=transform_test, train=False, noise='none',
                         full_set=full_class_set)
        for domain in domains
    ]

    print(f" train #cls: {[s.num_classes for s in train_sets]}")
    print(f" train size: {[len(s) for s in train_sets]}")
    print(f" test  size: {[len(s) for s in test_sets]}")

    assert eq_domain_train_size
    train_len = int(min([len(s) for s in train_sets]) * percent)
    num_classes = max([s.num_classes for s in train_sets])
    # assert all([len(s) == train_len for s in train_sets]), f"Should be equal length."

    if n_user_per_domain > 1:
        split = Partitioner(np.random.RandomState(partition_seed), min_n_sample_per_share=100,
                            partition_mode=partition_mode)
        splitted_datasets = []

        val_sets, sub_train_sets = [], []
        for dname, tr_set in zip(datasets, train_sets):
            _train_len_by_user = split(train_len, n_user_per_domain)
            print(f" train split size: {_train_len_by_user}")

            base_idx = 0
            for i_user, tl in zip(range(n_user_per_domain), _train_len_by_user):
                vl = int(val_ratio * tl)
                tl = tl - vl

                sub_train_sets.append(Subset(tr_set, list(range(base_idx, base_idx + tl))))
                base_idx += tl

                val_sets.append(Subset(tr_set, list(range(base_idx, base_idx + vl))))
                base_idx += vl

                splitted_datasets.append(f"{dname}-{i_user}")

        sub_test_sets = []
        for te_set in test_sets:
            _test_len_by_user = split(len(te_set), n_user_per_domain)

            base_idx = 0
            for tl in _test_len_by_user:
                sub_test_sets.append(Subset(te_set, list(range(base_idx, base_idx + tl))))
                base_idx += tl

        # rename
        train_sets = sub_train_sets
        test_sets = sub_test_sets
        datasets = splitted_datasets
    else:
        val_len = int(train_len * val_ratio)

        val_sets = [Subset(tr_set, list(range(train_len-val_len, train_len))) for tr_set in train_sets]
        train_sets = [Subset(tr_set, list(range(train_len-val_len))) for tr_set in train_sets]

    # NOTE num_workers > 2 is not good when users increase.
    train_loaders = [torch.utils.data.DataLoader(tr_set, batch_size=args.batch, shuffle=shuffle_train,
                                                 drop_last=partition_mode != 'uni', num_workers=2) for tr_set in train_sets]
    val_loaders = [torch.utils.data.DataLoader(va_set, batch_size=args.test_batch, shuffle=shuffle_eval,
                                               num_workers=2, pin_memory=True) for va_set in val_sets]
    test_loaders = [torch.utils.data.DataLoader(te_set, batch_size=args.test_batch, shuffle=shuffle_eval,
                                                num_workers=2, pin_memory=True) for te_set in test_sets]

    return train_loaders, val_loaders, test_loaders, datasets


def prepare_digits_data(args, domains=['MNIST', 'SVHN'], shuffle_eval=False,
                        n_user_per_domain=1, partition_seed=42, partition_mode='uni', val_ratio=0.2,
                        eq_domain_train_size=True, shuffle_train=True):
    datasets = [f'{i}' for i in range(len(domains))]

    # Prepare data
    trns = {
        'MNIST': transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'SVHN': transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'USPS': transforms.Compose([
            transforms.Resize([28,28]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'SynthDigits': transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'MNIST_M': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
    }

    train_sets = [DigitsDataset(domain,
                                channels=3 if domain in ['SVHN', 'SynthDigits', 'MNIST_M'] else 1,
                                percent=args.percent, train=True,  transform=trns[domain])
                  for domain in domains]
    test_sets = [DigitsDataset(domain,
                               channels=3 if domain in ['SVHN', 'SynthDigits', 'MNIST_M'] else 1,
                               train=False,  transform=trns[domain])
                 for domain in domains]

    print(f" train size: {[len(s) for s in train_sets]}")
    print(f" test  size: {[len(s) for s in test_sets]}")

    train_len = [len(s) for s in train_sets]
    if eq_domain_train_size:
        train_len = [min(train_len)] * len(train_sets)
        assert all([len(s) == train_len[0] for s in train_sets]), f"Should be equal length."

    if n_user_per_domain > 1:
        split = Partitioner(np.random.RandomState(partition_seed), min_n_sample_per_share=40,
                            partition_mode=partition_mode)
        splitted_datasets = []

        val_sets, sub_train_sets = [], []
        for i_dataset, (dname, tr_set) in enumerate(zip(datasets, train_sets)):
            _train_len_by_user = split(train_len[i_dataset], n_user_per_domain)
            print(f" train split size: {_train_len_by_user}")

            base_idx = 0
            for i_user, tl in zip(range(n_user_per_domain), _train_len_by_user):
                vl = int(val_ratio * tl)
                tl = tl - vl

                sub_train_sets.append(Subset(tr_set, list(range(base_idx, base_idx + tl))))
                base_idx += tl

                val_sets.append(Subset(tr_set, list(range(base_idx, base_idx + vl))))
                base_idx += vl

                splitted_datasets.append(f"{dname}-{i_user}")

        sub_test_sets = []
        for te_set in test_sets:
            _test_len_by_user = split(len(te_set), n_user_per_domain)

            base_idx = 0
            for tl in _test_len_by_user:
                sub_test_sets.append(Subset(te_set, list(range(base_idx, base_idx + tl))))
                base_idx += tl

        # rename
        train_sets = sub_train_sets
        test_sets = sub_test_sets
        datasets = splitted_datasets
    else:
        val_len = [int(tl * val_ratio) for tl in train_len]

        val_sets = [Subset(tr_set, list(range(train_len[i_dataset]-val_len[i_dataset], train_len[i_dataset])))
                    for i_dataset, tr_set in enumerate(train_sets)]
        train_sets = [Subset(tr_set, list(range(train_len[i_dataset]-val_len[i_dataset])))
                      for i_dataset, tr_set in enumerate(train_sets)]

    train_loaders = [torch.utils.data.DataLoader(tr_set, batch_size=args.batch, shuffle=shuffle_train,
                                                 drop_last=partition_mode != 'uni') for tr_set in train_sets]
    val_loaders = [torch.utils.data.DataLoader(va_set, batch_size=args.test_batch, shuffle=shuffle_eval,
                                               num_workers=2, pin_memory=True) for va_set in val_sets]
    test_loaders = [torch.utils.data.DataLoader(te_set, batch_size=args.test_batch, shuffle=shuffle_eval,
                                                num_workers=4, pin_memory=True) for te_set in test_sets]

    return train_loaders, val_loaders, test_loaders, datasets


def prepare_office_data(args, domains=['amazon', 'webcam'],
                        shuffle_eval=False, n_user_per_domain=1, partition_seed=42, partition_mode='uni', val_ratio=0.4):
    datasets = [f'{i}' for i in range(len(domains))]
    # TODO try to expand single domain setting to multiple users.
    single_domain = domains[0] == domains[1] and len(datasets) == 2
    transform_train = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
    ])

    train_sets =[
        OfficeDataset(None, domain, transform=transform_train, noise='none')
        for domain in domains
    ]
    test_sets = [
        OfficeDataset(None, domain, transform=transform_test, train=False, noise='none')
        for domain in domains
    ]

    if single_domain:
        raise NotImplementedError()

    print(f" train size: {[len(s) for s in train_sets]}")
    print(f" test  size: {[len(s) for s in test_sets]}")

    train_len = min([len(s) for s in train_sets])
    # assert all([len(s) == train_len for s in train_sets]), f"Should be equal length."

    if n_user_per_domain > 1:
        partition_rng = np.random.RandomState(partition_seed)
        split = Partitioner(partition_rng, min_n_sample_per_share=60,
                            partition_mode=partition_mode)
        splitted_datasets = []

        val_sets, sub_train_sets = [], []
        for dname, tr_set in zip(datasets, train_sets):
            _train_len_by_user = split(train_len, n_user_per_domain)
            print(f" train split size: {_train_len_by_user}")

            base_idx = 0
            for i_user, tl in zip(range(n_user_per_domain), _train_len_by_user):
                vl = int(val_ratio * tl)
                tl = tl - vl

                sub_train_sets.append(Subset(tr_set, list(range(base_idx, base_idx + tl))))
                base_idx += tl

                val_sets.append(Subset(tr_set, list(range(base_idx, base_idx + vl))))
                base_idx += vl

                splitted_datasets.append(f"{dname}-{i_user}")

        sub_test_sets = []
        split = Partitioner(partition_rng, min_n_sample_per_share=10,
                            partition_mode=partition_mode)
        for te_set in test_sets:
            _test_len_by_user = split(len(te_set), n_user_per_domain)

            base_idx = 0
            for tl in _test_len_by_user:
                sub_test_sets.append(Subset(te_set, list(range(base_idx, base_idx + tl))))
                base_idx += tl

        # rename
        train_sets = sub_train_sets
        test_sets = sub_test_sets
        datasets = splitted_datasets
    else:
        val_len = int(train_len * val_ratio)

        val_sets = [Subset(tr_set, list(range(train_len-val_len, train_len))) for tr_set in train_sets]
        train_sets = [Subset(tr_set, list(range(train_len-val_len))) for tr_set in train_sets]

    # NOTE num_workers > 2 is no good when users increase.
    train_loaders = [torch.utils.data.DataLoader(tr_set, batch_size=args.batch, shuffle=True, num_workers=2) for tr_set in train_sets]
    val_loaders = [torch.utils.data.DataLoader(va_set, batch_size=args.batch, shuffle=shuffle_eval, num_workers=2, drop_last=False) for va_set in val_sets]
    test_loaders = [torch.utils.data.DataLoader(te_set, batch_size=args.batch, shuffle=shuffle_eval, num_workers=2, drop_last=False) for te_set in test_sets]

    return train_loaders, val_loaders, test_loaders, datasets
