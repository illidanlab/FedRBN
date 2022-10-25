import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import IMG_EXTENSIONS, has_file_allowed_extension
from PIL import Image
import os
from typing import Tuple, List, Dict, Optional, Callable, cast
from .config import data_root, DATA_PATHS


class DigitsDataset(Dataset):
    all_domains = ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST_M']
    resorted_domains = {
        0: ['MNIST',    'SVHN', 'USPS', 'SynthDigits', 'MNIST_M'],
        1: ['SVHN',     'USPS', 'SynthDigits', 'MNIST_M', 'MNIST'],
        2: ['USPS',     'SynthDigits', 'MNIST_M', 'MNIST', 'SVHN'],
        3: ['SynthDigits', 'MNIST_M', 'MNIST', 'SVHN', 'USPS'],
        4: ['MNIST_M',  'MNIST', 'SVHN', 'USPS', 'SynthDigits'],
    }

    def __init__(self, domain, channels, percent=0.1, filename=None, train=True, transform=None):
        data_path = DATA_PATHS['Digits'] + '/' + domain
        if filename is None:
            if train:
                if percent >= 0.1:
                    for part in range(int(percent*10)):
                        if part == 0:
                            self.images, self.labels = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                        else:
                            images, labels = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                            self.images = np.concatenate([self.images,images], axis=0)
                            self.labels = np.concatenate([self.labels,labels], axis=0)
                else:
                    self.images, self.labels = np.load(os.path.join(data_path, 'partitions/train_part0.pkl'), allow_pickle=True)
                    data_len = int(self.images.shape[0] * percent*10)
                    self.images = self.images[:data_len]
                    self.labels = self.labels[:data_len]
            else:
                self.images, self.labels = np.load(os.path.join(data_path, 'test.pkl'), allow_pickle=True)
        else:
            self.images, self.labels = np.load(os.path.join(data_path, filename), allow_pickle=True)

        self.transform = transform
        self.channels = channels
        self.labels = self.labels.astype(np.long).squeeze()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.channels == 1:
            image = Image.fromarray(image, mode='L')
        elif self.channels == 3:
            image = Image.fromarray(image, mode='RGB')
        else:
            raise ValueError("{} channel is not allowed.".format(self.channels))

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# PRESET_DISTORTION_SETS = {
#     "preset0": ['Gaussian Noise', 'Speckle Noise', 'Pixelate', 'Defocus Blur', 'Contrast'],
#     "preset1": ['Shot Noise', 'Impulse Noise', 'Glass Blur', 'Zoom Blur', 'Frost'],
#     "preset2": ['Fog', 'Brightness', 'Elastic', 'Speckle Noise', 'Spatter'],
#     "preset3": ['Saturate', 'JPEG', 'Defocus Blur', 'Frost', 'Shot Noise'],
# }


class OfficeDataset(Dataset):
    all_domains = ['amazon', 'caltech', 'dslr', 'webcam']
    resorted_domains = {
        0: ['amazon', 'caltech', 'dslr', 'webcam'],
        1: ['caltech', 'dslr', 'webcam', 'amazon'],
        2: ['dslr', 'webcam', 'amazon', 'caltech'],
        3: ['webcam', 'amazon', 'caltech', 'dslr'],
    }
    def __init__(self, base_path, site, train=True, transform=None, noise='none', noise_severity=3, noise_ratio=1.):
        self.rng = np.random.RandomState(42)
        if train:
            self.paths, self.text_labels = np.load('../data/office_caltech_10/{}_train.pkl'.format(site), allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load('../data/office_caltech_10/{}_test.pkl'.format(site), allow_pickle=True)
            
        label_dict={'back_pack':0, 'bike':1, 'calculator':2, 'headphones':3, 'keyboard':4, 'laptop_computer':5, 'monitor':6, 'mouse':7, 'mug':8, 'projector':9}
        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        # self.base_path = base_path if base_path is not None else '../data/office_caltech_10'
        # if noise != 'none':
        self.base_path = '/home/jyhong/projects/fed_dmtl/data/office31'
        self.noise = noise
        self.noise_severity = noise_severity
        self.noise_ratio = noise_ratio
        self.noise_flag = np.zeros(len(self.paths)) == 0  # noise all
        if self.noise_ratio < 1.:
            n = len(self.noise_flag)
            self.noise_flag[:] = False
            self.noise_flag[self.rng.choice(n, int(n*self.noise_ratio))] = True

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        head, site, cls, fname = self.paths[idx].split('/')
        if self.noise == 'none' or not self.noise_flag[idx]:
            img_path = os.path.join(self.base_path, site, 'images', cls, fname)
        else:
            img_path = os.path.join(self.base_path, site, 'distorted_images', self.noise,
                                    f'severity_{self.noise_severity}', cls, fname)
        label = self.labels[idx]
        noise_flag = self.noise_flag[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class DomainNetDataset(Dataset):
    all_domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    resorted_domains = {
        0: ['real',      'clipart',   'infograph', 'painting',  'quickdraw', 'sketch'],
        1: ['clipart',   'infograph', 'painting',  'quickdraw', 'sketch',    'real'],
        2: ['infograph', 'painting',  'quickdraw', 'sketch',    'real',      'clipart'],
        3: ['painting',  'quickdraw', 'sketch',    'real',      'clipart',   'infograph'],
        4: ['quickdraw', 'sketch',    'real',      'clipart',   'infograph', 'painting'],
        5: ['sketch',    'real',      'clipart',   'infograph', 'painting',  'quickdraw'],
    }

    def __init__(self, base_path, site, train=True, transform=None, noise='none', noise_severity=3,
                 full_set=False):
        noise = noise.replace('_', ' ')
        self.full_set = full_set
        self.base_path = DATA_PATHS['DomainNet']
        if full_set:
            classes, class_to_idx = find_classes(f"{self.base_path}/{site}")
            self.text_labels = classes
            self.paths, self.labels = make_dataset_from_dir(f"{self.base_path}/{site}", class_to_idx, IMG_EXTENSIONS)
            self.num_classes = len(class_to_idx)
        else:
            self.paths, self.text_labels = np.load(DATA_PATHS['DomainNetPathList'] + 'DomainNet/{}_{}.pkl'.format(
                site, 'train' if train else 'test'), allow_pickle=True)
            
            class_to_idx = {'bird':0, 'feather':1, 'headphones':2, 'ice_cream':3, 'teapot':4, 'tiger':5, 'whale':6, 'windmill':7, 'wine_glass':8, 'zebra':9}
        
            self.labels = [class_to_idx[text] for text in self.text_labels]
            self.num_classes = len(class_to_idx)
        self.transform = transform
        self.noise = noise
        self.noise_severity = noise_severity

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        site, cls, fname = self.paths[idx].split('/')[-3:]
        if self.noise == 'none':
            img_path = os.path.join(self.base_path, site, cls, fname)
        else:
            img_path = os.path.join(self.base_path, 'distorted_' + site, self.noise,
                                    f'severity_{self.noise_severity}', cls, fname)

        label = self.labels[idx]
        image = Image.open(img_path)
        
        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

# //////////// Data processing ////////////


class Partitioner:
    """Partition a sequence into shares."""
    def __init__(self, rng=None, partition_mode="dir",
                 max_n_sample_per_share=-1,
                 min_n_sample_per_share=2,
                 max_n_sample=-1
                 ):
        assert max_n_sample_per_share < 0 or max_n_sample_per_share > min_n_sample_per_share, \
            f"max ({max_n_sample_per_share}) > min ({min_n_sample_per_share})"
        self.rng = rng if rng else np.random
        self.partition_mode = partition_mode
        self.max_n_sample_per_share = max_n_sample_per_share
        self.min_n_sample_per_share = min_n_sample_per_share
        self.max_n_sample = max_n_sample

    def __call__(self, n_sample, n_share):
        """Partition a sequence of `n_sample` into `n_share` shares.
        Returns:
            partition: A list of num of samples for each share.
        """
        print(f"{n_sample} samples => {n_share} shards by {self.partition_mode} distribution.")
        if self.max_n_sample > 0:
            n_sample = min((n_sample, self.max_n_sample))
        if self.max_n_sample_per_share > 0:
            n_sample = min((n_sample, n_share * self.max_n_sample_per_share))

        if n_sample < self.min_n_sample_per_share * n_share:
            raise ValueError(f"Not enough samples. Require {self.min_n_sample_per_share} samples"
                             f" per share at least for {n_share} shares. But only {n_sample} is"
                             f" available.")
        n_sample -= self.min_n_sample_per_share * n_share
        if self.partition_mode == "dir":
            partition = (self.rng.dirichlet(n_share * [1]) * n_sample).astype(int)
        elif self.partition_mode == "uni":
            partition = int(n_sample // n_share) * np.ones(n_share, dtype='int')
        else:
            raise ValueError(f"Invalid partition_mode: {self.partition_mode}")

        partition[-1] += n_sample - np.sum(partition)  # add residual
        assert sum(partition) == n_sample, f"{sum(partition)} != {n_sample}"
        partition = partition + self.min_n_sample_per_share
        n_sample += self.min_n_sample_per_share * n_share
        # partition = np.minimum(partition, max_n_sample_per_share)
        partition = partition.tolist()

        assert sum(partition) == n_sample, f"{sum(partition)} != {n_sample}"
        assert len(partition) == n_share, f"{len(partition)} != {n_share}"
        return partition


def find_classes(dir: str) -> Tuple[List[str], Dict[str, int]]:
    """
    Finds the class folders in a dataset.

    Args:
        dir (string): Root directory path.

    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

    Ensures:
        No class is a subdirectory of another.
    """
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def make_dataset_from_dir(
    directory: str,
    class_to_idx: Dict[str, int],
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> Tuple[List[str], List[int]]:
    """Different Pytorch version, we return path and labels in two lists."""
    paths, labels = [], []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    paths.append(path)
                    labels.append(class_index)
                    # instances.append(item)
    return paths, labels

