import torch
import torch.nn as nn
import torch.nn.functional as func
from collections import OrderedDict
from torch.nn.modules.batchnorm import _NormBase
from .dual_bn import DualNormLayer
from torchvision.models.resnet import Bottleneck, ResNet, model_urls, BasicBlock, conv1x1
from torch.hub import load_state_dict_from_url

from typing import List, Union
import logging


class BaseModule(nn.Module):
    def set_bn_mode(self, is_noised: Union[bool, torch.Tensor]):
        """Set BN mode to be noised or clean. This is only effective for 
        DualNormLayer."""
        def set_bn_eval_(m):
            if isinstance(m, DualNormLayer):
                if isinstance(is_noised, torch.Tensor):
                    m.clean_input = ~is_noised
                else:
                    m.clean_input = not is_noised
        self.apply(set_bn_eval_)

    def set_dbn_n_train(self, mode: bool):
        """Set BN_n mode to be train mode. This is only effective for
        DualNormLayer."""
        def set_bn_eval_(m):
            if isinstance(m, DualNormLayer):
                m.noise_bn.train(mode)
        self.apply(set_bn_eval_)

    # BN operations
    def reset_bn_running_stat(self):
        def reset_bn(m):
            if isinstance(m, _NormBase):
                m.reset_running_stats()
        self.apply(reset_bn)

    # forward
    def forward(self, x):
        z = self.encode(x)
        logits = self.decode_clf(z)
        return logits

    def encode(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        z = torch.flatten(x, 1)
        return z

    def decode_clf(self, z):
        logits = self.classifier(z)
        return logits

    # freeze layers
    def freeze_shallow_layers(self, n_layer=0, freeze=True):
        """Freeze the first n layers (closest to the input)"""
        raise NotImplementedError()

    def _freeze_layers(self, blocks, n_layer, freeze):
        assert n_layer <= len(blocks) and n_layer >= 0, f"Invalid n_layer={n_layer}. Should be in [0, {len(blocks)}]"
        for i_block in range(n_layer):
            block = blocks[i_block]
            for m in block:
                for p in m.parameters():
                    p.requires_grad = not freeze


class DigitModel(BaseModule):
    """
    Model for benchmark experiment on Digits. 
    """
    input_shape = [None, 3, 28, 28]

    def __init__(self, num_classes=10, bn_type='bn', track_running_stats=True, ex_depth=0, **kwargs):
        super(DigitModel, self).__init__()
        bn_class = get_bn_layer(bn_type)
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.bn1 = bn_class['2d'](64, track_running_stats=track_running_stats)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.bn2 = bn_class['2d'](64, track_running_stats=track_running_stats)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.bn3 = bn_class['2d'](128, track_running_stats=track_running_stats)
        if ex_depth > 0:
            self.inplanes = 128
            self.ex_layer1 = self._make_layer(BasicBlock, 128, ex_depth, bn_class['2d'])
        else:
            self.ex_layer1 = None
    
        self.fc1 = nn.Linear(6272, 2048)
        self.bn4 = bn_class['1d'](2048, track_running_stats=track_running_stats)
        self.fc2 = nn.Linear(2048, 512)
        self.bn5 = bn_class['1d'](512, track_running_stats=track_running_stats)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x, **kwargs):
        z = self.encode(x)
        return self.decode_clf(z, **kwargs)

    def encode(self, x):
        x = func.relu(self.bn1(self.conv1(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn2(self.conv2(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn3(self.conv3(x)))

        if self.ex_layer1 is not None:
            x = self.ex_layer1(x)

        x = x.view(x.shape[0], -1)
        return x

    def decode_clf(self, x, return_pen_fea=False):
        x = self.fc1(x)
        x = self.bn4(x)
        x = func.relu(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x = func.relu(x)

        logits = self.fc3(x)
        if return_pen_fea:
            return logits, x
        else:
            return logits

    def freeze_shallow_layers(self, n_layer=0, freeze=True):
        # layers
        blocks = [
            [self.conv1, self.bn1],
            [self.conv2, self.bn2],
            [self.conv3, self.bn3],
            [self.fc1, self.bn4],
            [self.fc2, self.bn5],
            [self.fc3],
        ]
        self._freeze_layers(blocks, n_layer, freeze)

    def _make_layer(self, block, planes, blocks, norm_layer, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(OrderedDict([
                ('conv', conv1x1(self.inplanes, planes * block.expansion, stride)),
                ('bn', norm_layer(planes * block.expansion)),
            ]))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

class AlexNet(BaseModule):
    """
    used for DomainNet and Office-Caltech10
    """
    input_shape = [None, 3, 256, 256]

    def load_state_dict(self, state_dict, strict: bool = True):
        legacy_keys = []
        for key in state_dict:
            if 'noise_disc' in key:
                legacy_keys.append(key)
        if len(legacy_keys) > 0:
            logging.debug(f"Found old version of AlexNet. Ignore {len(legacy_keys)} legacy keys: {legacy_keys}")
            for key in legacy_keys:
                state_dict.pop(key)
        return super().load_state_dict(state_dict, strict)

    def __init__(self, num_classes=10, track_running_stats=True, bn_type='bn', share_affine=True):
        super(AlexNet, self).__init__()
        bn_class = get_bn_layer(bn_type)
        # share_affine
        bn_kwargs = dict(
            track_running_stats=track_running_stats,
        )
        if bn_type.startswith('d'):  # dual BN
            bn_kwargs['share_affine'] = share_affine
        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
                ('bn1', bn_class['2d'](64, **bn_kwargs)),
                ('relu1', nn.ReLU(inplace=True)),
                ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
                ('bn2', bn_class['2d'](192, **bn_kwargs)),
                ('relu2', nn.ReLU(inplace=True)),
                ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
                ('bn3', bn_class['2d'](384, **bn_kwargs)),
                ('relu3', nn.ReLU(inplace=True)),

                ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
                ('bn4', bn_class['2d'](256, **bn_kwargs)),
                ('relu4', nn.ReLU(inplace=True)),

                ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ('bn5', bn_class['2d'](256, **bn_kwargs)),
                ('relu5', nn.ReLU(inplace=True)),
                ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
            ])
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(256 * 6 * 6, 4096)),
                ('bn6', bn_class['1d'](4096, track_running_stats=track_running_stats)),
                ('relu6', nn.ReLU(inplace=True)),

                ('fc2', nn.Linear(4096, 4096)),
                ('bn7', bn_class['1d'](4096, track_running_stats=track_running_stats)),
                ('relu7', nn.ReLU(inplace=True)),

                ('fc3', nn.Linear(4096, num_classes)),
            ])
        )

    def freeze_shallow_layers(self, n_layer=0, freeze=True):
        # layers
        blocks = [
            [self.features[name] for name in ['conv1', 'bn1', 'relu1', 'maxpool1']],
            [self.features[name] for name in ['conv2', 'bn2', 'relu2', 'maxpool2']],
            [self.features[name] for name in ['conv3', 'bn3', 'relu3']],
            [self.features[name] for name in ['conv4', 'bn4', 'relu4']],
            [self.features[name] for name in ['conv5', 'bn5', 'relu5', 'maxpool5']],
            [self.classifier[name] for name in ['fc1', 'bn6', 'relu6']],
            [self.classifier[name] for name in ['fc2', 'bn7', 'relu7']],
            [self.classifier[name] for name in ['fc3']],
        ]
        self._freeze_layers(blocks, n_layer, freeze)

# BN modules
class _MockBatchNorm(_NormBase):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_MockBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return func.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            torch.zeros_like(self.running_mean),
            torch.ones_like(self.running_var),
            self.weight, self.bias, False, exponential_average_factor, self.eps)

class MockBatchNorm1d(_MockBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))

class MockBatchNorm2d(_MockBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

def get_bn_layer(bn_type: str):
    if bn_type.startswith('d'):  # dual norm layer. Example: sbn, sbin, sin
        base_norm_class = get_bn_layer(bn_type[1:])
        bn_class = {
            '1d': lambda num_features, **kwargs: DualNormLayer(num_features, bn_class=base_norm_class['1d'], **kwargs),
            '2d': lambda num_features, **kwargs: DualNormLayer(num_features, bn_class=base_norm_class['2d'], **kwargs),
        }
    elif bn_type == 'bn':
        bn_class = {'1d': nn.BatchNorm1d, '2d': nn.BatchNorm2d}
    elif bn_type == 'none':
        bn_class = {'1d': MockBatchNorm1d,
                    '2d': MockBatchNorm2d}
    else:
        raise ValueError(f"Invalid bn_type: {bn_type}")
    return bn_class
