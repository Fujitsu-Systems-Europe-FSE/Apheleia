import math
import torch

from functools import partial
from torch import nn
from torch.nn import init
from torch.nn.modules.conv import _ConvNd
from bibad.model.utils.act_utils import LRELU_SLOPE, PRELU_INIT_SLOPE


def layer_init(layer, fn, params):
    fn(layer.weight.data, **params)
    if layer.bias is not None:
        init.zeros_(layer.bias.data)


def weights_init(m, conv=(init.kaiming_normal_, {}), linear=(init.kaiming_uniform_, {})):
    if isinstance(m, _ConvNd) or isinstance(m, nn.Linear):
        # for spectral normed layers
        if 'Parametrized' in m.__class__.__name__:
            layer_init(m, init.orthogonal_, {})
        else:
            fn, params = conv if isinstance(m, _ConvNd) else linear
            layer_init(m, fn, params)


def ortho_reg(model, strength=1e-4, blacklist=[]):
    with torch.no_grad():
        for param in model.parameters():
            # Only apply this to parameters with at least 2 axes, and not in the blacklist
            if len(param.shape) < 2 or any([param is item for item in blacklist]):
                continue

            if param.grad is None:
                continue

            w = param.view(param.shape[0], -1)
            grad = (2 * torch.mm(torch.mm(w, w.t()) * (1. - torch.eye(w.shape[0], device=w.device)), w))
            param.grad.data += strength * grad.view(param.shape)


def get_initializer(act):
    conv, linear = None, None
    if act == 'linear':
        conv = (init.xavier_normal_, dict(gain=1))
        linear = (init.xavier_uniform_, dict(gain=1))
    elif act == 'relu':
        conv = (init.kaiming_normal_, dict(nonlinearity=act, mode='fan_in'))
        linear = (init.kaiming_uniform_, dict(nonlinearity=act, mode='fan_in'))
    elif act == 'leaky_relu':
        conv = (init.kaiming_normal_, dict(nonlinearity=act, a=LRELU_SLOPE, mode='fan_in'))
        linear = (init.kaiming_uniform_, dict(nonlinearity=act, a=LRELU_SLOPE, mode='fan_in'))
    elif act == 'prelu':
        conv = (init.kaiming_normal_, dict(nonlinearity='leaky_relu', a=PRELU_INIT_SLOPE, mode='fan_in'))
        linear = (init.kaiming_uniform_, dict(nonlinearity='leaky_relu', a=PRELU_INIT_SLOPE, mode='fan_in'))
    elif act == 'selu':
        conv = (init.xavier_normal_, dict(gain=math.sqrt(1 / 2)))
        linear = (init.xavier_uniform_, dict(gain=math.sqrt(1 / 2)))
    elif act == 'sigmoid':
        conv = (init.xavier_normal_, dict(gain=init.calculate_gain(act)))
        linear = (init.xavier_uniform_, dict(gain=init.calculate_gain(act)))
    elif act == 'tanh':
        conv = (init.xavier_normal_, dict(gain=init.calculate_gain(act)))
        linear = (init.xavier_uniform_, dict(gain=init.calculate_gain(act)))
    else:
        raise NotImplementedError

    return partial(weights_init, conv=conv, linear=linear)
