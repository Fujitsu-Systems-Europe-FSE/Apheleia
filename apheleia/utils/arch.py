import math


def convolution(in_size, kernel, padding=0, stride=1, dilation=1):
    assert in_size >= kernel, 'Kernel size can\'t be greater than actual input size'
    out_size = ((in_size + 2 * padding - (dilation * (kernel - 1)) - 1) / stride) + 1
    return math.floor(out_size)


def deconvolution(in_size, kernel, padding=0, stride=1, dilation=1, out_padding=0):
    out_size = (in_size - 1) * stride - 2 * padding + dilation * (kernel - 1) + out_padding + 1
    return out_size


def trace_layer(in_size, layer_params, mode):
    out_padding = None
    if len(layer_params) == 4:
        kernel, stride, padding, dilation = layer_params
    else:
        kernel, stride, padding, dilation, out_padding = layer_params

    if padding == 'auto':
        padding = (kernel - 1) // 2

    assert mode in ['conv', 'deconv'], 'Invalid conv mode.'
    name = 'Conv1d' if mode == 'conv' else 'ConvTranpose1d'

    if out_padding is None:
        out_padding = 0

    out_size = convolution(in_size, kernel, padding, stride, dilation) if mode == 'conv' else deconvolution(in_size, kernel, padding, stride, dilation, out_padding)
    print(f'{in_size} --> {name} {kernel}x1, s={stride}, p={padding}, d={dilation}, op={out_padding} --> {out_size}')
    return out_size


def trace_net(name, net, input, mode):
    print('{}\n{}\n{}'.format(''.join(['-'] * len(name)), name, ''.join(['-'] * len(name))))
    for layer in net:
        input = trace_layer(input, layer, mode)
    print('')
