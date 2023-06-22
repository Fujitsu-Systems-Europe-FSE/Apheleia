from functools import partial
from torch.nn import LeakyReLU, ReLU, PReLU, SELU

LRELU_SLOPE = .2
PRELU_INIT_SLOPE = .25

act_types = {
    'relu': ReLU,
    'leaky_relu': partial(LeakyReLU, LRELU_SLOPE),
    'prelu': partial(PReLU, init=PRELU_INIT_SLOPE),
    'selu': SELU,
    # 'fullsort': GroupSort,
    # 'groupsort': GroupSort
}


def act_opt(act, out_channels):
    act_opt = None
    if act == 'fullsort':
        # expected option for fullsort is number of channels
        act_opt = out_channels
    elif act == 'groupsort':
        # expected option for groupsort is number of channels
        act_opt = 8
    elif act == 'leaky_relu':
        # expected option for leaky_relu is inplace
        act_opt = False
    return act_opt


def get_act(act, out_channels):
    # activation need to be rebuilt on the fly to be unique in tensorboard graphs
    option = act_opt(act, out_channels)
    if option is None:
        # To avoid messing with torchscript export
        return act_types[act]()
    return act_types[act](option)
