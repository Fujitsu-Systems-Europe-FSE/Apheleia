from apheleia.utils.logger import ProjectLogger

import torch


class ModelStore(dict):
    def __init__(self, opts):
        super().__init__()
        self._opts = opts

    def eval(self):
        if hasattr(self._opts, 'intel') and self._opts.intel:
            import intel_extension_for_pytorch as ipex

        # Set networks in inference mode
        for k in self.keys():
            self[k] = self[k].eval()
            if hasattr(self._opts, 'intel') and self._opts.intel:
                self[k] = ipex.optimize(self[k], dtype=torch.bfloat16 if self._opts.fp16 else None)

        ProjectLogger().info(f'{list(self.keys())} set to eval mode.')

    def train(self):
        # Set networks in training mode
        for k in self.keys():
            self[k] = self[k].train()

        ProjectLogger().info(f'{list(self.keys())} set to train mode.')

    def get_raw(self, k):
        net = self.get(k)
        if hasattr(net, 'module'):
            net = net.module
        return net

    def get_all_raw(self):
        nets = []
        for k in self.keys():
            net = self.get(k)
            if hasattr(net, 'module'):
                net = net.module
            nets.append(net)
        return nets
