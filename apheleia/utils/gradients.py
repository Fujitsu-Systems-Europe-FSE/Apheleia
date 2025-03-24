from torch import autograd

import torch

def calc_jacobian_norm(output, inputs):
    gradients = autograd.grad(output.sum(), inputs, create_graph=True, retain_graph=True)
    gradients = torch.cat([g.flatten(start_dim=1) for g in gradients], dim=0).detach()
    return gradients.norm(2, dim=1).cpu().numpy()

def calc_net_gradient_norm(network):
    gradients = [v.grad for k, v in network.named_parameters() if v.grad is not None]
    if len(gradients) > 0:
        norms = torch.stack([g.flatten().norm(2) for g in gradients]).detach()
        return norms.cpu().numpy()