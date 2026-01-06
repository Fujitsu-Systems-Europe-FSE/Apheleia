from torch import autograd

import torch

def calc_jacobian_norm(output, inputs):
    gradients = autograd.grad(output.sum(), inputs, create_graph=True, retain_graph=True)
    gradients = torch.cat([g.flatten(start_dim=1) for g in gradients], dim=0).detach()
    return gradients.norm(2, dim=1).cpu().numpy()

def calc_net_gradient_norm(network):
    gradients = [v.grad for k, v in network.named_parameters() if v.grad is not None]
    if len(gradients) > 0:
        with torch.no_grad():
            norms = torch.stack([g.norm(2) for g in gradients])
        return norms.norm(2)
    return None


def calc_net_gradient_norm_per_loss(loss, network):
    """
    Calcule ‖∇_θ loss‖₂ pour tous les paramètres entraînables du `model`.
    Ne modifie pas .grad, car on n’utilise pas .backward().
    """
    params = [p for p in network.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, retain_graph=True, create_graph=False, allow_unused=True)
    grads = [g for g in grads if g is not None]
    norms = torch.stack([g.norm(2) for g in grads])
    return torch.norm(norms, 2), norms