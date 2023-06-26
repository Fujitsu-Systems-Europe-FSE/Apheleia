import torch

def to_tensor(data, ctx=torch.device('cpu'), dtype=torch.float32):
    if type(data) != torch.Tensor:
        data = torch.tensor(data, dtype=dtype)
    return data.to(ctx)