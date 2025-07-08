import torch

def positional_encoding(x, L_embed=6):
    rets = [x]
    for i in range(L_embed):
        for fn in [torch.sin, torch.cos]:
            rets.append(fn(2 ** i * x))
    return torch.cat(rets, dim=-1)
