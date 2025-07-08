import torch

def get_rays(height, width, focal_length, cam2world):
    i, j = torch.meshgrid(
        torch.arange(width).to(cam2world),
        torch.arange(height).to(cam2world),
        indexing="xy"
    )
    dirs = torch.stack([
        (i.cpu() - width / 2) / focal_length,
        - (j.cpu() - height / 2) / focal_length,
        - torch.ones_like(i.cpu())
    ], dim=-1).to(cam2world)
    rays_d = torch.sum(dirs[..., None, :] * cam2world[:3, :3], dim=-1)
    rays_o = cam2world[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d
