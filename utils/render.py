import torch
import torch.nn.functional as F

def batch_generator(inputs, batch_size):
    l = inputs.shape[0]
    for i in range(0, l, batch_size):
        yield inputs[i:min(i + batch_size, l)]

def render_rays(model, rays_o, rays_d, near, far, N_samples, encoding_fn, rand=True):
    z_vals = torch.linspace(near, far, N_samples).to(rays_o)
    if rand:
        z_vals = (
            torch.rand(list(rays_o.shape[:-1]) + [N_samples]) 
            * (far - near) / N_samples
        ).to(rays_o) + z_vals
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    pts_flat = pts.reshape((-1, 3))
    encoded_pts_flat = encoding_fn(pts_flat)
    batches = batch_generator(encoded_pts_flat, batch_size=16384)
    preds = [model(batch) for batch in batches]
    radiance_fields_flat = torch.cat(preds, dim=0)
    radiance_fields = torch.reshape(radiance_fields_flat, list(pts.shape[:-1]) + [4])

    sigma_a = F.relu(radiance_fields[..., 3])
    rgb = torch.sigmoid(radiance_fields[..., :3])

    oneE10 = torch.tensor([1e10], dtype=rays_o.dtype, device=rays_o.device)
    dists = torch.cat([z_vals[..., 1:] - z_vals[..., :-1], oneE10.expand(z_vals[..., :1].shape)], dim=-1)
    alpha = 1 - torch.exp(-sigma_a * dists)
    weights = torch.roll(torch.cumprod(1 - alpha + 1e-10, dim=-1), 1, dims=-1)
    weights[..., 0] = 1
    weights = alpha * weights

    rgb_map = (weights[..., None] * rgb).sum(dim=-2)
    depth_map = (weights * z_vals).sum(dim=-1)
    acc_map = weights.sum(dim=-1)
    return rgb_map, depth_map, acc_map
