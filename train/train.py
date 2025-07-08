import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import imageio
from base64 import b64encode
from IPython.display import HTML

from models.nerf_model import NeRF
from utils.rays import get_rays
from utils.render import render_rays
from utils.positional_encoding import positional_encoding
from utils.transforms import pose_spherical
from config.config import *

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Data
if not os.path.exists('data/tiny_nerf_data.npz'):
    os.makedirs('data', exist_ok=True)
    print("Downloading dataset...")
    import urllib.request
    url = 'https://bmild.github.io/nerf/tiny_nerf_data.npz'
    urllib.request.urlretrieve(url, 'data/tiny_nerf_data.npz')

data = np.load('data/tiny_nerf_data.npz')
images = data['images']
poses = data['poses']
focal = data['focal']
testimg, testpose = images[101], poses[101]
images = images[:100, ..., :3]
poses = poses[:100]

images = torch.from_numpy(images).to(device)
poses = torch.from_numpy(poses).to(device)
testimg = torch.from_numpy(testimg).to(device)
testpose = torch.from_numpy(testpose).to(device)

# Image dimensions
HEIGHT, WIDTH = images.shape[1:3]

# Seed
torch.manual_seed(SEED)
np.random.seed(SEED)

# Model, loss, optimizer
encoding_fn = lambda x: positional_encoding(x, L_embed=NUM_ENCODING_FUNCTIONS)
model = NeRF(L_embed=NUM_ENCODING_FUNCTIONS).to(device)
loss_fn = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

psnrs = []
iternums = []

for i in range(NUM_EPOCHS + 1):
    img_idx = np.random.randint(images.shape[0])
    target = images[img_idx]
    pose = poses[img_idx]

    rays_o, rays_d = get_rays(HEIGHT, WIDTH, focal, pose)
    rgb, _, _ = render_rays(
        model, rays_o, rays_d, near=NEAR, far=FAR,
        N_samples=DEPTH_SAMPLES, encoding_fn=encoding_fn
    )

    loss = loss_fn(rgb, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % DISPLAY_EVERY == 0:
        rays_o, rays_d = get_rays(HEIGHT, WIDTH, focal, testpose)
        rgb, _, _ = render_rays(
            model, rays_o, rays_d, near=NEAR, far=FAR,
            N_samples=DEPTH_SAMPLES, encoding_fn=encoding_fn
        )
        test_loss = loss_fn(rgb, testimg)
        print(f"Step {i} | Test Loss: {test_loss.item()}")
        psnr = -10 * torch.log10(test_loss)
        psnrs.append(psnr.item())
        iternums.append(i)

        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.imshow(rgb.detach().cpu().numpy())
        plt.title(f'Iteration: {i}')
        plt.subplot(122)
        plt.plot(iternums, psnrs)
        plt.title('PSNR')
        plt.show()

print("Training Done!")

# Video rendering
print("Rendering video...")

frames = []
for th in np.linspace(0., 360., 120, endpoint=False):
    c2w = pose_spherical(th, -30, 4)
    c2w = torch.from_numpy(c2w).to(device).float()
    rays_o, rays_d = get_rays(HEIGHT, WIDTH, focal, c2w[:3, :4])
    rgb, _, _ = render_rays(
        model, rays_o, rays_d, near=NEAR, far=FAR,
        N_samples=DEPTH_SAMPLES, encoding_fn=encoding_fn
    )
    frames.append((255 * np.clip(rgb.cpu().detach().numpy(), 0, 1)).astype(np.uint8))

os.makedirs("output", exist_ok=True)
video_path = "output/video.mp4"
imageio.mimwrite(video_path, frames, fps=30, quality=7)

print(f"Video saved at: {video_path}")

import platform
import subprocess

def open_video(path):
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Linux":
        subprocess.run(["xdg-open", path])

open_video(video_path)
