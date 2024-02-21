import torch
from torch import nn
from torch import Tensor

import numpy as np
from typing import Tuple
from tqdm import tqdm

import matplotlib.pyplot as plt

def diffusion_test_beta(start:float, end:float, steps:int=1000):
  beta = np.linspace(start, end, steps)
  alpha = 1.0 - beta

  alpha_hat = np.cumprod(alpha, axis=0)

  alpha_hat_sqrt = np.sqrt(alpha_hat)
  alpha_minus_one_hat_sqrt = np.sqrt(1.0 - alpha_hat)

  plt.subplot(2, 1, 1)
  plt.plot(beta, label='beta')
  plt.plot(alpha, label='1-beta')
  plt.legend()
  plt.subplot(2, 1, 2)
  plt.plot(alpha_hat, label='A')
  plt.plot(alpha_hat_sqrt, label='sqrt(A)')
  plt.plot(alpha_minus_one_hat_sqrt, label='sqrt(1-A)')
  plt.legend()
  plt.show()

class Diffusion:
  def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=2e-2, img_size=64, device='cpu'):
    self.noise_steps = noise_steps
    self.beta_start = beta_start
    self.beta_end = beta_end
    self.img_size = img_size
    self.device = device

    self.beta = torch.linspace(self.beta_start, self.beta_end, self.noise_steps, device=device)
    self.alpha = 1.0 - self.beta
    self.alpha_hat = torch.cumprod(self.alpha, dim=0)
  
  def noise_image(self, x:Tensor, t:Tensor) -> Tuple[Tensor, Tensor]:
    sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat[t])[:, None, None, None]
    e = torch.randn_like(x, device=self.device)
    return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * e, e

  def sample_timesteps(self, n:int) -> Tensor:
    return torch.randint(low=1, high=self.noise_steps, size=(n,), device=self.device)
  
  def sample(self, model:nn.Module, n:int, src:Tensor=None) -> Tensor:
    model.eval()
    with torch.inference_mode():
      x = torch.randn((n, 3, self.img_size, self.img_size), device=self.device)
      pbar = tqdm(total=self.noise_steps, desc='Sampling', colour='magenta', position=0)
      pbar.update(1)
      for i in reversed(range(1, self.noise_steps)):
        t = (torch.ones(n, device=self.device) * i).long()
        predicted_noise:Tensor = model(x, t, src)
        alpha = self.alpha[t][:, None, None, None]
        alpha_hat = self.alpha_hat[t][:, None, None, None]
        beta = self.beta[t][:, None, None, None]
        if i > 1:
          noise = torch.randn_like(x)
        else:
          noise = torch.zeros_like(x)
        x = 1 / torch.sqrt(alpha) * (x - ((1-alpha)/(torch.sqrt(1 - alpha_hat)))*predicted_noise) + torch.sqrt(beta)*noise
        pbar.update(1)
      pbar.close()
    model.train()
    x = (x.clamp(-1, 1) + 1) / 2
    x = (x * 255).type(torch.uint8)
    return x


if __name__ == '__main__':
  from dataset import collate_maker
  import pandas as pd
  from torch.utils.data import DataLoader

  img_size = 64
  batch_size = 8

  df = pd.read_csv('train_set.csv')
  collate_fn = collate_maker(size=img_size, th=0.3)
  dataloader = DataLoader(df['path'].values, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
  diffusion = Diffusion(100, 1e-4, 1e-1)

  x, y = next(iter(dataloader))

  t = diffusion.sample_timesteps(batch_size)
  x_t, noise = diffusion.noise_image(y, t)

  y = y * 0.5 + 0.5
  x_t = (x_t * 0.5 + 0.5).clamp(0.0, 1.0)

  plt.figure(figsize=(batch_size*2, 4))
  for i in range(batch_size):
    plt.subplot(3, 8, i+1)
    plt.imshow(y[i].detach().cpu().moveaxis(0, -1).numpy())
    plt.title(f'{t[i]}')
    plt.axis('off')
    plt.subplot(3, 8, 8+i+1)
    plt.imshow(x_t[i].detach().cpu().moveaxis(0, -1).numpy())
    plt.axis('off')
    plt.subplot(3, 8, 2*8+i+1)
    plt.imshow(x[i].detach().cpu().moveaxis(0, -1).numpy(), cmap='gray')
    plt.axis('off')
  plt.show()