import torch
from torch import nn
import torch.nn.functional as F

from torch import Tensor

class DoubleConv(nn.Module):
  def __init__(self, in_ch:int, out_ch:int, mid_ch:int=None, residual:bool=False, device='cpu', dtype=torch.float32):
    kwargs_factory = { 'device': device, 'dtype': dtype }
    super(DoubleConv, self).__init__()
    self.residual = residual
    if not mid_ch:
      mid_ch = out_ch
    self.double_conv = nn.Sequential(
      nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False, **kwargs_factory),
      nn.GroupNorm(1, mid_ch, **kwargs_factory),
      nn.GELU(),
      nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False, **kwargs_factory),
      nn.GroupNorm(1, out_ch, **kwargs_factory),
    )
  def forward(self, x: Tensor) -> Tensor:
    if self.residual:
      return F.gelu(x + self.double_conv(x))
    else:
      return self.double_conv(x)

class Down(nn.Module):
  def __init__(self, in_ch:int, out_ch:int, emb_dim:int=256, device='cpu', dtype=torch.float32):
    kwargs_factory = { 'device': device, 'dtype': dtype }
    super(Down, self).__init__()
    self.maxpool_conv = nn.Sequential(
      nn.MaxPool2d(2),
      DoubleConv(in_ch, in_ch, residual=True, **kwargs_factory),
      DoubleConv(in_ch, out_ch, **kwargs_factory),
    )
    self.emb_layer = nn.Sequential(
      nn.SiLU(),
      nn.Linear(
        emb_dim,
        out_ch,
        **kwargs_factory,
      )
    )
  def forward(self, x:Tensor, t:Tensor, src:Tensor=None) -> Tensor:
    if src is not None:
      x[:, ::2, :, :] += src
    x = self.maxpool_conv(x)
    emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
    return x + emb

class Up(nn.Module):
  def __init__(self, in_ch:int, out_ch:int, emb_dim:int=256, device='cpu', dtype=torch.float32):
    kwargs_factory = { 'device':device, 'dtype':dtype }
    super(Up, self).__init__()

    self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    self.conv = nn.Sequential(
      DoubleConv(in_ch, in_ch, residual=True, **kwargs_factory),
      DoubleConv(in_ch, out_ch, in_ch // 2, **kwargs_factory)
    )
    self.emb_layer = nn.Sequential(
      nn.SiLU(),
      nn.Linear(
        emb_dim,
        out_ch,
        **kwargs_factory,
      )
    )
  
  def forward(self, x:Tensor, skip_x:Tensor, t:Tensor, src:Tensor=None) -> Tensor:
    x = self.up(x)
    if src is not None:
      x[:,::2,:,:] += src
    x = torch.cat([skip_x, x], dim=1)
    x = self.conv(x)
    emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
    return x + emb

class SelfAttention(nn.Module):
  def __init__(self, channels:int, size:int, device='cpu', dtype=torch.float32):
    kwargs_factory = { 'device':device, 'dtype': dtype }
    super(SelfAttention, self).__init__()
    self.channels = channels
    self.size = size
    self.mha = nn.MultiheadAttention(
      channels,
      4,
      batch_first=True,
      **kwargs_factory,
    )
    self.ln = nn.LayerNorm([channels])
    self.ff_self = nn.Sequential(
      nn.LayerNorm([channels], **kwargs_factory),
      nn.Linear(channels, channels, **kwargs_factory),
      nn.GELU(),
      nn.Linear(channels, channels, **kwargs_factory),
    )

  def forward(self, x:Tensor) -> Tensor:
    x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
    x_ln = self.ln(x)
    attention_value, _ = self.mha(x_ln, x_ln, x_ln)
    attention_value = attention_value + x
    attention_value = self.ff_self(attention_value) + attention_value
    return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)  
  

class UNet(nn.Module):
  def __init__(self, c_in:int=3, c_out:int=3, time_dim:int=256, device='cpu', dtype=torch.float32):
    kwargs_factory = { 'device': device, 'dtype': dtype }
    super(UNet, self).__init__()
    self.device = device
    self.time_dim = time_dim
    self.inc1 = DoubleConv(c_in, 64, **kwargs_factory)
    self.down1 = Down(64, 128, **kwargs_factory)
    self.sa1 = SelfAttention(128, 32, **kwargs_factory)
    self.down2 = Down(128, 256, **kwargs_factory)
    self.sa2 = SelfAttention(256, 16, **kwargs_factory)
    self.down3 = Down(256, 256, **kwargs_factory)
    self.sa3 = SelfAttention(256, 8, **kwargs_factory)

    self.bot1 = DoubleConv(256, 512, **kwargs_factory)
    self.bot2 = DoubleConv(512, 512, **kwargs_factory)
    self.bot3 = DoubleConv(512, 256, **kwargs_factory)

    self.up1 = Up(512, 128, **kwargs_factory)
    self.sa4 = SelfAttention(128, 16, **kwargs_factory)
    self.up2 = Up(256, 64, **kwargs_factory)
    self.sa5 = SelfAttention(64, 32, **kwargs_factory)
    self.up3 = Up(128, 64, **kwargs_factory)
    self.sa6 = SelfAttention(64, 64, **kwargs_factory)
    self.outc = nn.Conv2d(64, c_out, kernel_size=1, **kwargs_factory)

  def pos_encoding(self, t:Tensor, channels:int) -> Tensor:
    inv_freq = 1.0 / (
      10000 **
      (torch.arange(0, channels, 2, device=self.device).float() / channels)
    ).to(self.device)
    pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
    pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
    pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
    return pos_enc

  def forward(self, x:Tensor, t:Tensor, src:Tensor=None) -> Tensor:
    t = t.unsqueeze(-1).type(torch.float)
    t = self.pos_encoding(t, self.time_dim)

    x1 = self.inc1(x)
    x2 = self.down1(x1, t, src)
    x2 = self.sa1(x2)
    x3 = self.down2(x2, t, None if src is None else F.max_pool2d(src, 2))
    x3 = self.sa2(x3)
    x4 = self.down3(x3, t, None if src is None else F.max_pool2d(src, 4))
    x4 = self.sa3(x4)

    x4 = self.bot1(x4)
    x4 = self.bot2(x4)
    x4 = self.bot3(x4)

    x = self.up1(x4, x3, t, None if src is None else F.max_pool2d(src, 4))
    x = self.sa4(x)
    x = self.up2(x, x2, t, None if src is None else F.max_pool2d(src, 2))
    x = self.sa5(x)
    x = self.up3(x, x1, t, src)
    x = self.sa6(x)
    output = self.outc(x)
    return output


if __name__ == '__main__':
  import pandas as pd

  import torch
  from torch.utils.data import DataLoader
  from torch import optim

  import random

  from dataset import collate_maker
  from diffusion import Diffusion, diffusion_test_beta

  from tqdm import tqdm

  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  batch_size = 8

  df = pd.read_csv('train_set.csv')
  collate_fn = collate_maker(64, 0.3)
  dataloader = DataLoader(df['path'].values, batch_size, True, collate_fn=collate_fn)

  diffusion = Diffusion(100, 1e-4, 1e-1, img_size=64, device=device)
  model = UNet(3, 3, 256, device)
  model.train()

  loss_fn = nn.MSELoss()
  optimizer = optim.AdamW(model.parameters(), lr=1e-4)

  pbar = tqdm(total=len(dataloader), desc='Train', position=0, colour='yellow')
  for i, (src, tgt) in enumerate(dataloader):
    src = src.to(device)
    tgt = tgt.to(device)

    t = diffusion.sample_timesteps(tgt.shape[0])
    x_t, noise = diffusion.noise_image(tgt, t)
    predicted_noise = model(x_t, t, src if random.random() < 0.1 else None)

    loss = loss_fn(noise, predicted_noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    pbar.set_postfix(MSE=loss.item())
    pbar.update(1)

    if i > 3:
      break
  pbar.close()
