import torch
import torchvision.transforms.functional as TF

from threading import Thread, Lock
import multiprocessing

from PIL import Image, ImageFilter

from typing import Tuple

def center_crop(image: Image.Image):
  size = min(image.width, image.height) // 2
  centerx = image.width // 2
  centery = image.height // 2
  image = image.crop([ centerx-size, centery-size, centerx+size, centery+size ])
  return image

def load_image(image_path: str, size) -> Image.Image:
  image = Image.open(image_path)
  image = center_crop(image)
  image = image.resize((size, size))
  return image

def process_image(image: Image.Image, th:float=0.2) -> Tuple[torch.Tensor, torch.Tensor]:
  source = TF.to_tensor(image.filter(ImageFilter.FIND_EDGES))
  source = (source.mean(0, keepdim=True) > th).type(torch.float32)
  source = TF.normalize(source, [0.5,], [0.5,])

  target = TF.to_tensor(image)
  target = TF.normalize(target, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

  return torch.cat([source, target], dim=0)

def thread_process_image(size:int, th:float, args:str, index:int, output:list, lock:Lock):
  out = []
  for path in args:
    img = load_image(path, size)
    img = process_image(img, th)
    out.append(img)
  
  with lock:
    output[index] = out

def collate_maker(size:int=64, th:float=0.2, cores:int=None):
  def collate_fn(batch):
    num_cores = multiprocessing.cpu_count() if cores is None else cores
    thread_args = [ batch[i::num_cores] for i in range(num_cores) ]

    lock = Lock()
    batch_output = [ None for _ in range(num_cores) ]
    threads = []
    for i in range(num_cores):
      thread = Thread(target=thread_process_image, args=[size, th, thread_args[i], i, batch_output, lock])
      threads.append(thread)
      thread.start()
    
    for thread in threads:
      thread.join()
    
    for i, out in enumerate(batch_output):
      batch[i::num_cores] = out

    stack = torch.stack(batch, dim=0)
    return stack[:,:1,:,:], stack[:,1:,:,:]
  return collate_fn


if __name__ == '__main__':
  import pandas as pd
  from torch.utils.data import DataLoader
  from tqdm import tqdm

  df = pd.read_csv('train_set.csv')
  collate_fn = collate_maker()
  dataloader = DataLoader(df['path'].values, 8, True, collate_fn=collate_fn)
  
  pbar = tqdm(total=len(dataloader), desc='Testing Dataloader...', position=0, colour='yellow')
  for batch in dataloader:
    pbar.update(1)
  pbar.close()
