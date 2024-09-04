from typing import Dict, Optional, Tuple
import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from torchvision.models import inception_v3, Inception_V3_Weights

from diffusion.unet import NaiveUnet
from diffusion.ddpm import DDPM

from utils.crypto_punk import PunkImgDataset

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

# 引入学习率调度器
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts

# 是否启用 WandB 的开关
USE_WANDB = True
# USE_WANDB = False

# Add a switch to enable or disable FID and IS calculation
COMPUTE_FID_IS = False

if USE_WANDB:
    import wandb

PUNK_PATH = "/data/PunkFusion/crypto_data"
OUT_DIR = "./output/lab0_default_cosine"  # 全局变量，用于指定输出目录


def compute_metrics(real_images, generated_images, fid, is_metric):
    with torch.no_grad():
        fid.update(real_images, real=True)
        fid.update(generated_images, real=False)
        is_metric.update(generated_images)

    return fid.compute(), is_metric.compute()


def train_punk(
        n_epoch: int = 310,
        device: str = "cuda:0",
        load_pth: Optional[str] = None
) -> None:
    ddpm = DDPM(eps_model=NaiveUnet(3, 3, n_feat=128), betas=(1e-4, 0.02), n_T=1000)

    if load_pth is not None:
        ddpm.load_state_dict(torch.load(load_pth))

    ddpm.to(device)

    tf = transforms.Compose(  # resize to 512 x 512, convert to tensor, normalize
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = PunkImgDataset(root_dir=PUNK_PATH, transform=tf)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=14)
    optim = torch.optim.Adam(ddpm.parameters(), lr=2e-5)

    # 添加学习率调度器
    scheduler = CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2, eta_min=1e-6)

    if USE_WANDB:
        wandb.init(project="punk-diffusion", config={"n_epochs": n_epoch, "lr": 2e-5})

    global_avg_loss = 0.0
    global_fid_list = []
    global_is_list = []

    # Initialize Inception V3 model and metrics only if COMPUTE_FID_IS is True
    if COMPUTE_FID_IS:
        inception = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device)
        inception.eval()
        fid = FrechetInceptionDistance(normalize=True).to(device)
        is_metric = InceptionScore(normalize=True).to(device)

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "contents"), exist_ok=True)

    for i in range(n_epoch):
        print(f"Epoch {i} : ")
        ddpm.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        for batch_idx, x in enumerate(pbar):
            optim.zero_grad()
            x = x.to(device)
            loss = ddpm(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
            scheduler.step(i + batch_idx / len(dataloader))

        global_avg_loss = (global_avg_loss * i + loss_ema) / (i + 1)
        print(f"Global average loss after epoch {i}: {global_avg_loss:.4f}")

        ddpm.eval()
        with torch.no_grad():
            xh = ddpm.sample(8, (3, 128, 128), device)
            xset = torch.cat([xh, x[:8]], dim=0)
            grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=4)
            save_image(grid, os.path.join(OUT_DIR, "contents", f"ddpm_sample_punk{i:03d}.png"))

            if COMPUTE_FID_IS:
                # Compute FID and IS only if the switch is enabled
                fid_score, is_score = compute_metrics(x[:8], xh, fid, is_metric)
                global_fid_list.append(fid_score.item())
                global_is_list.append(is_score[0].item())

                print(f"FID: {fid_score.item():.4f}, IS: {is_score[0].item():.4f}")

                if USE_WANDB:
                    wandb.log({
                        "epoch": i,
                        "loss": loss_ema,
                        "avg_loss": global_avg_loss,
                        "fid": fid_score.item(),
                        "is": is_score[0].item(),
                        "samples": wandb.Image(grid)
                    })
            else:
                if USE_WANDB:
                    wandb.log({
                        "epoch": i,
                        "loss": loss_ema,
                        "avg_loss": global_avg_loss,
                        "samples": wandb.Image(grid)
                    })

        # Save checkpoint every 10 epochs
        if i % 10 == 0 or i == n_epoch - 1:
            torch.save(ddpm.state_dict(), os.path.join(OUT_DIR, f"ddpm_punk_epoch_{i:03d}.pth"))

    # Save the final checkpoint
    torch.save(ddpm.state_dict(), os.path.join(OUT_DIR, "ddpm_punk_final.pth"))


if __name__ == "__main__":
    train_punk()
