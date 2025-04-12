import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from python.models.pointnet_generator import PointNetGenerator, PointNetDiscriminator
from python.training.dataset import LidarForgeDataset


def compute_chamfer(pred: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    B, N, C = pred.shape
    _, M, _ = real.shape
    pred_exp = pred.unsqueeze(2).expand(B, N, M, C)
    real_exp = real.unsqueeze(1).expand(B, N, M, C)
    dist = ((pred_exp - real_exp) ** 2).sum(-1)
    return dist.min(2)[0].mean() + dist.min(1)[0].mean()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    train_ds = LidarForgeDataset(args.data_dir, num_points=args.num_points, split="train")
    val_ds   = LidarForgeDataset(args.data_dir, num_points=args.num_points, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    G = PointNetGenerator(condition_dim=64, num_points=args.num_points).to(device)
    D = PointNetDiscriminator(in_channels=4).to(device)

    opt_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))
    scaler = GradScaler()

    bce = nn.BCEWithLogitsLoss()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        G.train(); D.train()
        g_total = d_total = 0.0

        for real_pts, cond in train_loader:
            real_pts = real_pts.to(device)
            cond     = cond.to(device)
            B = real_pts.size(0)
            ones  = torch.ones(B, 1, device=device)
            zeros = torch.zeros(B, 1, device=device)

            with autocast():
                fake_pts = G(cond)
                d_real = D(real_pts)
                d_fake = D(fake_pts.detach())
                d_loss = bce(d_real, ones) + bce(d_fake, zeros)

            opt_D.zero_grad()
            scaler.scale(d_loss).backward()
            scaler.step(opt_D)

            with autocast():
                fake_pts = G(cond)
                adv = bce(D(fake_pts), ones)
                chamfer = compute_chamfer(fake_pts, real_pts)
                g_loss = adv + args.lambda_chamfer * chamfer

            opt_G.zero_grad()
            scaler.scale(g_loss).backward()
            scaler.step(opt_G)
            scaler.update()

            g_total += g_loss.item()
            d_total += d_loss.item()

        G.eval()
        val_chamfer = 0.0
        with torch.no_grad():
            for real_pts, cond in val_loader:
                real_pts = real_pts.to(device)
                cond     = cond.to(device)
                fake_pts = G(cond)
                val_chamfer += compute_chamfer(fake_pts, real_pts).item()
        val_chamfer /= len(val_loader)

        print(f"Epoch {epoch}/{args.epochs}  G={g_total/len(train_loader):.4f}  "
              f"D={d_total/len(train_loader):.4f}  val_chamfer={val_chamfer:.4f}")

        if val_chamfer < best_val:
            best_val = val_chamfer
            torch.save(G.state_dict(), os.path.join(args.checkpoint_dir, "best_generator.pt"))
        torch.save(G.state_dict(), os.path.join(args.checkpoint_dir, "last_generator.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",       default="output")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--epochs",         type=int,   default=50)
    parser.add_argument("--batch_size",     type=int,   default=8)
    parser.add_argument("--num_points",     type=int,   default=16384)
    parser.add_argument("--lambda_chamfer", type=float, default=10.0)
    train(parser.parse_args())
