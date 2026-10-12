# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Shaw Yiran Xiao and Weihao Qiu
import os
import json
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image, make_grid
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.nn.utils import spectral_norm


# ========== Dataset ==========
class ConditionalImageDataset(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.transform = transform or transforms.Compose(
            [transforms.Resize((192, 256)), transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("L")
        image = self.transform(image)
        label = item["label"]
        return image, label


# ========== Models ==========
class Generator(nn.Module):
    def __init__(self, latent_dim, label_dim):
        super().__init__()
        self.label_embed = nn.Sequential(
            nn.Linear(label_dim, 16), nn.ReLU(), nn.Linear(16, 32), nn.ReLU()
        )
        self.model = nn.Sequential(
            nn.Linear(latent_dim + 32, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * 48 * 64),
            nn.ReLU(),
            nn.Unflatten(1, (128, 48, 64)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Tanh(),
            nn.Dropout(0.3),
        )

    def forward(self, z, labels):
        label_feat = self.label_embed(labels)
        x = torch.cat([z, label_feat], dim=1)
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, label_dim):
        super().__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(2, 64, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 48 * 64, 1),
        )

    def forward(self, img, labels):
        label_map = labels.view(-1, 1, 1, 1).expand(-1, 1, 192, 256)
        d_in = torch.cat([img, label_map], dim=1)
        return self.model(d_in)


# ========== Losses ==========
def gradient_penalty(D, real, fake, labels, device):
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = alpha * real + (1 - alpha) * fake
    interpolates.requires_grad_(True)

    d_interpolates = D(interpolates, labels)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:8]
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.eval()

    def forward(self, x, y):
        x = x.expand(-1, 3, -1, -1)
        y = y.expand(-1, 3, -1, -1)
        x_features = self.vgg(x)
        y_features = self.vgg(y)
        return F.l1_loss(x_features, y_features)


def minibatch_stddev(x):
    batch_std = torch.std(x, dim=0, keepdim=True)
    return batch_std.mean()


def setup_tensorboard(log_dir="runs/conditional_gan_wgan_gp"):
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir=log_dir)


def log_to_tensorboard(writer, epoch, g_loss, d_loss, div_loss, generator, latent_dim, device):
    writer.add_scalar("Loss/Generator", g_loss, epoch)
    writer.add_scalar("Loss/Discriminator", d_loss, epoch)
    writer.add_scalar("Loss/Diversity", div_loss, epoch)

    generator.eval()
    with torch.no_grad():
        num_seeds = 3
        num_conditions = 4
        condition_values = torch.linspace(0, 1, num_conditions).to(device)
        all_imgs = []
        for _ in range(num_seeds):
            z = torch.randn(1, latent_dim).repeat(num_conditions, 1).to(device)
            labels = condition_values.unsqueeze(1)
            gen = generator(z, labels)
            all_imgs.append(gen)
        all_imgs = torch.cat(all_imgs, dim=0)
        img_grid = make_grid(all_imgs, nrow=4, normalize=True)
        writer.add_image("Generated Images", img_grid, global_step=epoch)
    generator.train()


# ========== Training Loop ==========
def train_conditional_gan(data_json="image_labels.json", epochs=100, batch_size=32, latent_dim=256):
    dataset = ConditionalImageDataset(data_json)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = Generator(latent_dim, 1).to(device)
    D = Discriminator(1).to(device)
    perceptual_loss = PerceptualLoss().to(device)
    writer = setup_tensorboard()

    optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0001, betas=(0.5, 0.9))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.9))

    lambda_gp = 10
    lambda_perc = 0.5
    lambda_div = 0.3

    for epoch in range(epochs):
        for real_imgs, labels in dataloader:
            real_imgs = real_imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            valid = torch.ones((real_imgs.size(0), 1), device=device)
            fake = torch.zeros((real_imgs.size(0), 1), device=device)

            # Train Discriminator
            optimizer_D.zero_grad()
            z = torch.randn(real_imgs.size(0), latent_dim).to(device)
            fake_imgs = G(z, labels)
            d_real = D(real_imgs, labels).mean()
            d_fake = D(fake_imgs.detach(), labels).mean()
            gp = gradient_penalty(D, real_imgs, fake_imgs.detach(), labels, device)
            d_loss = -d_real + d_fake + lambda_gp * gp
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            fake_imgs = G(z, labels)
            g_loss_adv = -D(fake_imgs, labels).mean()
            g_loss_perc = perceptual_loss(fake_imgs, real_imgs)
            g_loss_div = -minibatch_stddev(fake_imgs)
            g_loss = g_loss_adv + lambda_perc * g_loss_perc + lambda_div * g_loss_div
            g_loss.backward()
            optimizer_G.step()

        print(
            f"[Epoch {epoch + 1}] D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}, G_div: {g_loss_div.item():.4f}"
        )
        log_to_tensorboard(
            writer,
            epoch + 1,
            g_loss.item(),
            d_loss.item(),
            g_loss_div.item(),
            G,
            latent_dim,
            device,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json", type=str, default="image_labels.json", help="path to label json file"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--latent_dim", type=int, default=256)
    args = parser.parse_args()

    train_conditional_gan(
        data_json=args.json,
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
    )
