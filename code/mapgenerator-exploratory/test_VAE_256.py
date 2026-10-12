# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Shaw Yiran Xiao and Weihao Qiu
import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import json
import os
import matplotlib.pyplot as plt
import argparse
import torchvision.models as models


class PerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        self.slice = nn.Sequential(*list(vgg.children())[:16])  # up to relu_2_2
        for param in self.slice.parameters():
            param.requires_grad = False
        self.resize = resize
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def normalize(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def forward(self, x, y):
        x = x.repeat(1, 3, 1, 1)
        y = y.repeat(1, 3, 1, 1)
        x = self.normalize(x)
        y = self.normalize(y)
        if self.resize:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            y = F.interpolate(y, size=(224, 224), mode="bilinear", align_corners=False)
        fx = self.slice(x)
        fy = self.slice(y)
        return F.mse_loss(fx, fy)


# ========== Dataset ==========
class ConditionalImageDataset(Dataset):
    def __init__(self, image_dir, score_csv, transform=None):
        self.image_dir = image_dir
        self.df = pd.read_csv(score_csv)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row["filename"])
        image = Image.open(image_path).convert("L")  # grayscale
        if self.transform:
            image = self.transform(image)
        score = torch.tensor([row["score"]], dtype=torch.float32)
        return image, score


class JSONConditionalDataset(Dataset):
    def __init__(self, json_file, transform=None):
        with open(json_file, "r") as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("L")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor([item["label"]], dtype=torch.float32)
        return image, label


# ========== Model ==========
class CNNConditionalVAE(nn.Module):
    def __init__(self, cond_dim=1, latent_dim=64):
        super().__init__()

        # ------------ Encoder ------------
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # -> (32, 96, 128)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # -> (64, 48, 64)
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # -> (128, 24, 32)
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),  # -> (256, 12, 16)
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()
        self.encoder_output_dim = 256 * 12 * 16  #  The encoder output uses a 12 by 16 spatial grid

        self.fc_mu = nn.Linear(self.encoder_output_dim + cond_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_output_dim + cond_dim, latent_dim)

        # ------------ Decoder ------------
        self.decoder_input = nn.Linear(latent_dim + cond_dim, self.encoder_output_dim)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 12, 16)),  #  Match the encoder output's 12 by 16 spatial grid
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # -> (128, 24, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # -> (64, 48, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # -> (32, 96, 128)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),  # -> (1, 192, 256)  Target output dimensions
            nn.Sigmoid(),
        )

    def encode(self, x, cond):
        x = self.encoder(x)
        x = self.flatten(x)
        x = torch.cat([x, cond], dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, cond):
        z = torch.cat([z, cond], dim=1)
        x = self.decoder_input(z)
        return self.decoder(x)

    def forward(self, x, cond):
        mu, logvar = self.encode(x, cond)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, cond), mu, logvar


device = (
    "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
)
percep_loss_fn = PerceptualLoss().to(device)


def vae_loss(recon, x, mu, logvar, alpha=1.0, beta=0.01):
    # alpha: perceptual loss weight
    # beta: KL divergence weight
    perceptual = percep_loss_fn(recon, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return alpha * perceptual + beta * kl


# # ========== Loss ==========
# def vae_loss(recon, x, mu, logvar):
#     recon_loss = F.mse_loss(recon, x, reduction='mean')
#     kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
#     return recon_loss + kl


# ========== Visualization ==========
def show_images(images, scores, n=4):
    images = images[:n].detach().cpu()
    scores = scores[:n].squeeze().cpu().numpy()
    fig, axes = plt.subplots(1, n, figsize=(n * 3, 3))
    for i in range(n):
        img = images[i].squeeze(0)
        axes[i].imshow(img * 0.5 + 0.5, cmap="viridis")  # denormalize
        axes[i].set_title(f"Score: {scores[i]:.2f}")
        axes[i].axis("off")
    plt.show()


def save_reconstruction_images(inputs, recons, scores, epoch, save_dir="recon_images", n=4):
    os.makedirs(save_dir, exist_ok=True)
    inputs = inputs[:n].detach().cpu()
    recons = recons[:n].detach().cpu()
    scores = scores[:n].squeeze().cpu().numpy()

    fig, axes = plt.subplots(2, n, figsize=(n * 3, 6))
    for i in range(n):
        input_img = inputs[i].squeeze(0) * 0.5 + 0.5  # denorm
        recon_img = recons[i].squeeze(0) * 0.5 + 0.5  # denorm

        axes[0, i].imshow(input_img, cmap="viridis")
        axes[0, i].set_title(f"Input\nScore: {scores[i]:.2f}")
        axes[0, i].axis("off")

        axes[1, i].imshow(recon_img, cmap="viridis")
        axes[1, i].set_title("Reconstruction")
        axes[1, i].axis("off")

    plt.tight_layout()
    out_path = os.path.join(save_dir, f"epoch_{epoch:03d}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved reconstruction image: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt) to load"
    )
    args = parser.parse_args()

    # Config
    save_every = 10
    batch_size = 8
    num_epochs = 250
    lr = 3e-4
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # Dataset
    transform = transforms.Compose(
        [transforms.Resize((192, 256)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    json_path = "./image_labels.json"
    dataset = JSONConditionalDataset(json_path, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = CNNConditionalVAE().to(device)

    #  Optional: load checkpoint
    if args.checkpoint:
        print(f"Loading model checkpoint from: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for imgs, scores in dataloader:
            imgs, scores = imgs.to(device), scores.to(device)
            recon, mu, logvar = model(imgs, scores)
            loss = vae_loss(recon, imgs, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.6f}")

        if (epoch + 1) % save_every == 0:
            save_path = f"vae_192x256_e_{epoch + 1}.pt"
            torch.save(model.state_dict(), save_path)
            print(f"Saved: {save_path}")
            model.eval()
            with torch.no_grad():
                sample_imgs, sample_scores = next(iter(dataloader))
                sample_imgs = sample_imgs.to(device)
                sample_scores = sample_scores.to(device)
                recon_imgs, _, _ = model(sample_imgs, sample_scores)
                save_reconstruction_images(sample_imgs, recon_imgs, sample_scores, epoch + 1)


if __name__ == "__main__":
    main()
