import warnings as w

w.simplefilter(action="ignore")
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import cv2
from tqdm.auto import tqdm

parser = argparse.ArgumentParser(description="Variational Autoencoder")
parser.add_argument(
    "--input", type=str, required=True, help="Directory containing images eg: data/"
)
parser.add_argument("--name", type=str, required=True, help="Name for the model")
parser.add_argument(
    "--epoch", type=int, default=500, help="Number of training iterations"
)
parser.add_argument("--batch", type=int, default=100, help="Batch Size")
parser.add_argument(
    "--original_dim", type=int, default=128, help="Dimension of Intermediate Layer"
)
parser.add_argument(
    "--inter_dim", type=int, default=256, help="Dimension of Intermediate Layer"
)
args = parser.parse_args()

# Parameters
epochs = args.epoch
intermediate_dim = args.inter_dim
batch_size = args.batch
original_dim = args.original_dim * args.original_dim * 3
latent_dim = 2
epsilon_std = 1.0


# Custom Dataset
class ImageDataset(Dataset):
    def __init__(self, img_dir):
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        img = img.astype("float32") / 255.0
        img = cv2.resize(img, (args.original_dim, args.original_dim))
        return torch.tensor(img, dtype=torch.float32).view(-1)


# VAE Model
class VAE(nn.Module):
    def __init__(self, original_dim, intermediate_dim, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(original_dim, intermediate_dim)
        self.fc2_mean = nn.Linear(intermediate_dim, latent_dim)
        self.fc2_log_var = nn.Linear(intermediate_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, intermediate_dim)
        self.fc4 = nn.Linear(intermediate_dim, original_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2_mean(h), self.fc2_log_var(h)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return self.decode(z), mean, log_var


# Loss Function
def loss_function(recon_x, x, mean, log_var):
    BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return BCE + KLD


# Load Data
train_dataset = ImageDataset(args.input)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize Model
vae = VAE(original_dim, intermediate_dim, latent_dim).to(
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

if os.path.exists(f"{args.name}.h5"):
    vae.load_state_dict(torch.load(f"{args.name}.h5", map_location=torch.device("cpu")))

# Training Loop
vae.train()
pbar = tqdm(range(epochs))
for epoch in pbar:
    train_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        recon_batch, mean, log_var = vae(batch)
        loss = loss_function(recon_batch, batch, mean, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    pbar.set_description_str(
        f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss / len(train_loader.dataset):.4f}"
    )
torch.save(vae.state_dict(), f"{args.name}.h5")

# Generate Images
vae.eval()
with torch.no_grad():
    n = 4
    digit_size = args.original_dim
    figure = np.zeros((digit_size * n, digit_size * n, 3))
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = torch.tensor([[xi, yi]], dtype=torch.float32)
            x_decoded = vae.decode(z_sample).numpy()
            newimage = x_decoded.reshape(digit_size, digit_size, 3)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = newimage

    plt.imshow(figure)
    plt.axis("off")
    plt.show()
