"""
Autoencoder Training Script
Train a scene-level autoencoder: 512D CLIP features -> 3D embeddings
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class AutoEncoder(nn.Module):
    """Scene-level autoencoder"""
    def __init__(self, input_dim=512, compressed_dim=3):
        super().__init__()

        # Encoder: 512 -> 256 -> 128 -> 64 -> 32 -> 3
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, compressed_dim),
        )

        # Decoder: 3 -> 16 -> 32 -> 64 -> 128 -> 256 -> 256 -> 512
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def encode(self, x):
        """Encode into three dimensions"""
        encoded = self.encoder(x)
        return nn.functional.normalize(encoded, p=2, dim=-1)

    def decode(self, z):
        """Decode back to 512 dimensions"""
        decoded = self.decoder(z)
        return nn.functional.normalize(decoded, p=2, dim=-1)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class LanguageFeatureDataset(Dataset):
    """Language feature dataset"""
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.feature_folder = os.path.join(dataset_path, 'language_features')

        if not os.path.exists(self.feature_folder):
            raise FileNotFoundError(f"Language features not found: {self.feature_folder}")

        # Load all feature files
        self.features = []
        feature_files = sorted([f for f in os.listdir(self.feature_folder) if f.endswith('_f.npy')])

        print(f"Loading features from {len(feature_files)} files...")
        for fname in tqdm(feature_files, desc="Loading"):
            fpath = os.path.join(self.feature_folder, fname)
            features = np.load(fpath)  # (N, 512)
            self.features.append(torch.from_numpy(features).float())

        # Combine all features
        self.features = torch.cat(self.features, dim=0)
        print(f"Total features: {self.features.shape[0]}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


def train_autoencoder(dataset_path, output_path, epochs=100, batch_size=64, lr=0.0007, device='cuda'):
    """Train the autoencoder"""

    print("=" * 80)
    print("Autoencoder Training (512D -> 3D Compression)")
    print("=" * 80)
    print(f"\nDataset: {dataset_path}")
    print(f"Output: {output_path}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {lr}\n")

    # Create the output directory
    os.makedirs(output_path, exist_ok=True)

    # Load the dataset
    print("[Step 1/4] Loading dataset...")
    dataset = LanguageFeatureDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Create the model
    print("\n[Step 2/4] Initializing model...")
    model = AutoEncoder(input_dim=512, compressed_dim=3).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Loss function
    mse_loss = nn.MSELoss()
    cosine_similarity = nn.CosineSimilarity(dim=1)

    # Training loop
    print("\n[Step 3/4] Training...")
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_mse = 0
        total_cosine = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, features in enumerate(progress_bar):
            features = features.to(device)

            # Forward pass
            optimizer.zero_grad()
            reconstructed, encoded = model(features)

            # Compute the loss
            loss_mse = mse_loss(reconstructed, features)
            loss_cosine = 1 - cosine_similarity(reconstructed, features).mean()
            loss = loss_mse + loss_cosine

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            total_loss += loss.item()
            total_mse += loss_mse.item()
            total_cosine += loss_cosine.item()

            # Update the progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mse': f'{loss_mse.item():.4f}',
                'cos': f'{loss_cosine.item():.4f}'
            })

        # EpochStatistics
        avg_loss = total_loss / len(dataloader)
        avg_mse = total_mse / len(dataloader)
        avg_cosine = total_cosine / len(dataloader)

        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Avg MSE: {avg_mse:.4f}")
        print(f"  Avg Cosine: {avg_cosine:.4f}")

        # Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(output_path, 'encoder.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"  [Saved] Best model at epoch {epoch+1} (loss: {avg_loss:.4f})")

        # Save periodic checkpoints
        if (epoch + 1) % 20 == 0:
            checkpoint_path = os.path.join(output_path, f'encoder_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)

    print("\n[Step 4/4] Training complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Model saved to: {os.path.join(output_path, 'encoder.pth')}")

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train autoencoder for feature compression")
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to the dataset with language_features/')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output path for saving checkpoints')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0007,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')

    args = parser.parse_args()

    train_autoencoder(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )
