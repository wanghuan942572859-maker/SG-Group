"""
Feature Compression Script
Compress 512D features to 3D with the trained encoder
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class AutoEncoder(nn.Module):
    """Autoencoder matching the definition in train_autoencoder.py"""
    def __init__(self, input_dim=512, compressed_dim=3):
        super().__init__()

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
        encoded = self.encoder(x)
        return nn.functional.normalize(encoded, p=2, dim=-1)

    def decode(self, z):
        decoded = self.decoder(z)
        return nn.functional.normalize(decoded, p=2, dim=-1)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


def compress_features(dataset_path, encoder_path, device='cuda'):
    """Compress features"""

    print("=" * 80)
    print("Feature Compression (512D -> 3D)")
    print("=" * 80)
    print(f"\nDataset: {dataset_path}")
    print(f"Encoder: {encoder_path}\n")

    # Validate input paths
    feature_folder = os.path.join(dataset_path, 'language_features')
    if not os.path.exists(feature_folder):
        raise FileNotFoundError(f"Language features not found: {feature_folder}")

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder checkpoint not found: {encoder_path}")

    # Create the output directory
    output_folder = os.path.join(dataset_path, 'language_features_dim3')
    os.makedirs(output_folder, exist_ok=True)

    # Load the model
    print("[Step 1/2] Loading encoder...")
    model = AutoEncoder(input_dim=512, compressed_dim=3).to(device)
    checkpoint = torch.load(encoder_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Encoder loaded (trained for {checkpoint['epoch']} epochs)")

    # Collect all feature files
    feature_files = sorted([f for f in os.listdir(feature_folder) if f.endswith('_f.npy')])
    print(f"\nFound {len(feature_files)} feature files")

    # Compress features
    print("\n[Step 2/2] Compressing features...")
    with torch.no_grad():
        for fname in tqdm(feature_files, desc="Compressing"):
            # Load 512D features
            fpath = os.path.join(feature_folder, fname)
            features_512d = np.load(fpath)  # (N, 512)
            features_512d = torch.from_numpy(features_512d).float().to(device)

            # Encode into 3D
            features_3d = model.encode(features_512d)
            features_3d = features_3d.cpu().numpy()

            # Save 3D features
            output_path = os.path.join(output_folder, fname)
            np.save(output_path, features_3d)

            # Copy the segmentation map unchanged
            seg_fname = fname.replace('_f.npy', '_s.npy')
            seg_path_src = os.path.join(feature_folder, seg_fname)
            seg_path_dst = os.path.join(output_folder, seg_fname)
            if os.path.exists(seg_path_src):
                seg_maps = np.load(seg_path_src)
                np.save(seg_path_dst, seg_maps)

    print("\n" + "=" * 80)
    print("Compression Complete!")
    print(f"3D features saved to: {output_folder}")
    print(f"Compression ratio: 512D -> 3D (~170x smaller)")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compress 512D features to 3D")
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to the dataset')
    parser.add_argument('--encoder_path', type=str, required=True,
                       help='Path to the trained encoder checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')

    args = parser.parse_args()

    compress_features(
        dataset_path=args.dataset_path,
        encoder_path=args.encoder_path,
        device=args.device
    )
