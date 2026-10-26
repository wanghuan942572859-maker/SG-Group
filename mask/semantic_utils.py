"""
SG-Group semantic feature utilities.
SG-Group language feature utilities.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List


class AutoEncoder(nn.Module):
    """
    Scene-wise autoencoder for compressing 512D CLIP features to 3D
    Scene-level autoencoder: 512D CLIP features -> 3D embeddings
    """
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
        """Compress to three dimensions"""
        encoded = self.encoder(x)
        # L2 normalize to unit sphere
        return nn.functional.normalize(encoded, p=2, dim=-1)

    def decode(self, z):
        """Decode back to 512 dimensions"""
        decoded = self.decoder(z)
        # L2 normalize
        return nn.functional.normalize(decoded, p=2, dim=-1)

    def forward(self, x):
        """Full encode-decode pass"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


def load_autoencoder(checkpoint_path: str, device: torch.device = torch.device("cuda")) -> AutoEncoder:
    """
    Load a trained autoencoder

    Args:
        checkpoint_path: Autoencoder checkpoint path
        device: Device
    Returns:
        Loaded autoencoder model
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Autoencoder checkpoint not found: {checkpoint_path}")

    model = AutoEncoder().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle supported checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"[Semantic] Loaded autoencoder from {checkpoint_path}")
    return model


def load_language_features(dataset_path: str, image_name: str, dim: int = 3) -> Optional[np.ndarray]:
    """
    Load language features for an image

    Args:
        dataset_path: Dataset root directory
        image_name: Image stem without extension
        dim: Feature dimension (3 or 512)
    Returns:
        Language feature array (N, dim) or None
    """
    folder_name = f"language_features_dim{dim}" if dim == 3 else "language_features"
    feature_path = os.path.join(dataset_path, folder_name, f"{image_name}_f.npy")

    if not os.path.exists(feature_path):
        print(f"[Warning] Language feature not found: {feature_path}")
        return None

    features = np.load(feature_path)
    return features


def load_segmentation_map(dataset_path: str, image_name: str, dim: int = 3) -> Optional[np.ndarray]:
    """
    Load a SAM segmentation map

    Args:
        dataset_path: Dataset root directory
        image_name: Image name
        dim: Corresponding feature dimension
    Returns:
        Segmentation map (4, H, W) or None
    """
    folder_name = f"language_features_dim{dim}" if dim == 3 else "language_features"
    seg_path = os.path.join(dataset_path, folder_name, f"{image_name}_s.npy")

    if not os.path.exists(seg_path):
        print(f"[Warning] Segmentation map not found: {seg_path}")
        return None

    seg_map = np.load(seg_path)
    return seg_map


def compute_semantic_similarity(feat1: torch.Tensor, feat2: torch.Tensor,
                                method: str = "cosine") -> torch.Tensor:
    """
    Compute similarity between semantic features

    Args:
        feat1: First feature: (D,) or (N, D)
        feat2: Second feature: (D,) or (M, D)
        method: Similarity metric ("cosine", "l2", "dot")
    Returns:
        Similarity score
    """
    if method == "cosine":
        # Cosine similarity
        feat1_norm = feat1 / (feat1.norm(dim=-1, keepdim=True) + 1e-8)
        feat2_norm = feat2 / (feat2.norm(dim=-1, keepdim=True) + 1e-8)

        if feat1_norm.dim() == 1 and feat2_norm.dim() == 1:
            similarity = torch.dot(feat1_norm, feat2_norm)
        else:
            similarity = torch.mm(feat1_norm, feat2_norm.T)

        return similarity

    elif method == "l2":
        # Convert L2 distance to similarity
        if feat1.dim() == 1 and feat2.dim() == 1:
            distance = torch.norm(feat1 - feat2)
        else:
            distance = torch.cdist(feat1.unsqueeze(0), feat2.unsqueeze(0)).squeeze()

        similarity = 1.0 / (1.0 + distance)
        return similarity

    elif method == "dot":
        # Dot product
        if feat1.dim() == 1 and feat2.dim() == 1:
            similarity = torch.dot(feat1, feat2)
        else:
            similarity = torch.mm(feat1, feat2.T)
        return similarity

    else:
        raise ValueError(f"Unknown similarity method: {method}")


def aggregate_features(features: torch.Tensor, indices: torch.Tensor,
                       method: str = "mean") -> torch.Tensor:
    """
    Aggregate features at the given indices

    Args:
        features: All features (N, D)
        indices: Indices to aggregate
        method: Aggregation method ("mean", "median", "max")
    Returns:
        Aggregated feature (D,)
    """
    if len(indices) == 0:
        return torch.zeros(features.shape[1], device=features.device)

    selected_features = features[indices]

    if method == "mean":
        return selected_features.mean(dim=0)
    elif method == "median":
        return selected_features.median(dim=0)[0]
    elif method == "max":
        return selected_features.max(dim=0)[0]
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def update_feature_with_ema(old_feature: torch.Tensor, new_feature: torch.Tensor,
                            alpha: float = 0.1) -> torch.Tensor:
    """
    Update features with an exponential moving average

    Args:
        old_feature: Previous feature
        new_feature: New feature
        alpha: Update weight (0-1)
    Returns:
        Updated feature
    """
    updated = (1 - alpha) * old_feature + alpha * new_feature
    # Renormalize
    updated = updated / (updated.norm() + 1e-8)
    return updated


class SemanticFeatureBank:
    """
    Semantic feature bank for mask identities
    """
    def __init__(self, device: torch.device = torch.device("cuda")):
        self.device = device
        self.features: List[torch.Tensor] = []
        self.num_samples: List[int] = []  # Number of samples contributing to each feature

    def add(self, feature: torch.Tensor, num_samples: int = 1):
        """Add a semantic feature"""
        feature = feature.to(self.device)
        feature = feature / (feature.norm() + 1e-8)  # Normalize
        self.features.append(feature)
        self.num_samples.append(num_samples)

    def update(self, idx: int, new_feature: torch.Tensor, new_count: int = 1):
        """
        Update an existing semantic feature by weighted averaging

        Args:
            idx: Feature index
            new_feature: New semantic feature
            new_count: Number of newly added Gaussians
        """
        if idx >= len(self.features):
            raise IndexError(f"Index {idx} out of range")

        new_feature = new_feature.to(self.device)
        new_feature = new_feature / (new_feature.norm() + 1e-8)  # Normalize

        old_count = self.num_samples[idx]
        old_feature = self.features[idx]

        # Weighted average by Gaussian count
        if old_count == 0:
            # Use the new feature if there are no previous samples.
            self.features[idx] = new_feature
        else:
            # Weighted average
            weighted_feature = (old_feature * old_count + new_feature * new_count) / (old_count + new_count)
            weighted_feature = weighted_feature / (weighted_feature.norm() + 1e-8)  # Renormalize
            self.features[idx] = weighted_feature

        self.num_samples[idx] = old_count + new_count

    def get(self, idx: int) -> torch.Tensor:
        """Get a feature by index"""
        if idx >= len(self.features):
            raise IndexError(f"Index {idx} out of range")
        return self.features[idx]

    def compute_similarity(self, query_feature: torch.Tensor, method: str = "cosine") -> torch.Tensor:
        """
        Compare a query feature with all bank features

        Returns:
            Similarity vector (num_masks,)
        """
        if len(self.features) == 0:
            return torch.tensor([], device=self.device)

        query_feature = query_feature.to(self.device)
        query_feature = query_feature / (query_feature.norm() + 1e-8)

        similarities = []
        for feat in self.features:
            sim = compute_semantic_similarity(query_feature, feat, method=method)
            similarities.append(sim.item())

        return torch.tensor(similarities, dtype=torch.float32, device=self.device)

    def __len__(self):
        return len(self.features)

    def save(self, path: str):
        """Save the feature bank"""
        save_dict = {
            'features': [f.cpu().numpy() for f in self.features],
            'num_samples': self.num_samples
        }
        np.save(path, save_dict)
        print(f"[Semantic] Saved feature bank to {path}")

    def load(self, path: str):
        """Load the feature bank"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Feature bank not found: {path}")

        save_dict = np.load(path, allow_pickle=True).item()
        self.features = [torch.tensor(f, device=self.device) for f in save_dict['features']]
        self.num_samples = save_dict['num_samples']
        print(f"[Semantic] Loaded feature bank from {path}")


def visualize_semantic_similarity(similarity_matrix: np.ndarray, save_path: str):
    """
    Visualize the semantic similarity matrix

    Args:
        similarity_matrix: Similarity matrix (N, N)
        save_path: Output path
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, annot=False, cmap='viridis',
                   vmin=0, vmax=1, square=True)
        plt.title('Semantic Similarity Matrix')
        plt.xlabel('Mask Index')
        plt.ylabel('Mask Index')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[Semantic] Saved similarity matrix to {save_path}")
    except ImportError:
        print("[Warning] matplotlib/seaborn not available, skipping visualization")


if __name__ == "__main__":
    # Test code
    print("Testing semantic_utils...")

    # Test AutoEncoder
    model = AutoEncoder()
    x = torch.randn(10, 512)
    x_recon, z = model(x)
    print(f"Input shape: {x.shape}, Encoded shape: {z.shape}, Reconstructed shape: {x_recon.shape}")

    # Test SemanticFeatureBank
    bank = SemanticFeatureBank()
    for i in range(5):
        feat = torch.randn(3)
        bank.add(feat)

    query = torch.randn(3)
    similarities = bank.compute_similarity(query)
    print(f"Similarities with {len(bank)} features: {similarities}")

    print("All tests passed!")
