"""
Semantic-enhanced GaussianProjector
Semantic Gaussian projector: associate masks using geometric overlap and semantic similarity
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import json
from PIL import Image

from arguments import ModelParams, PipelineParams
from mask.utils import get_n_different_colors, ndc2Pixel, transformPoint4x4, convert_matched_mask, mask_id_to_binary_mask
from mask.semantic_utils import (
    load_autoencoder, compute_semantic_similarity,
    aggregate_features, SemanticFeatureBank
)

from scene import Scene
from scene.gaussian_model_semantic import GaussianModelSemantic


default_params = {
    "seg_method": "sam",
    "front_percentage": 0.2,
    "overlap_threshold": 0.1,  # Geometric overlap threshold (IOCUR)
    "semantic_threshold": 0.5,  # Semantic similarity threshold
    "semantic_weight": 0.5,  # Semantic weight
    "combined_threshold": 0.2,  # Combined-score threshold
    "num_patch": 32,
    "visualize": False,
    "autoencoder_path": None,  # Autoencoder path
    "use_full_dim": False,  # Whether to compare decoded 512D features
    "similarity_method": "cosine",  # Similarity metric
    "feature_update_alpha": 0.1,  # Legacy EMA weight; weighted averaging is used
}


class GaussianProjectorSemantic(torch.nn.Module):
    """
    Semantic Gaussian projector
    Associate masks using geometric and semantic constraints.
    """
    def __init__(self,
                 dataset: ModelParams,
                 pipeline: PipelineParams,
                 iteration: int,
                 params: dict = default_params,
                 device: torch.device = torch.device("cuda"),
                 ):
        super(GaussianProjectorSemantic, self).__init__()
        self.device = device

        # ========== Load the Gaussian model ==========
        self.gaussians = GaussianModelSemantic(dataset.sh_degree)
        scene = Scene(dataset, self.gaussians, load_iteration=iteration, shuffle=False)
        self.gaussians_xyz = self.gaussians.get_xyz.to(self.device)
        self.viewpoint_camera = scene.getTrainCameras()

        self.gaussians_language_features = self.gaussians.get_language_feature.to(self.device)
        self.has_language_features = True
        print(f"[Semantic] Loaded language features: {self.gaussians_language_features.shape}")

        # ========== Geometric parameters ==========
        self.front_percentage = params["front_percentage"]
        self.iou_threshold = params["overlap_threshold"]
        self.num_patches = params["num_patch"]

        # ========== Semantic parameters==========
        self.semantic_threshold = params.get("semantic_threshold", 0.7)
        self.semantic_weight = params.get("semantic_weight", 0.3)
        self.combined_threshold = params.get("combined_threshold", 0.35)
        self.use_full_dim = params.get("use_full_dim", False)
        self.similarity_method = params.get("similarity_method", "cosine")
        self.feature_update_alpha = params.get("feature_update_alpha", 0.1)

        # Load the autoencoder for 512D comparison
        self.autoencoder = None
        if self.use_full_dim:
            if not params.get("autoencoder_path"):
                raise ValueError("512D comparison requires --autoencoder_path")
            self.autoencoder = load_autoencoder(params["autoencoder_path"], device)
            print("[Semantic] Loaded autoencoder for 512D comparison")

        # ========== Paths ==========
        self.source_path = dataset.source_path
        self.seg_method = params["seg_method"]
        self.raw_mask_folder = os.path.join(self.source_path, f"raw_{self.seg_method}_mask")
        assert os.path.exists(self.raw_mask_folder), f"Mask folder does not exist: {self.raw_mask_folder}"

        self.associated_mask_folder = os.path.join(self.source_path, f"{self.seg_method}_mask_semantic")
        os.makedirs(self.associated_mask_folder, exist_ok=True)

        if params["visualize"]:
            self.visualize = True
            self.visualize_folder = os.path.join(self.source_path, f"{self.seg_method}_mask_semantic_vis")
            os.makedirs(self.visualize_folder, exist_ok=True)
            self.random_colors = get_n_different_colors(10000)  # Increased from 1000 to handle more masks
        else:
            self.visualize = False

        # ========== Mask patches ==========
        random_mask = self.load_mask(self.viewpoint_camera[0])
        self.image_width, self.image_height = random_mask.shape[1], random_mask.shape[2]
        self.patch_width = (self.image_width // self.num_patches + 1
                           if self.image_width % self.num_patches != 0
                           else self.image_width // self.num_patches)
        self.patch_height = (self.image_height // self.num_patches + 1
                            if self.image_height % self.num_patches != 0
                            else self.image_height // self.num_patches)

        self.patch_mask = torch.zeros(
            (self.num_patches, self.num_patches, self.image_width, self.image_height),
            dtype=torch.bool, device=self.device
        )
        for i in range(self.num_patches):
            for j in range(self.num_patches):
                self.patch_mask[i, j,
                               i * self.patch_width: (i + 1) * self.patch_width,
                               j * self.patch_height: (j + 1) * self.patch_height] = True
        self.flatten_patch_mask = self.patch_mask.flatten(start_dim=2)

        # ========== Mask association state ==========
        self.gaussian_idx_bank = []  # Gaussian indices for each mask
        self.semantic_feature_bank = SemanticFeatureBank(device=self.device)  # Semantic feature bank
        self.num_mask = 0
        self.assigned_gaussians = torch.tensor([], dtype=torch.long, device=self.device)

    @property
    def get_num_mask(self):
        if len(self.gaussian_idx_bank) == 0:
            self.num_mask = 0
            return 0
        self.num_mask = len(self.gaussian_idx_bank)
        return self.num_mask

    def get_semantic_feature_of_mask(self, front_gaussian_indices: torch.Tensor) -> torch.Tensor:
        """
        Extract a mask's semantic feature

        Args:
            front_gaussian_indices: Indices of foreground Gaussians
        Returns:
            Normalized mean semantic feature for the mask
        """
        if not self.has_language_features:
            # Return a zero vector if no language features are available
            dim = 512 if self.use_full_dim else 3
            return torch.zeros(dim, device=self.device)

        if len(front_gaussian_indices) == 0:
            return torch.zeros(
                512 if self.use_full_dim else 3,
                device=self.device
            )

        # Get language features for these Gaussians
        language_features_3d = self.gaussians_language_features[front_gaussian_indices]  # (N, 3)

        # Aggregate features by averaging
        avg_feature_3d = aggregate_features(
            language_features_3d,
            torch.arange(len(front_gaussian_indices), device=self.device),
            method="mean"
        )

        # Decode for 512D comparison when requested
        if self.use_full_dim and self.autoencoder is not None:
            with torch.no_grad():
                avg_feature_512d = self.autoencoder.decode(avg_feature_3d.unsqueeze(0)).squeeze(0)
            return avg_feature_512d / (avg_feature_512d.norm() + 1e-8)
        else:
            return avg_feature_3d / (avg_feature_3d.norm() + 1e-8)

    def maintain_gaussian_idx_bank(self, idx, front_gaussian_of_mask):
        """Maintain the Gaussian index bank."""
        assert not idx > self.num_mask, "idx is larger than the number of masks"

        if idx == self.num_mask:
            # New mask
            self.gaussian_idx_bank.append(front_gaussian_of_mask)
            if len(self.assigned_gaussians) == 0:
                self.assigned_gaussians = front_gaussian_of_mask
            else:
                self.assigned_gaussians = torch.unique(
                    torch.cat([self.assigned_gaussians, front_gaussian_of_mask])
                )
        else:
            # Update an existing mask
            non_assigned_gaussians = front_gaussian_of_mask[
                ~torch.isin(front_gaussian_of_mask, self.assigned_gaussians)
            ]
            if len(non_assigned_gaussians) > 0:
                non_assigned_gaussians = torch.unique(non_assigned_gaussians)
                self.gaussian_idx_bank[idx] = torch.unique(
                    torch.cat([self.gaussian_idx_bank[idx], non_assigned_gaussians])
                )
                self.assigned_gaussians = torch.unique(
                    torch.cat([self.assigned_gaussians, non_assigned_gaussians])
                )

    def initialize(self, viewpoint):
        """Initialize the first view"""
        front_gaussian, mask = self.get_patch_front_gaussian_of_mask(viewpoint)
        self.gaussian_idx_bank.extend(front_gaussian)

        # Initialize the semantic feature bank
        for fg in front_gaussian:
            semantic_feature = self.get_semantic_feature_of_mask(fg)
            self.semantic_feature_bank.add(semantic_feature, num_samples=len(fg))

        # Update assigned Gaussians
        if len(front_gaussian) > 0:
            self.assigned_gaussians = torch.unique(torch.cat(front_gaussian))
        else:
            self.assigned_gaussians = torch.tensor([], dtype=torch.long, device=self.device)

        self.get_num_mask
        labels = torch.arange(self.num_mask, dtype=torch.long, device=self.device)

        print(f"[Semantic] Initialized with {self.num_mask} masks")
        return labels

    def associate(self, viewpoint):
        """
        Associate masks using geometry and semantics
        Combine geometric overlap (IOCUR) and semantic similarity
        """
        front_gaussian, mask = self.get_patch_front_gaussian_of_mask(viewpoint)
        num_mask_cur_view = len(front_gaussian)

        if num_mask_cur_view == 0:
            return torch.tensor([], dtype=torch.long, device=self.device)

        self.get_num_mask
        labels = torch.zeros(num_mask_cur_view, dtype=torch.long, device=self.device)

        for m_idx in range(num_mask_cur_view):
            front_gaussian_of_mask = front_gaussian[m_idx]

            if len(front_gaussian_of_mask) == 0:
                labels[m_idx] = self.num_mask
                self.maintain_gaussian_idx_bank(self.num_mask, front_gaussian_of_mask)
                dim = 512 if self.use_full_dim else 3
                self.semantic_feature_bank.add(
                    torch.zeros(dim, device=self.device), num_samples=0
                )
                self.get_num_mask
                continue

            # ========== 1. Compute geometric overlap (IOCUR)==========
            num_union = [
                len(torch.unique(torch.cat([self.gaussian_idx_bank[i], front_gaussian_of_mask])))
                for i in range(self.num_mask)
            ]
            num_intersection = [
                len(self.gaussian_idx_bank[i]) + len(front_gaussian_of_mask) - num_union[i]
                for i in range(self.num_mask)
            ]
            num_cur = len(front_gaussian_of_mask)

            # IOCUR: Intersection Over Current
            io_cur = [
                num_intersection[i] / (num_cur + num_intersection[i] + 1e-8)
                for i in range(self.num_mask)
            ]
            io_cur = torch.tensor(io_cur, dtype=torch.float32, device=self.device)

            # ========== 2. Compute semantic similarity==========
            if self.has_language_features:
                current_semantic_feature = self.get_semantic_feature_of_mask(front_gaussian_of_mask)
                semantic_similarity = self.semantic_feature_bank.compute_similarity(
                    current_semantic_feature,
                    method=self.similarity_method
                )
            else:
                # Set similarity to zero if semantic features are unavailable
                semantic_similarity = torch.zeros(self.num_mask, device=self.device)

            # ========== 3. Combine geometric and semantic scores ==========
            # Use a weighted average
            combined_score = (
                (1 - self.semantic_weight) * io_cur +
                self.semantic_weight * semantic_similarity
            )

            # ========== 4. Select the best match ==========
            if self.num_mask == 0:
                selected_mask = 0
            else:
                selected_mask = torch.argmax(combined_score).item()

            # ========== 5. Apply the match threshold ==========
            if self.num_mask == 0 or combined_score[selected_mask] < self.combined_threshold:
                # Create a new mask.
                selected_mask = self.num_mask
                self.maintain_gaussian_idx_bank(selected_mask, front_gaussian_of_mask)

                # Add the new semantic feature
                if self.has_language_features:
                    new_semantic_feature = self.get_semantic_feature_of_mask(front_gaussian_of_mask)
                    self.semantic_feature_bank.add(new_semantic_feature, num_samples=len(front_gaussian_of_mask))
                else:
                    dim = 512 if self.use_full_dim else 3
                    self.semantic_feature_bank.add(torch.zeros(dim, device=self.device), num_samples=0)

            else:
                # Match an existing mask
                self.maintain_gaussian_idx_bank(selected_mask, front_gaussian_of_mask)

                # Update the semantic feature by weighted averaging
                if self.has_language_features:
                    new_semantic_feature = self.get_semantic_feature_of_mask(front_gaussian_of_mask)
                    self.semantic_feature_bank.update(
                        selected_mask,
                        new_semantic_feature,
                        new_count=len(front_gaussian_of_mask)  # Weight by the number of Gaussians
                    )

            labels[m_idx] = selected_mask
            self.get_num_mask

            # Print debug information for the first few masks
            if m_idx < 3 and self.num_mask > 1:
                # New masks have no IOCUR score; check the selected index
                if selected_mask < len(io_cur):
                    print(f"[Debug] Mask {m_idx}: Geometric={io_cur[selected_mask]:.3f}, "
                          f"Semantic={semantic_similarity[selected_mask]:.3f}, "
                          f"Combined={combined_score[selected_mask]:.3f}, "
                          f"Selected={selected_mask}")
                else:
                    print(f"[Debug] Mask {m_idx}: New mask created, Selected={selected_mask}")

        return labels

    def build_mask_association(self):
        """Build mask associations for all views"""
        info_path = os.path.join(self.associated_mask_folder, "info.json")
        if os.path.exists(info_path):
            os.remove(info_path)
        print(f"\n[Semantic] Building mask association with semantic enhancement...")
        print(f"[Semantic] Parameters:")
        print(f"  - Geometric weight: {1 - self.semantic_weight:.2f}")
        print(f"  - Semantic weight: {self.semantic_weight:.2f}")
        print(f"  - Combined threshold: {self.combined_threshold:.2f}")
        print(f"  - Has language features: {self.has_language_features}")

        for view_idx, view in enumerate(tqdm(self.viewpoint_camera, desc="Associating masks")):
            view = view.to(self.device)

            if self.num_mask == 0:
                labels = self.initialize(view)
            else:
                labels = self.associate(view)

            # Save the associated mask
            mask_path = os.path.join(self.raw_mask_folder, view.image_name + ".png")
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            object_mask = convert_matched_mask(labels, mask)
            object_mask_path = os.path.join(self.associated_mask_folder, view.image_name + ".png")
            cv2.imwrite(object_mask_path, object_mask)

            # Visualize
            if self.visualize:
                visualize_mask = self.visualize_mask_association(object_mask)
                visualize_mask_path = os.path.join(self.visualize_folder, view.image_name + ".png")
                cv2.imwrite(visualize_mask_path, visualize_mask)

            if (view_idx + 1) % 10 == 0:
                print(f"[Semantic] Processed {view_idx + 1}/{len(self.viewpoint_camera)} views, "
                      f"Current masks: {self.num_mask}")

        # Save metadata
        info = {
            "num_mask": self.get_num_mask,
            "raw_mask_folder": self.raw_mask_folder,
            "associated_mask_folder": self.associated_mask_folder,
            "front_percentage": self.front_percentage,
            "overlap_threshold": self.iou_threshold,
            "semantic_threshold": self.semantic_threshold,
            "semantic_weight": self.semantic_weight,
            "combined_threshold": self.combined_threshold,
            "num_patch": self.num_patches,
            "has_language_features": self.has_language_features,
            "use_full_dim": self.use_full_dim,
            "similarity_method": self.similarity_method,
        }
        # Save the semantic feature bank
        if self.has_language_features:
            feature_bank_path = os.path.join(self.associated_mask_folder, "semantic_feature_bank.npy")
            self.semantic_feature_bank.save(feature_bank_path)

        with open(info_path, "w") as info_file:
            json.dump(info, info_file, indent=2)

        print(f"\n[Semantic] Association complete!")
        print(f"[Semantic] Total masks: {self.get_num_mask}")
        print(f"[Semantic] Saved to: {self.associated_mask_folder}")

    def visualize_mask_association(self, object_mask):
        """Visualize mask associations"""
        h, w = object_mask.shape
        visualize_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(self.num_mask):
            # Keep color indices within the palette
            color_idx = i % len(self.random_colors)
            visualize_mask[object_mask == i + 1] = self.random_colors[color_idx]

        visualize_mask = cv2.cvtColor(visualize_mask, cv2.COLOR_RGB2BGR)
        return visualize_mask

    # ========== Geometric projection and mask processing ==========

    def project_gaussian(self, viewpoint):
        """Project Gaussians into the image plane"""
        proj_matrix = viewpoint.full_proj_transform

        p_hom = transformPoint4x4(self.gaussians_xyz, proj_matrix)
        p_hom_z = p_hom[:, 2]

        p_w = 1 / (p_hom[:, 3:] + 1e-8)
        p_proj = p_hom[:, :3] * p_w

        p_proj[:, 0] = ndc2Pixel(p_proj[:, 0], self.image_width)
        p_proj[:, 1] = ndc2Pixel(p_proj[:, 1], self.image_height)
        p_proj = torch.round(p_proj[:, :2]).long()

        # Discard points outside the image
        p_proj_inside_mask = (
            (p_proj[:, 0] >= 0) & (p_proj[:, 0] < self.image_width) &
            (p_proj[:, 1] >= 0) & (p_proj[:, 1] < self.image_height) &
            (p_hom_z > 0)
        )
        p_proj_inside = p_proj[p_proj_inside_mask]
        p_proj_inside_indices = p_proj_inside_mask.nonzero().squeeze()
        p_proj_inside_reverse_mapping = {
            p_proj_inside_indices[i].item(): i
            for i in range(len(p_proj_inside_indices))
        }
        p_proj_flatten = p_proj_inside[:, 0] * self.image_height + p_proj_inside[:, 1]

        projected_gaussian = {
            "p_proj_flatten": p_proj_flatten,
            "p_proj_inside_indices": p_proj_inside_indices,
            "p_proj_inside_reverse_mapping": p_proj_inside_reverse_mapping,
            "p_hom_z": p_hom_z
        }

        return projected_gaussian

    def load_mask(self, viewpoint):
        """Load a mask"""
        mask_path = os.path.join(self.raw_mask_folder, viewpoint.image_name + ".png")
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        binary_mask_mask = mask_id_to_binary_mask(mask)
        binary_mask = torch.tensor(binary_mask_mask, dtype=torch.bool, device=self.device).transpose(1, 2)
        return binary_mask

    def get_patch_front_gaussian_of_mask(self, viewpoint):
        """Get foreground Gaussians for a mask in patches"""
        projected_gaussian = self.project_gaussian(viewpoint)

        p_proj_flatten = projected_gaussian["p_proj_flatten"]
        p_proj_inside_indices = projected_gaussian["p_proj_inside_indices"]
        p_hom_z = projected_gaussian["p_hom_z"]

        mask = self.load_mask(viewpoint)
        assert mask.shape[1] == self.image_width and mask.shape[2] == self.image_height, \
            f"Mask and image have different sizes: {mask.shape} vs ({self.image_width}, {self.image_height})"
        mask_flatten = mask.flatten(start_dim=1)

        front_gaussian = []
        for obj_m in mask_flatten:
            front_gaussian_of_mask = []
            for i in range(self.num_patches):
                for j in range(self.num_patches):
                    patch_m = self.flatten_patch_mask[i, j]
                    m = obj_m & patch_m
                    if m.sum() == 0:
                        continue
                    gaussian_of_mask_inside = m[p_proj_flatten].nonzero().squeeze(-1)
                    if gaussian_of_mask_inside.shape[0] == 0:
                        continue

                    gaussian_of_mask = p_proj_inside_indices[gaussian_of_mask_inside]
                    p_hom_z_of_mask = p_hom_z[gaussian_of_mask]
                    num_front_gaussians = max(int(self.front_percentage * len(gaussian_of_mask)), 1)
                    front_gaussian_of_mask.append(
                        gaussian_of_mask[torch.argsort(p_hom_z_of_mask)][:num_front_gaussians]
                    )

            if len(front_gaussian_of_mask) == 0:
                front_gaussian.append(torch.tensor([], dtype=torch.long, device=self.device))
            else:
                front_gaussian.append(torch.cat(front_gaussian_of_mask))

        return front_gaussian, mask
