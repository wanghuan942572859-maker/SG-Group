#!/usr/bin/env python3
"""
Attach compressed CLIP features to vanilla 3D Gaussians
Attach compressed CLIP features to a trained RGB 3DGS model
"""

import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path

from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams
from utils.general_utils import safe_state
from utils.semantic_projection import segment_features_for_gaussians


def attach_language_features(dataset, pipeline, args):
    """
    Lift 2D CLIP features onto 3D Gaussians

    Steps:
    1. Load the trained RGB model
    2. Iterate over views and render visible Gaussians
    3. Assign view CLIP features to visible Gaussians
    4. Aggregate features across views
    5. Save the model with semantic features
    """

    print("\n" + "="*80)
    print("Attaching CLIP Features to 3D Gaussians")
    print("="*80 + "\n")

    # Check that compressed features exist
    language_features_path = Path(args.language_features_path)
    if not language_features_path.exists():
        print(f"[Error] Language features not found at: {language_features_path}")
        return False

    feature_files = sorted(language_features_path.glob("*_f.npy"))
    if len(feature_files) == 0:
        print(f"[Error] No _f.npy files found in: {language_features_path}")
        return False

    print(f"[Info] Found {len(feature_files)} compressed feature files")

    # Load the trained RGB model
    print(f"\n[Step 1/4] Loading vanilla RGB Gaussian model...")
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.start_checkpoint_iter, shuffle=False)

    num_gaussians = gaussians.get_xyz.shape[0]
    print(f"[Info] Loaded {num_gaussians} Gaussians")

    # Initialize feature storage
    feature_dim = 3  # Compressed feature dimension
    accumulated_features = torch.zeros(num_gaussians, feature_dim, device="cuda")
    feature_counts = torch.zeros(num_gaussians, 1, device="cuda")

    # Get training cameras
    viewpoint_cameras = scene.getTrainCameras()
    print(f"[Info] Processing {len(viewpoint_cameras)} training views")

    # Background color
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print(f"\n[Step 2/4] Projecting CLIP features to 3D Gaussians...")

    # Iterate over views
    for viewpoint_cam in tqdm(viewpoint_cameras, desc="Processing views"):
        # Load compressed CLIP features for this view
        image_name = viewpoint_cam.image_name
        feature_file = language_features_path / f"{image_name}_f.npy"
        seg_file = language_features_path / f"{image_name}_s.npy"

        if not feature_file.is_file() or not seg_file.is_file():
            raise FileNotFoundError(f"Missing feature or segmentation map for {image_name}")

        # Load features with shape (N_segments, 3)
        segment_features = torch.from_numpy(np.load(feature_file)).float().cuda()
        if segment_features.ndim != 2 or segment_features.shape[1] != feature_dim:
            raise ValueError(f"Invalid feature shape for {image_name}: {segment_features.shape}")

        # Load the segmentation map (scales, H, W), using the first scale
        seg_maps = np.load(seg_file)
        if seg_maps.ndim != 3 or seg_maps.shape[0] != 1:
            raise ValueError(f"Invalid segmentation map shape for {image_name}: {seg_maps.shape}")
        seg_map = torch.from_numpy(seg_maps[0]).long().cuda()

        # Render to find visible Gaussians, then project their centers onto the segmentation map
        with torch.no_grad():
            render_pkg = render(viewpoint_cam, gaussians, pipeline, background)

            indices, projected_features = segment_features_for_gaussians(
                gaussians.get_xyz, viewpoint_cam.full_proj_transform,
                render_pkg["radii"], seg_map, segment_features,
            )
            accumulated_features[indices] += projected_features
            feature_counts[indices] += 1

    print(f"\n[Step 3/4] Aggregating features across views...")

    # Average the features of each Gaussian across views
    mask = feature_counts > 0
    if not mask.any():
        raise RuntimeError("No Gaussian projected into a language-feature segment")
    aggregated_features = torch.zeros_like(accumulated_features)
    aggregated_features[mask.squeeze()] = accumulated_features[mask.squeeze()] / feature_counts[mask.squeeze()]

    # Use zero vectors for Gaussians unseen from all views
    num_unobserved = (~mask.squeeze()).sum().item()
    if num_unobserved > 0:
        print(f"[Warning] {num_unobserved} Gaussians were not observed in any view")

    # L2-normalize features
    norms = aggregated_features.norm(dim=1, keepdim=True)
    aggregated_features = aggregated_features / (norms + 1e-8)

    print(f"[Info] Feature aggregation complete")
    print(f"  - Observed Gaussians: {mask.sum().item()} / {num_gaussians}")
    print(f"  - Feature dimension: {feature_dim}")

    # Save the model with semantic features
    print(f"\n[Step 4/4] Saving model with language features...")

    output_dir = Path(dataset.model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save semantic features
    feature_save_path = output_dir / "language_features_3d.pt"
    torch.save({
        'language_features': aggregated_features.cpu(),
        'num_gaussians': num_gaussians,
        'feature_dim': feature_dim,
        'num_observed': mask.sum().item()
    }, feature_save_path)

    print(f"[Success] Language features saved to: {feature_save_path}")

    # Also save a copy in the point-cloud directory
    iteration = args.start_checkpoint_iter if args.start_checkpoint_iter > 0 else 30000
    point_cloud_path = output_dir / "point_cloud" / f"iteration_{iteration}"
    point_cloud_path.mkdir(parents=True, exist_ok=True)

    feature_pc_path = point_cloud_path / "language_features_3d.pt"
    torch.save({
        'language_features': aggregated_features.cpu(),
        'num_gaussians': num_gaussians,
        'feature_dim': feature_dim,
        'num_observed': mask.sum().item()
    }, feature_pc_path)

    print(f"[Success] Language features also saved to: {feature_pc_path}")

    print("\n" + "="*80)
    print("Feature Attachment Complete!")
    print("="*80 + "\n")

    return True


if __name__ == "__main__":
    parser = ArgumentParser(description="Attach CLIP features to Gaussians")

    # Use the standard ModelParams and PipelineParams
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)

    # Add custom arguments
    parser.add_argument("--language_features_path", type=str, required=True,
                       help="Path to compressed CLIP features (language_features_dim3)")
    parser.add_argument("--start_checkpoint_iter", type=int, default=30000,
                       help="Iteration of RGB model to load")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    print("Parameters:")
    print(f"  Source path: {args.source_path}")
    print(f"  Model path: {args.model_path}")
    print(f"  Language features: {args.language_features_path}")
    print(f"  Checkpoint iteration: {args.start_checkpoint_iter}")

    # Initialize system state
    safe_state(args.quiet)

    # Attach the features
    with torch.no_grad():
        success = attach_language_features(
            model.extract(args),
            pipeline.extract(args),
            args
        )

    if success:
        print("\n[Done] You can now use this model for semantic association!")
        print("Next step: Run associate_semantic.py with this model")
    else:
        print("\n[Failed] Feature attachment failed!")
        exit(1)
