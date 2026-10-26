#
# Copyright (C) 2024, Gaga
# Gaga research group, https://github.com/weijielyu/Gaga
# All rights reserved.

import os

import cv2
import torch
import numpy as np
from typing import Dict
from argparse import ArgumentParser
from tqdm import tqdm
import json

def get_n_different_colors(n: int) -> np.ndarray:
    np.random.seed(0)
    return np.random.randint(1, 256, (n, 3), dtype=np.uint8)

def visualize_mask(mask: np.ndarray) -> np.ndarray:
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    num_masks = np.max(mask)
    random_colors = get_n_different_colors(num_masks)
    for i in range(num_masks):
        color_mask[mask == i+1] = random_colors[i]
    return color_mask

def get_seg_model(config: Dict, device: str):
    from segment_anything import sam_model_registry
    from automatic_mask_generator import SamAutomaticMaskGenerator

    sam = sam_model_registry[config['sam_encoder_version']](checkpoint=config['sam_checkpoint_path']).to(device=device)
    return SamAutomaticMaskGenerator(sam,
                                    points_per_side=config['sam_num_points_per_side'],
                                    points_per_batch=config['sam_num_points_per_batch'],
                                    pred_iou_thresh=config['sam_pred_iou_threshold'])

def get_sam_mask(auto_sam, image, confidence_threshold):
    mask_data = auto_sam.generate(image)

    pred_masks = mask_data['masks'].float()  # num masks * H * W
    pred_scores = mask_data['iou_preds']  # num masks * num masks

    # select by confidence threshold
    selected_indexes = (pred_scores >= confidence_threshold)
    selected_scores = pred_scores[selected_indexes]
    selected_masks  = pred_masks[selected_indexes]
    _, m_H, m_W = selected_masks.shape
    mask_id = np.zeros((m_H, m_W), dtype=np.uint8)

    # rank
    selected_scores, ranks = torch.sort(selected_scores)
    ranks = ranks + 1
    for index in ranks:
        mask_id[(selected_masks[index-1]==1).cpu().numpy()] = int(index)

    # Compress the masks
    mask_indices = np.unique(mask_id)
    cur_idx = 1
    output_mask = np.zeros((m_H, m_W), dtype=np.uint8)
    for idx in mask_indices:
        if idx == 0:
            continue
        mask = (mask_id == idx)
        if mask.sum() > 0 and (mask.sum() / selected_masks[idx-1].sum()) > 0.1:
            output_mask[mask] = cur_idx
            cur_idx += 1

    return output_mask

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--start", type=int, default=0, help="Start index of images")
    parser.add_argument("--end", type=int, default=-1, help="End index of images (-1 means all)")

    parser.add_argument("--dataset_folder", "-d", required=True, type=str)
    parser.add_argument("--scene", "-s", required=True, type=str)
    parser.add_argument("--image", "-i", default="images", type=str)
    parser.add_argument("--seg_method", "-m", default="sam", choices=["sam"])
    parser.add_argument("--visualize", "-v", action="store_true")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # make output folders
    output_folder = os.path.join(args.dataset_folder, args.scene, "raw_{}_mask".format(args.seg_method))
    os.makedirs(output_folder, exist_ok=True)
    if args.visualize:
        vis_output_folder = os.path.join(args.dataset_folder, args.scene, "raw_{}_mask_vis".format(args.seg_method))
        os.makedirs(vis_output_folder, exist_ok=True)

    # locate image folder
    image_folder = os.path.join(args.dataset_folder, args.scene, args.image)
    assert os.path.exists(image_folder)

    # load config
    config = json.load(open(os.path.join(os.path.dirname(__file__), 'config.json'), 'r'))[args.seg_method]
    config['sam_checkpoint_path'] = os.environ.get('SAM_CHECKPOINT', config['sam_checkpoint_path'])

    # load model
    print("Loading {} model...".format(args.seg_method))
    seg_model = get_seg_model(config, device=device)

    # generate masks
    print("Generating masks...")
    image_list = sorted(name for name in os.listdir(image_folder)
                        if name.lower().endswith(('.png', '.jpg', '.jpeg')))
    if args.end == -1:
        args.end = len(image_list)
    image_list = image_list[args.start:args.end]

    # Check for existing masks (resume capability)
    existing_masks = set()
    if os.path.exists(output_folder):
        existing_masks = set(os.listdir(output_folder))

    already_processed = sum(1 for img in image_list if os.path.splitext(img)[0] + ".png" in existing_masks)
    total_images = len(image_list)

    if already_processed > 0:
        print(f"\n[Resume] Found {already_processed}/{total_images} already processed masks")
        print(f"[Resume] Continuing from image {already_processed + 1}/{total_images}\n")

    for image_name in tqdm(image_list):
        # Skip if mask already exists (for resume capability)
        mask_path = os.path.join(output_folder, os.path.splitext(image_name)[0] + ".png")
        if os.path.exists(mask_path):
            continue

        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = get_sam_mask(seg_model, image, config["confidence_threshold"])

        # mask_path already defined above for resume capability
        cv2.imwrite(mask_path, mask)

        if args.visualize:
            vis_mask = visualize_mask(mask)
            vis_mask_path = os.path.join(vis_output_folder, os.path.splitext(image_name)[0] + ".png")
            cv2.imwrite(vis_mask_path, vis_mask)
