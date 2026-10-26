"""
SG-Group semantic feature preprocessing.
Generate language features with SAM and OpenCLIP.
"""

import os
import random
import argparse
import numpy as np
import torch
from tqdm import tqdm
from segment_anything import sam_model_registry
from mask.automatic_mask_generator import SamAutomaticMaskGenerator
import cv2

from dataclasses import dataclass, field
from typing import Tuple, Type

import torchvision
from torch import nn

try:
    import open_clip
except ImportError:
    print("[Error] open_clip is not installed!")
    print("Install it with: pip install open-clip-torch")
    raise


@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)


class OpenCLIPNetwork(nn.Module):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])

        print(f"[CLIP] Loading {config.clip_model_type} model...")
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,
            pretrained=self.config.clip_model_pretrained,
            precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims
        print(f"[CLIP] Model loaded successfully!")

        self.positives = self.config.positives
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims

    def encode_image(self, input):
        """Encode an image into CLIP features"""
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)


def get_seg_img(mask, image):
    """Extract the masked image region"""
    image = image.copy()
    image[mask['segmentation'] == 0] = np.array([0, 0, 0], dtype=np.uint8)
    x, y, w, h = np.int32(mask['bbox'])
    seg_img = image[y:y+h, x:x+w, ...]
    return seg_img


def pad_img(img):
    """Pad the image to a square"""
    h, w, _ = img.shape
    l = max(w, h)
    pad = np.zeros((l, l, 3), dtype=np.uint8)
    if h > w:
        pad[:, (h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad


def filter_masks(keep: torch.Tensor, masks_result):
    """Filter masks using the keep tensor"""
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep:
            result_keep.append(m)
    return result_keep


def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2):
    """
    Mask non-maximum suppression
    """
    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]

    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)

    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou

            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)

    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr

    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True

    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx


def masks_update(*args, **kwargs):
    """Update masks and remove redundant ones"""
    masks_new = ()
    for masks_lvl in args:
        seg_pred = torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter_masks(keep_mask_nms, masks_lvl)

        masks_new += (masks_lvl,)
    return masks_new


def sam_encoder(image, mask_generator):
    """
    Generate single-scale masks with SAM.
    """
    image = cv2.cvtColor(image[0].permute(1, 2, 0).numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)

    # Generate single-scale masks.
    mask_data = mask_generator.generate(image)

    # Convert MaskData to a standard list
    masks = []
    if len(mask_data['masks']) > 0:
        pred_masks = mask_data['masks']  # [N, H, W] tensor
        pred_boxes = mask_data['boxes']  # [N, 4] tensor (xyxy format)

        for i in range(len(pred_masks)):
            # Convert boxes from xyxy to xywh
            x0, y0, x1, y1 = pred_boxes[i].cpu().numpy()
            bbox = [x0, y0, x1 - x0, y1 - y0]  # xywh

            masks.append({
                'segmentation': pred_masks[i].cpu().numpy(),
                'bbox': bbox
            })

    def mask2segmap(masks, image):
        seg_img_list = []
        seg_map = -np.ones(image.shape[:2], dtype=np.int32)
        for i in range(len(masks)):
            mask = masks[i]
            seg_img = get_seg_img(mask, image)
            pad_seg_img = cv2.resize(pad_img(seg_img), (224, 224))
            seg_img_list.append(pad_seg_img)

            seg_map[masks[i]['segmentation']] = i

        if len(seg_img_list) == 0:
            # Return empty outputs if there are no masks
            return torch.zeros((0, 3, 224, 224)).to('cuda'), seg_map

        seg_imgs = np.stack(seg_img_list, axis=0)
        seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0, 3, 1, 2) / 255.0).to('cuda')

        return seg_imgs, seg_map

    seg_images, seg_map = mask2segmap(masks, image)

    return seg_images, seg_map


def embed_clip_sam_tiles(image, sam_encoder_fn, clip_model):
    """Extract CLIP features from single-scale SAM masks"""
    aug_imgs = torch.cat([image])
    seg_images, seg_map = sam_encoder_fn(aug_imgs)

    # Single-scale processing
    if seg_images.shape[0] == 0:
        # No masks
        return torch.zeros((0, 512)).half(), seg_map

    tiles = seg_images.to("cuda")
    with torch.no_grad():
        clip_embed = clip_model.encode_image(tiles)
    clip_embed /= clip_embed.norm(dim=-1, keepdim=True)
    clip_embed = clip_embed.detach().cpu().half()

    return clip_embed, seg_map


def save_numpy(save_path, data):
    """Save NumPy arrays"""
    save_path_s = save_path + '_s.npy'
    save_path_f = save_path + '_f.npy'
    np.save(save_path_s, data['seg_maps'].numpy())
    np.save(save_path_f, data['feature'].numpy())
    print(f"[Saved] {save_path_f} and {save_path_s}")


def create_language_features(img_folder, data_list, save_folder, sam_encoder_fn, clip_model, resolution):
    """Process one image at a time, keeping image names paired with pixels."""
    for data_path in tqdm(data_list, desc="Embedding images"):
        base_name = os.path.splitext(data_path)[0]
        save_path = os.path.join(save_folder, base_name)
        if os.path.isfile(save_path + '_f.npy') and os.path.isfile(save_path + '_s.npy'):
            continue

        image = cv2.imread(os.path.join(img_folder, data_path))
        if image is None:
            raise ValueError(f"Could not read image: {data_path}")
        orig_h, orig_w = image.shape[:2]
        scale = max(orig_h / 1080, 1) if resolution == -1 else orig_w / resolution
        image = cv2.resize(image, (int(orig_w / scale), int(orig_h / scale)))
        img = torch.from_numpy(image).permute(2, 0, 1)
        try:
            img_embed, seg_map = embed_clip_sam_tiles(img.unsqueeze(0), sam_encoder_fn, clip_model)
        except Exception as e:
            raise RuntimeError(f"Failed to process image {data_path}") from e
        if img_embed.shape[0] == 0:
            raise ValueError(f"SAM found no segments in {data_path}")
        seg_map_tensor = torch.from_numpy(seg_map)[None, ...]
        save_numpy(save_path, {'feature': img_embed, 'seg_maps': seg_map_tensor})

    print(f"[Done] Features saved for {len(data_list)} images to {save_folder}")


def seed_everything(seed_value):
    """Set random seeds"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Generate semantic language features")
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to the dataset')
    parser.add_argument('--resolution', type=int, default=-1,
                       help='Image resolution (-1 for auto)')
    parser.add_argument('--sam_ckpt_path', type=str, default="weight/sam_vit_h_4b8939.pth",
                       help='Path to SAM checkpoint')
    args = parser.parse_args()

    seed_everything(42)
    torch.set_default_dtype(torch.float32)

    print("=" * 80)
    print("Semantic Feature Preprocessing (SAM + OpenCLIP)")
    print("=" * 80)
    print(f"\nDataset: {args.dataset_path}")
    print(f"SAM Checkpoint: {args.sam_ckpt_path}\n")

    # Validate input paths
    dataset_path = args.dataset_path
    sam_ckpt_path = args.sam_ckpt_path
    img_folder = os.path.join(dataset_path, 'images')

    if not os.path.isdir(img_folder):
        raise FileNotFoundError(f"Images folder not found: {img_folder}")

    if not os.path.isfile(sam_ckpt_path):
        raise FileNotFoundError(f"SAM checkpoint not found: {sam_ckpt_path}")

    # Load the image list
    data_list = sorted(name for name in os.listdir(img_folder)
                       if name.lower().endswith(('.png', '.jpg', '.jpeg')))
    if not data_list:
        raise ValueError(f"No supported images in {img_folder}")
    print(f"[Found] {len(data_list)} images")

    # Initialize the OpenCLIP model
    print("\n[Step 1/3] Initializing OpenCLIP model...")
    clip_model = OpenCLIPNetwork(OpenCLIPNetworkConfig())

    # Initialize SAM
    print("\n[Step 2/3] Initializing SAM model...")
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )
    print("[SAM] Model initialized!")

    # Create the output directory
    save_folder = os.path.join(dataset_path, 'language_features')
    os.makedirs(save_folder, exist_ok=True)

    # Generate features
    print("\n[Step 3/3] Generating language features...")
    sam_encoder_fn = lambda x: sam_encoder(x, mask_generator)
    create_language_features(img_folder, data_list, save_folder, sam_encoder_fn,
                             clip_model, args.resolution)

    print("\n" + "=" * 80)
    print("Preprocessing Complete!")
    print(f"Features saved to: {save_folder}")
    print("=" * 80)


if __name__ == '__main__':
    main()
