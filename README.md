# SG-Group

Code for the manuscript **SG-Group: Semantic-Geometric Fusion for Robust 3D Gaussian Instance Segmentation via Decode-and-Compare**. The manuscript has not yet been published; a paper link and citation will be added when available.

![SG-Group method overview and example applications](abstract.png)

SG-Group combines geometric overlap and semantic similarity to associate 2D instance masks across views. It compresses OpenCLIP features for storage with 3D Gaussians, then decodes them for semantic comparison during association. This repository contains preprocessing, association, grouping, rendering, and evaluation code.

## Environment

Use Linux or WSL with an NVIDIA CUDA GPU, a compatible PyTorch and torchvision installation, and a CUDA compiler for the bundled extensions. From the repository root:

```bash
pip install -r requirements.txt
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

The repository does not currently provide a version lockfile verified by a full GPU run. Record the commands, random seed, `python --version`, `pip freeze`, `nvcc --version`, and `nvidia-smi` when reproducing an experiment.

## Inputs and model weights

- A COLMAP scene with `images/` and `sparse/0/`. Images may be PNG, JPG, or JPEG; image names must remain unique after removing their extensions. The current instance-label pipeline does not support Blender scenes.
- A trained RGB Gaussian model for the same scene, containing `point_cloud/iteration_<RGB_ITER>/point_cloud.ply`. Training this RGB model is outside this repository. The scene and RGB model directories must be writable because intermediate features and masks are saved there.
- [SAM ViT-H checkpoint](https://github.com/facebookresearch/segment-anything#model-checkpoints), saved at `weight/sam_vit_h_4b8939.pth` or another path supplied through `SAM_CHECKPOINT`.
- [LAION OpenCLIP ViT-B-16 `laion2b_s34b_b88k` weights](https://huggingface.co/laion/CLIP-ViT-B-16-laion2B-s34B-b88K). `preprocess_semantic.py` requests this model through OpenCLIP and downloads it on first use if it is not already cached. No fixed `.pth` path is required or assumed by this code.

Dataset download instructions are available for [LERF-Mask](https://github.com/lkeab/gaussian-grouping/blob/main/docs/dataset.md) and [Replica and ScanNet](https://github.com/weijielyu/Gaga/blob/main/USAGE.md#datasets). [MUSeg](https://doi.org/10.6084/m9.figshare.28749098) is also available, but its published RGB-D files are not a ready-to-run COLMAP scene for this pipeline. Prepare a scene in the COLMAP layout above before running SG-Group.

## Run SG-Group

Run these commands from the repository root in the same Bash session. Set the paths and iterations for your scene:

```bash
SCENE=/path/to/scene
RGB_MODEL=/path/to/rgb_model
OUT=/path/to/output
RGB_ITER=30000
SG_ITER=10000
EVAL_SPLIT=interval
EVAL_HOLDOUT=8
export SAM_CHECKPOINT=/path/to/sam_vit_h_4b8939.pth
mkdir -p "$OUT"
```

With `EVAL_SPLIT=interval`, every eighth image is held out for testing. For Replica scenes whose files use `train_rgb_*` and `test_rgb_*` names, set `EVAL_SPLIT=replica` instead. Then run each stage in order:

```bash
# 1. Extract 512-dimensional OpenCLIP features for SAM regions.
python preprocess_semantic.py --dataset_path "$SCENE" --sam_ckpt_path "$SAM_CHECKPOINT"

# 2. Train an autoencoder and compress the region features to three dimensions.
python autoencoder/train_autoencoder.py --dataset_path "$SCENE" --output_path "$OUT/autoencoder" --epochs 100
python autoencoder/compress_features.py --dataset_path "$SCENE" --encoder_path "$OUT/autoencoder/encoder.pth"

# 3. Attach the compressed features to the RGB Gaussian model.
python train_semantic.py -s "$SCENE" -m "$RGB_MODEL" \
  --start_checkpoint_iter "$RGB_ITER" --language_features_path "$SCENE/language_features_dim3" \
  --eval --eval_split "$EVAL_SPLIT" --eval_holdout "$EVAL_HOLDOUT"

# 4. Generate 2D SAM masks and associate them across views.
python mask/get_raw_mask.py --dataset_folder "$(dirname "$SCENE")" \
  --scene "$(basename "$SCENE")" --seg_method sam
python associate_semantic.py -s "$SCENE" -m "$RGB_MODEL" --iteration "$RGB_ITER" \
  --seg_method sam --use_full_dim --semantic_weight 0.2 \
  --combined_threshold 0.3 --autoencoder_path "$OUT/autoencoder/encoder.pth" \
  --eval --eval_split "$EVAL_SPLIT" --eval_holdout "$EVAL_HOLDOUT"

# 5. Train the 3D grouping model and render held-out views.
python lift.py -s "$SCENE" -m "$OUT/final" --trained_model_path "$RGB_MODEL" \
  --object_path sam_mask_semantic --rgb_iteration "$RGB_ITER" --iterations "$SG_ITER" \
  --save_iterations "$SG_ITER" --checkpoint_iterations "$SG_ITER" \
  --test_iterations "$SG_ITER" --eval --eval_split "$EVAL_SPLIT" --eval_holdout "$EVAL_HOLDOUT"
python render.py -m "$OUT/final" --iteration "$SG_ITER" --skip_train
```

The association command compares decoded 512-dimensional features with a semantic weight of 0.2 and a combined-score threshold of 0.3. The trained model is saved under `$OUT/final/point_cloud/iteration_<SG_ITER>/`. Intermediate files are written into the scene, RGB model, and output directories. When changing the split or association settings, use clean scene and RGB model copies and a new output directory to avoid reusing old intermediates.

## Evaluation

Rendered instance-label PNGs are saved under `$OUT/final/test/ours_<SG_ITER>/objects_test/`. With matching ground-truth instance PNGs, run:

```bash
python eval.py --gt_masks /path/to/ground_truth_masks \
  --pred_masks /path/to/output/final/test/ours_10000/objects_test
```

The evaluator reports mean IoU, precision, and recall at IoU > 0.5. Record the data split, ground-truth source, model iteration, and evaluation command when comparing with the paper.

## License and attribution

Original SG-Group contributions are licensed under [MIT](LICENSE.md) by the named copyright holders. This grant applies only to contributions they have the right to license. Upstream source and license details are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); third-party components retain their own terms, including research-use restrictions on bundled CUDA code.
