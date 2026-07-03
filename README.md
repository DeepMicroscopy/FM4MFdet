# FM4MFdet

Abstract:

```
Pathology foundation models (FMs) are models trained on vast amounts of typically unlabeled data and have been
shown to yield regularized latent spaces that can be used effectively in downstream classification tasks. This
is also true for the classification of mitotic figures vs. other cells. However, it is so far unclear if the
latent space of current FMs provides features that are discriminant and spatially suitably resolved to also serve
as a backbone for dense object detection paradigms. In this work, we investigate this question for common current
pathology FMs (UNI, UNI2-h, Virchow, Virchow2, H-optimus-0, H-optimus-1) and compare their performance against a
fully end-to-end trained baseline based on a ResNet50 architecture. We combine FM backbones with representatives
of single stage, dual stage and self-attention-based detectors (RetinaNet, Faster R-CNN, Deformable DETR respectively)
on the multi-domain MIDOG++ dataset, and on the TUPAC16 dataset as an out-of-domain case. We show that the H-optimus-0
and Virchow models yielded competitive performance, indicating that the latent spaces of current FMs—all trained on
image-level self-supervision—are suitable for direct mitotic figure detection and may be slightly more robuston our
out-of-domain test case.

```
# FM4MFdet

Benchmarking **frozen pathology foundation-model (FM) backbones** for **mitotic figure detection** on MIDOG++ (in-domain) and TUPAC16 (out-of-domain). Each experiment pairs one frozen ViT backbone with one MMDetection head, compared against a fully end-to-end trained ResNet-50 baseline.

> **Abstract.** Pathology foundation models yield regularized latent spaces that work well for classifying mitotic figures vs. other cells, but it is unclear whether those features are discriminant and spatially resolved enough to also back **dense object detection**. We investigate this for current pathology FMs (UNI, UNI2-h, Virchow, Virchow2, H-optimus-0, H-optimus-1), combining each with a single-stage (RetinaNet), dual-stage (Faster R-CNN), and self-attention-based (Deformable DETR) detector on MIDOG++, and on TUPAC16 as an out-of-domain case. H-optimus-0 and Virchow are competitive, indicating that the image-level self-supervised latent spaces of current FMs are suitable for direct mitotic figure detection and may be slightly more robust out-of-domain.

**Backbones:** ResNet-50 (baseline), UNI, UNI2-h, Virchow, Virchow2, H-optimus-0, H-optimus-1

**Heads:** Faster R-CNN, RetinaNet, Deformable DETR

## Layout

```
configs/   MMDetection configs, one per {head}_{backbone}_midogpp.py
scripts/   Data prep, training launchers, whole-slide inference/scoring
src/       Custom backbones, necks, and transforms
```

- **`configs/`** — one config per backbone × head (`faster_rcnn_uni_midogpp.py`, etc.). Naming: `r50` trainable baseline, `r50frozen` frozen baseline, `uni` / `uni2h` / `virchow` / `virchow2` / `h0` / `h1` = FM backbones. Edit the `data/` paths near the bottom of each config to match your machine.
- **`scripts/`** — data prep (`tile_rois.py`, `make_patient_splits.py`, `check_split_leakage.py`), training launchers (`train_{head}_{backbone}_midogpp.py`), and whole-slide inference + scoring (`infer_wsi.py`, `infer_wsi_tupac.py`). See below.
- **`src/custom_mmdet/`** — `backbones/` (six FM ViT wrappers, frozen, loaded from Hugging Face via `timm`), `necks/` (`SimpleFeaturePyramid`), `transforms/` (`HEDStainAugment`). Registered through `custom_imports` in the configs.

## Setup

Needs a CUDA GPU (H100/A100-class for the larger FMs). Dependencies are pinned in `requirements.txt` (PyTorch 2.11 / CUDA 12.8, `mmcv` 2.1, `mmdet` 3.3, `timm`, etc.).

```bash
pip install -r requirements.txt
huggingface-cli login          # some FM weights (UNI, etc.) are gated
```

Run everything from the repo root so `src.custom_mmdet.*` imports and relative `data/` paths resolve.

## Data prep

The detectors train on square COCO patches. Starting from the MIDOG++ ROIs and the supplied `MIDOGpp.json`:

**1. Tile ROIs into COCO patches:**

```bash
python scripts/tile_rois.py \
  --roi-dir   data/rois \
  --ann-file  data/MIDOGpp.json \
  --out-dir   data/coco_annotations/patches_1024 \
  --tile-size 1024        # 1024 for UNI/UNI2-h, else 1008
```

**2. Make patient-stratified splits** (one slide = one patient; patches from a slide never cross splits):

```bash
python scripts/make_patient_splits.py \
  --inputs  data/coco_annotations/patches_1024/midogpp_all.json \
  --out-dir data/coco_annotations/patches_1024 \
  --val-frac 0.15 --test-frac 0.15 --seed 42
```

This writes `midogpp_{train,val,test}.json`.

**3. Verify no leakage:**

```bash
python scripts/check_split_leakage.py \
  --train data/coco_annotations/patches_1024/midogpp_train.json \
  --val   data/coco_annotations/patches_1024/midogpp_val.json \
  --test  data/coco_annotations/patches_1024/midogpp_test.json
```

Expected data layout after prep:

```
data/
├── coco_annotations/patches_1024/midogpp_{train,val,test}.json
└── Datensatz/patches_1024/          # patch images
```

## Train

Run from the repo root (e.g. Faster R-CNN + UNI):

```bash
python scripts/train_faster_rcnn_uni_midogpp.py
```

Swap the script name for any other backbone/head combo. Each launcher loads its matching config, re-checks patient stratification before any GPU work, and trains with an `EarlyStoppingHook` on `coco/bbox_mAP` (`max_epochs` is only an upper bound). Checkpoints and logs go to the config's `work_dir`.

## Whole-slide inference + scoring

Sliding-window inference over whole ROIs with MIDOG-style scoring. The threshold is tuned on **val** and applied once to **test**; detections are cached so thresholds sweep without re-running the model.

```bash
python scripts/infer_wsi.py \
  --config     configs/faster_rcnn_uni_midogpp.py \
  --checkpoint work_dirs/faster_rcnn_uni_midogpp/best.pth \
  --roi-dir    data/rois \
  --ann-file   data/coco_annotations/midogpp_all.json \
  --slides     data/splits/val_test_slides.json \
  --out-dir    results/faster_rcnn_uni \
  --window     1024        # use 1024 for UNI/UNI2-h configs, else 1008
```

TUPAC16 (out-of-domain) uses `scripts/infer_wsi_tupac.py`, which adds `--fixed-thresh` to score at a set threshold instead of sweeping val.

## Notes

- One slide = one patient; splits are patient-stratified and re-checked before training.
- WSI window size must match the config `Resize` and backbone `img_size` (asserted at runtime).
- Detection threshold is tuned on val, applied once to test.
