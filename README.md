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
# COMPAYL / FM4MFdet

Benchmarking **frozen pathology foundation-model backbones** for **mitotic figure detection** on MIDOG++ and TUPAC16. Each experiment = one frozen ViT backbone + one MMDetection head.

**Backbones:** ResNet-50 (baseline), UNI, UNI2-h, Virchow, Virchow2, H-optimus-0, H-optimus-1

**Heads:** Faster R-CNN, RetinaNet, Deformable DETR

## Layout

```
configs/   MMDetection configs, one per {head}_{backbone}_midogpp.py
scripts/   Training launchers + whole-slide inference/scoring
src/       Custom backbones, necks, and transforms
```

Expected data layout:

```
data/
├── coco_annotations/patches_1024/midogpp_{train,val,test}.json
└── Datensatz/patches_1024/          # patch images
```

## Run

Run from the repo root.

**Train** (e.g. Faster R-CNN + UNI):

```bash
python scripts/train_faster_rcnn_uni_midogpp.py
```

Swap the script name for any other combo.

**Whole-slide inference + scoring:**

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

## Notes

- One slide = one patient; please ensure that the splits are patient-stratified (enforced before training).
- WSI window size must match the config `Resize` and backbone `img_size` (asserted).
- Detection threshold is tuned on val, applied once to test.
