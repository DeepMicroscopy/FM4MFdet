import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None


# The detection configs are single-class. This is the category kept in the
# MIDOG++ source data.
MITOTIC_CATEGORY_ID = 1
OUTPUT_CATEGORY = {"id": 1, "name": "mitotic figure"}

MIN_VISIBLE_AREA_FRAC = 0.30


def tile_origins(extent: int, tile: int, stride: int):
    if extent <= tile:
        return [0]
    origins = list(range(0, extent - tile + 1, stride))
    last = extent - tile
    if origins[-1] != last:
        origins.append(last)
    return origins


def clip_box_to_roi(x1, y1, x2, y2, w, h):
    return (max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2))


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Tile MIDOG++ ROIs into COCO patches for detection.")
    p.add_argument("--roi-dir", required=True,
                   help="Directory containing the ROI .tiff files.")
    p.add_argument("--ann-file", required=True,
                   help="MIDOG++ annotation JSON (the supplied MIDOGpp.json).")
    p.add_argument("--out-dir", required=True,
                   help="Output directory for images/ and midogpp_all.json.")
    p.add_argument("--tile-size", type=int, required=True,
                   help="Square tile size in pixels (1008 or 1024).")
    p.add_argument("--overlap", type=float, default=0.20,
                   help="Fraction of overlap between adjacent tiles "
                        "(default 0.20).")
    p.add_argument("--keep-empty", action="store_true",
                   help="Also write tiles that contain no mitotic figure. "
                        "Off by default to keep the dataset compact; the "
                        "configs set filter_empty_gt=False so empty tiles are "
                        "harmless if included.")
    p.add_argument("--image-format", default="png", choices=["png", "jpg"],
                   help="Output patch image format (default png).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if not 0.0 <= args.overlap < 1.0:
        print("ERROR: --overlap must be in [0, 1).", file=sys.stderr)
        return 1

    roi_dir = Path(args.roi_dir)
    ann_path = Path(args.ann_file)
    out_dir = Path(args.out_dir)
    img_out_dir = out_dir / "images"

    for label, path in (("ROI directory", roi_dir),
                        ("annotation file", ann_path)):
        if not path.exists():
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            return 1

    img_out_dir.mkdir(parents=True, exist_ok=True)

    tile = args.tile_size
    stride = max(1, int(round(tile * (1.0 - args.overlap))))
    print(f"Tile size: {tile} px | overlap: {args.overlap:.0%} | "
          f"stride: {stride} px")

    with open(ann_path, "r", encoding="utf-8") as f:
        src = json.load(f)

    src_images = {im["id"]: im for im in src["images"]}
    ann_by_image = {img_id: [] for img_id in src_images}
    n_imposter = 0
    for a in src["annotations"]:
        if a["category_id"] != MITOTIC_CATEGORY_ID:
            n_imposter += 1
            continue
        ann_by_image[a["image_id"]].append(a)

    print(f"ROIs: {len(src_images)} | mitotic annotations: "
          f"{sum(len(v) for v in ann_by_image.values())} | "
          f"imposters dropped: {n_imposter}")

    out_images = []
    out_annotations = []
    next_img_id = 1
    next_ann_id = 1
    n_missing = 0
    n_empty_skipped = 0

    for img_id, info in tqdm(sorted(src_images.items()), desc="Tiling ROIs"):
        roi_path = roi_dir / info["file_name"]
        if not roi_path.exists():
            n_missing += 1
            continue

        slide_id = Path(info["file_name"]).stem        

        with Image.open(roi_path) as im:
            im = im.convert("RGB")
            roi = np.asarray(im)
        roi_h, roi_w = roi.shape[:2]

        boxes = []
        for a in ann_by_image[img_id]:
            x1, y1, x2, y2 = clip_box_to_roi(*a["bbox"], roi_w, roi_h)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
        boxes = np.array(boxes, dtype=np.float64).reshape(-1, 4)

        ys = tile_origins(roi_h, tile, stride)
        xs = tile_origins(roi_w, tile, stride)

        for row, oy in enumerate(ys):
            for col, ox in enumerate(xs):
                tile_anns = []
                for (x1, y1, x2, y2) in boxes:
                    ix1 = max(x1, ox)
                    iy1 = max(y1, oy)
                    ix2 = min(x2, ox + tile)
                    iy2 = min(y2, oy + tile)
                    iw = ix2 - ix1
                    ih = iy2 - iy1
                    if iw <= 0 or ih <= 0:
                        continue

                    orig_area = (x2 - x1) * (y2 - y1)
                    if orig_area <= 0:
                        continue
                    if (iw * ih) / orig_area < MIN_VISIBLE_AREA_FRAC:
                        continue

                    tile_anns.append([ix1 - ox, iy1 - oy, iw, ih])

                if not tile_anns and not args.keep_empty:
                    n_empty_skipped += 1
                    continue

                patch = roi[oy:oy + tile, ox:ox + tile]
                if patch.shape[0] != tile or patch.shape[1] != tile:
                    canvas = np.zeros((tile, tile, 3), dtype=patch.dtype)
                    canvas[:patch.shape[0], :patch.shape[1]] = patch
                    patch = canvas

                patch_name = f"{slide_id}_x{col}_y{row}.{args.image_format}"
                Image.fromarray(patch).save(img_out_dir / patch_name)

                out_images.append(dict(
                    id=next_img_id,
                    file_name=f"images/{patch_name}",
                    width=tile,
                    height=tile,
                ))

                for (bx, by, bw, bh) in tile_anns:
                    out_annotations.append(dict(
                        id=next_ann_id,
                        image_id=next_img_id,
                        category_id=1,
                        bbox=[round(bx, 2), round(by, 2),
                              round(bw, 2), round(bh, 2)],
                        area=round(bw * bh, 2),
                        iscrowd=0,
                    ))
                    next_ann_id += 1

                next_img_id += 1

    coco = dict(
        info=dict(description="MIDOG++ tiled patches",
                  tile_size=tile, overlap=args.overlap, stride=stride),
        licenses=src.get("licenses", []),
        images=out_images,
        annotations=out_annotations,
        categories=[OUTPUT_CATEGORY],
    )
    out_json = out_dir / "midogpp_all.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(coco, f)

    print("\nTiling complete.")
    print(f"  patches written : {len(out_images)}")
    print(f"  annotations     : {len(out_annotations)}")
    if not args.keep_empty:
        print(f"  empty tiles skipped: {n_empty_skipped} "
              f"(use --keep-empty to include them)")
    if n_missing:
        print(f"  WARNING: {n_missing} ROI file(s) listed in the JSON were "
              f"not found in {roi_dir}", file=sys.stderr)
    print(f"  images dir      : {img_out_dir}")
    print(f"  pooled COCO     : {out_json}")
    print("\nNext step - patient-stratified split:")
    print(f"  python scripts/make_patient_splits.py \\")
    print(f"      --inputs {out_json} \\")
    print(f"      --out-dir {out_dir} \\")
    print(f"      --val-frac 0.15 --test-frac 0.15 --seed 42")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
