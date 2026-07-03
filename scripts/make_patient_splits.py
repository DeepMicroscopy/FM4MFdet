import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path



_TILE_SUFFIX = re.compile(
    r"""
    (?:                          
        [_-]                     
        (?:                      
            x\d+[_-]y\d+        
          | tile[_-]?\d+         
          | patch[_-]?\d+       
          | \d+[_-]\d+           
          | \d+                  
        )
    )+$                          
    """,
    re.IGNORECASE | re.VERBOSE,
)


def slide_id_from_filename(file_name: str) -> str:
   
    stem = Path(file_name).stem
    stripped = _TILE_SUFFIX.sub("", stem)
    return stripped if stripped else stem



def load_coco(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key in ("images", "annotations", "categories"):
        if key not in data:
            raise ValueError(f"{path} is not valid COCO: missing '{key}'")
    return data


def pool_coco(paths):

    images = []
    annotations = []
    categories = None
    info = None
    licenses = None
    seen_files = {}          # file_name -> new image id (dedupe across inputs)
    next_img_id = 1
    next_ann_id = 1

    for path in paths:
        coco = load_coco(path)

        if categories is None:
            categories = coco["categories"]
            info = coco.get("info", {})
            licenses = coco.get("licenses", [])
        else:
            if _cat_signature(coco["categories"]) != _cat_signature(categories):
                raise ValueError(
                    f"Category mismatch: {path} has different categories than "
                    f"the first input file. Re-splitting requires identical "
                    f"category definitions."
                )

        old_to_new_img = {}
        for img in coco["images"]:
            fname = img["file_name"]
            if fname in seen_files:
                old_to_new_img[img["id"]] = seen_files[fname]
                continue
            new_img = dict(img)
            new_img["id"] = next_img_id
            old_to_new_img[img["id"]] = next_img_id
            seen_files[fname] = next_img_id
            images.append(new_img)
            next_img_id += 1

        for ann in coco["annotations"]:
            if ann["image_id"] not in old_to_new_img:
                continue
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = old_to_new_img[ann["image_id"]]
            annotations.append(new_ann)
            next_ann_id += 1

    return dict(info=info if info is not None else {},
                licenses=licenses if licenses is not None else [],
                images=images, annotations=annotations,
                categories=categories)


def _cat_signature(categories):
    return sorted((c["id"], c["name"]) for c in categories)




def split_slides(slide_ids, val_frac, test_frac, seed):
    """Split a list of unique slide ids into train/val/test.

    Splitting happens over *slides*, never over patches, which is what makes
    the result patient-stratified. A fixed seed makes the split reproducible.
    """
    if not 0.0 <= val_frac < 1.0 or not 0.0 <= test_frac < 1.0:
        raise ValueError("val-frac and test-frac must each be in [0, 1).")
    if val_frac + test_frac >= 1.0:
        raise ValueError("val-frac + test-frac must be < 1.0.")

    slides = sorted(slide_ids)            # sort first -> deterministic
    rng = random.Random(seed)
    rng.shuffle(slides)

    n = len(slides)
    n_test = round(n * test_frac)
    n_val = round(n * val_frac)
    # Guarantee a non-empty train split.
    n_val = min(n_val, max(0, n - n_test - 1))

    test_slides = set(slides[:n_test])
    val_slides = set(slides[n_test:n_test + n_val])
    train_slides = set(slides[n_test + n_val:])

    assert train_slides and not (train_slides & val_slides) \
        and not (train_slides & test_slides) and not (val_slides & test_slides)
    return train_slides, val_slides, test_slides


def subset_coco(coco, keep_image_ids):
    """Build a standalone COCO dict containing only the given image ids."""
    keep = set(keep_image_ids)
    images = [img for img in coco["images"] if img["id"] in keep]
    annotations = [a for a in coco["annotations"] if a["image_id"] in keep]
    return dict(
        info=coco.get("info", {}),
        licenses=coco.get("licenses", []),
        images=images,
        annotations=annotations,
        categories=coco["categories"],
    )


def write_coco(coco, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)



def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Patient-stratified COCO splitting (one slide = one "
                    "patient) for MIDOG++ mitotic figure detection.",
    )
    p.add_argument(
        "--inputs", nargs="+", required=True,
        help="One or more COCO JSON files. Multiple files are pooled and "
             "re-split together.",
    )
    p.add_argument(
        "--out-dir", required=True,
        help="Directory for midogpp_{train,val,test}.json and the manifest.",
    )
    p.add_argument("--val-frac", type=float, default=0.15,
                   help="Fraction of slides for validation (default 0.15).")
    p.add_argument("--test-frac", type=float, default=0.15,
                   help="Fraction of slides for test (default 0.15).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for the slide-level split (default 42).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    input_paths = [Path(p) for p in args.inputs]
    for p in input_paths:
        if not p.exists():
            print(f"ERROR: input file not found: {p}", file=sys.stderr)
            return 1

    out_dir = Path(args.out_dir)

    print(f"Pooling {len(input_paths)} COCO file(s)...")
    coco = pool_coco(input_paths)
    print(f"  pooled images:      {len(coco['images'])}")
    print(f"  pooled annotations: {len(coco['annotations'])}")

    # Group every image by its slide / patient id.
    slide_to_images = defaultdict(list)
    for img in coco["images"]:
        slide = slide_id_from_filename(img["file_name"])
        slide_to_images[slide].append(img["id"])

    n_slides = len(slide_to_images)
    print(f"  distinct slides (patients): {n_slides}")
    if n_slides < 3:
        print("ERROR: need at least 3 slides to form train/val/test splits.",
              file=sys.stderr)
        print("       Check the slide-id convention in "
              "`slide_id_from_filename`.", file=sys.stderr)
        return 1

    train_slides, val_slides, test_slides = split_slides(
        slide_to_images.keys(), args.val_frac, args.test_frac, args.seed,
    )

    split_of = {}
    for s in train_slides:
        split_of[s] = "train"
    for s in val_slides:
        split_of[s] = "val"
    for s in test_slides:
        split_of[s] = "test"

    image_ids = {"train": [], "val": [], "test": []}
    for slide, img_ids in slide_to_images.items():
        image_ids[split_of[slide]].extend(img_ids)

    # Write the three COCO files.
    out_files = {
        "train": out_dir / "midogpp_train.json",
        "val": out_dir / "midogpp_val.json",
        "test": out_dir / "midogpp_test.json",
    }
    for split, path in out_files.items():
        subset = subset_coco(coco, image_ids[split])
        write_coco(subset, path)

    # Write an auditable manifest.
    manifest = dict(
        seed=args.seed,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        grouping="one slide = one patient",
        slide_id_rule="stem with trailing tile/patch/coordinate suffix removed",
        slides=dict(
            train=sorted(train_slides),
            val=sorted(val_slides),
            test=sorted(test_slides),
        ),
        counts=dict(
            slides=dict(
                train=len(train_slides),
                val=len(val_slides),
                test=len(test_slides),
            ),
            images=dict(
                train=len(image_ids["train"]),
                val=len(image_ids["val"]),
                test=len(image_ids["test"]),
            ),
        ),
    )
    write_coco(manifest, out_dir / "split_manifest.json")

    # Leakage self-check: no slide may appear in more than one split.
    assert not (train_slides & val_slides)
    assert not (train_slides & test_slides)
    assert not (val_slides & test_slides)

    print("\nPatient-stratified split written:")
    for split in ("train", "val", "test"):
        print(f"  {split:<5} : {len(image_ids[split]):>6} patches "
              f"from {manifest['counts']['slides'][split]:>4} slides "
              f"-> {out_files[split]}")
    print(f"  manifest : {out_dir / 'split_manifest.json'}")
    print("\nNo slide id crosses splits - patient stratification verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
