import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Reuse the exact same slide-id rule as the splitter so the two never diverge.
from make_patient_splits import slide_id_from_filename


def slides_in(path: Path):
    """Return {slide_id: [file_name, ...]} for one COCO file."""
    with open(path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    slides = defaultdict(list)
    for img in coco["images"]:
        slides[slide_id_from_filename(img["file_name"])].append(
            img["file_name"])
    return slides


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Check a COCO train/val/test triple for patient leakage.",
    )
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--test", required=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    paths = {"train": Path(args.train),
             "val": Path(args.val),
             "test": Path(args.test)}
    for name, path in paths.items():
        if not path.exists():
            print(f"ERROR: {name} file not found: {path}", file=sys.stderr)
            return 1

    slides = {name: slides_in(path) for name, path in paths.items()}
    slide_sets = {name: set(s.keys()) for name, s in slides.items()}

    for name, s in slides.items():
        print(f"{name:<5}: {sum(len(v) for v in s.values()):>6} patches "
              f"from {len(s):>4} slides")

    leaks = []
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = slide_sets[a] & slide_sets[b]
        if shared:
            leaks.append((a, b, sorted(shared)))

    if leaks:
        print("\nLEAKAGE DETECTED - the following slides cross splits:")
        for a, b, shared in leaks:
            print(f"  {a} <-> {b}: {len(shared)} shared slide(s)")
            for slide in shared:
                print(f"    - {slide}")
        print("\nThis split is NOT patient-stratified. Re-run "
              "make_patient_splits.py.", file=sys.stderr)
        return 1

    total = len(slide_sets["train"] | slide_sets["val"] | slide_sets["test"])
    print(f"\nOK: {total} distinct slides, none crossing splits.")
    print("Split is patient-stratified (one slide = one patient).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
