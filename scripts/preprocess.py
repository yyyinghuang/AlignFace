"""Prepare raw face images before editing or model inference."""
import argparse
from pathlib import Path
from alignface import FacePreprocessor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--predictor", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.output.exists():
        p.error("Output already exists; choose a new path.")
    image = FacePreprocessor(args.predictor, args.device)(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)


if __name__ == "__main__":
    main()
