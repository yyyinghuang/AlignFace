"""Evaluate a CSV with reference/candidate_a/candidate_b paths and optional labels."""
import argparse
import csv
from pathlib import Path
from alignface import AlignFace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--image-root", type=Path, default=Path("."))
    parser.add_argument("--variant", choices=["overall", "asian", "white"], default="overall")
    parser.add_argument("--backbone", choices=["flip", "clip"], default="flip")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new path.")
    with args.input.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        parser.error("Input CSV is empty.")
    required = {"reference", "candidate_a", "candidate_b"}
    if not required.issubset(rows[0]):
        parser.error("CSV requires reference, candidate_a, candidate_b columns.")
    for row in rows:
        if row.get("label", "") not in ["", "0", "1"]:
            parser.error("Labels must be 1 for A, 0 for B, or empty.")
    model = AlignFace.from_pretrained(args.variant, backbone=args.backbone, device=args.device)
    predictions = []
    correct = []
    for row in rows:
        result = model.compare(args.image_root/row["reference"], args.image_root/row["candidate_a"], alternative=args.image_root/row["candidate_b"])
        pred = int(result.delta > 0)
        if row.get("label", "") != "":
            correct.append(pred == int(row["label"]))
        predictions.append(dict(reference=row["reference"], candidate_a=row["candidate_a"], candidate_b=row["candidate_b"], label=row.get("label", ""), prediction=pred, similarity=result.similarity, delta=result.delta, raw_score=result.raw_score))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    print(f"Evaluated {len(predictions)} triplets.")
    if correct:
        print(f"2AFC agreement: {sum(correct)/len(correct):.6f} ({len(correct)} labeled triplets)")


if __name__ == "__main__":
    main()
