"""Fit pair-score percentiles using only each final model's training split."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from alignface import AlignFace
from alignface.calibration import PercentileCalibration
from alignface.preprocessing import load_rgb


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("batch-size must be positive")
    torch.set_num_threads(4)
    source = args.source.resolve()
    root = Path(__file__).resolve().parent.parent
    splits = dict(overall="dataset_loader/final0602_aligned_split/train_split.csv", asian="dataset_loader/splits_asian_0602/train_split.csv", white="dataset_loader/train_split.csv")
    provenance = json.loads((root/"assets/provenance.json").read_text())
    report = []
    for variant, split in splits.items():
        split_path = source/split
        rows = list(csv.DictReader(split_path.open()))
        pairs = []
        for row in rows:
            face = row["Face idx R (Original)"]
            directory = source/"Dataset"/row["Target ID"]/face
            pairs.extend((directory/(face+".jpg"), directory/(row[col]+".jpg")) for col in ["Face idx A (Edited)", "Face idx B (Edited)"])
        images = sorted(set(path for pair in pairs for path in pair))
        for backbone in ["flip", "clip"]:
            print(f"Encoding {variant}/{backbone}: {len(images)} images, {len(pairs)} training pairs", flush=True)
            model = AlignFace.from_pretrained(variant, backbone=backbone, device="cpu", calibration="sigmoid")
            embeddings = {}
            for start in range(0, len(images), args.batch_size):
                paths = images[start:start+args.batch_size]
                pixels = model.processor(images=[load_rgb(path) for path in paths], return_tensors="pt")["pixel_values"]
                feats = model.stage1.encode_image(pixels)
                embeddings.update(zip(paths, feats))
            scores = []
            for start in range(0, len(pairs), args.batch_size):
                chunk = pairs[start:start+args.batch_size]
                refs = torch.stack([embeddings[a] for a,b in chunk])
                candidates = torch.stack([embeddings[b] for a,b in chunk])
                b, n = len(chunk), len(model.attributes)
                pair = model.stage1.build_pair(candidates, refs).unsqueeze(1).expand(b,n,-1)
                lt = model.stage1.text_learnable_tokens.unsqueeze(0).expand(b,n,-1,-1)
                prompt = model.text.unsqueeze(0).expand(b,n,-1).unsqueeze(2)
                text = torch.cat([lt,prompt],dim=2).reshape(b,n,-1)
                _, logits = model.stage1.head(pair,text)
                sim, _ = model.stage1._normalized_sim_from_logits(logits,logits)
                raw = model.stage2(sim)
                scores.extend(raw.tolist())
            # Verify the cached path against the public image-input path.
            check = model.compare_batch(pairs[:2], batch_size=2)
            np.testing.assert_allclose([r.raw_score for r in check], scores[:2], rtol=1e-4, atol=1e-5)
            calibration = PercentileCalibration.fit(scores)
            mapped = calibration.transform(scores)
            order = np.argsort(scores, kind="stable")
            assert np.array_equal(order, np.argsort(mapped, kind="stable"))
            assert np.all(np.diff(calibration.transform(calibration.scores)) > 0)
            data = calibration.to_dict()
            weights = [item for item in provenance if item["destination"].startswith(f"checkpoints/{variant}/{backbone}/")]
            data["reference"] = dict(variant=variant, backbone=backbone, split=split, split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(), split_type="train_only", triplets=len(rows), pairs=len(pairs), unique_scores=len(calibration.scores), checkpoint_sha256={Path(item["destination"]).name:item["sha256"] for item in weights}, score_definition="GAM(stage1_pair_similarities), before any link function")
            destination = root/f"checkpoints/{variant}/{backbone}/calibration.json"
            destination.write_text(json.dumps(data,indent=2)+"\n")
            record = dict(variant=variant, backbone=backbone, train_pairs=len(pairs), rank_order_exact=True, distinct_reference_scores_strictly_ordered=True, percentile_quantiles=np.quantile(mapped,[.1,.5,.9]).tolist(), cached_path_matches_image_api=True)
            report.append(record)
            print(record, flush=True)
            del model, embeddings
    (root/"assets/calibration_validation.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
