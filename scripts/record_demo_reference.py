"""Record CPU demo outputs for an intentional model release."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import torch
from alignface import AlignFace


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=root/"examples/expected_outputs.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Reference already exists; use a new output path for review.")
    torch.set_num_threads(4)
    paths = [root/"examples/images"/name for name in ["face_a.png", "face_b.png", "face_c.png"]]
    expected = dict(
        description="Synthetic demo regression baseline; not human similarity ground truth.",
        device="cpu",
        environment={"python":platform.python_version(), **{name:importlib.metadata.version(name) for name in ["torch", "transformers", "Pillow", "numpy"]}},
        tolerance={"rtol":1e-4, "atol":1e-5},
        image_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        models={},
    )
    for variant in ["overall", "asian", "white"]:
        expected["models"][variant] = {}
        for backbone in ["flip", "clip"]:
            model = AlignFace.from_pretrained(variant, backbone=backbone, device="cpu", root=root)
            results = model.compare_batch([(paths[0], paths[1]), (paths[0], paths[2])])
            record = {}
            for name, result in zip(["ab", "ac"], results):
                record[name] = dict(similarity=result.similarity, raw_score=result.raw_score, concepts={name:value["similarity"] for name,value in result.concepts.items()}, weights={name:value["weight"] for name,value in result.concepts.items()}, score_type=result.score_type)
            triplet = model.compare(paths[0], paths[1], alternative=paths[2])
            record["triplet"] = dict(delta=triplet.delta, preferred=triplet.preferred, preferred_demo_image="B" if triplet.preferred=="A" else "C")
            record["ab_higher_than_ac"] = results[0].similarity > results[1].similarity
            expected["models"][variant][backbone] = record
            print(f"{variant}/{backbone}: AB={results[0].similarity:.8f}, AC={results[1].similarity:.8f}, AB>AC={record['ab_higher_than_ac']}, triplet preference={triplet.preferred}", flush=True)
            del model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(expected, indent=2)+"\n")


if __name__ == "__main__":
    main()
