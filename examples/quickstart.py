"""Compare bundled similar/contrasting faces or your own image pair."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from alignface import AlignFace


EXAMPLES = Path(__file__).resolve().parent


def verify(result, record, tolerance, label):
    def check(actual, reference, name):
        if not abs(actual-reference) <= tolerance["atol"]+tolerance["rtol"]*abs(reference):
            raise SystemExit(f"Verification failed for {label} {name}: {actual} versus {reference}")
    for name, value in result.concepts.items():
        check(value["similarity"], record["concepts"][name], name)
        check(value["weight"], record["weights"][name], name+" weight")
    if result.level == "face":
        check(result.similarity, record["similarity"], "face similarity")
        check(result.raw_score, record["raw_score"], "raw GAM score")
    elif result.concept is not None:
        check(result.similarity, record["concepts"][result.concept], result.concept)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("face1", nargs="?", help="Omit both paths for the bundled comparison demo")
    p.add_argument("face2", nargs="?")
    p.add_argument("--variant", default="overall", choices=["overall", "asian", "white"])
    p.add_argument("--backbone", default="flip", choices=["flip", "clip"])
    p.add_argument("--concept", default=None, help="For example: eye color")
    p.add_argument("--level", default="face", choices=["face", "concept"])
    p.add_argument("--demo-pair", choices=["ab", "ac"], help="Inspect one demo pair; default compares both")
    p.add_argument("--device", default=None, help="For example: cpu or cuda:0")
    p.add_argument("--details", action="store_true", help="Include raw logits and GAM effects")
    p.add_argument("--verify", action="store_true", help="Check bundled images against recorded CPU outputs")
    args = p.parse_args()
    if (args.face1 is None) != (args.face2 is None):
        p.error("Supply both image paths, or omit both to run the bundled demo.")
    demo = args.face1 is None
    if not demo and (args.verify or args.demo_pair):
        p.error("--verify and --demo-pair require the bundled demo; omit image paths.")
    device = args.device or ("cpu" if args.verify else None)
    if device == "cpu" or (device is None and not torch.cuda.is_available()):
        torch.set_num_threads(min(4, torch.get_num_threads()))
    model = AlignFace.from_pretrained(args.variant, backbone=args.backbone, device=device)
    if demo:
        names = [args.demo_pair] if args.demo_pair else ["ab", "ac"]
        pairs = [(EXAMPLES/"images/face_a.png", EXAMPLES/"images"/f"face_{name[1]}.png") for name in names]
        results = model.compare_batch(pairs, level=args.level, concept=args.concept, details=args.details)
        print("Bundled synthetic face comparison: A is the reference; B is an edited variation; C is a different fictional face.")
    else:
        names = ["custom"]
        pairs = [(Path(args.face1), Path(args.face2))]
        results = [model.compare(*pairs[0], level=args.level, concept=args.concept, details=args.details)]
    for name, result in zip(names, results):
        label = f"A vs {name[1].upper()}" if demo else "Your pair"
        if result.similarity is not None:
            print(f"{label} similarity: {result.similarity:.8f}")
        if not demo or args.demo_pair or result.level == "concept" or args.details:
            print(json.dumps(result.concepts, indent=2))
    if demo and len(results) == 2 and results[0].level == "face":
        order = "A–B > A–C" if results[0].similarity > results[1].similarity else "A–B <= A–C"
        print(f"Observed ranking: {order}")
    if args.verify:
        expected = json.loads((EXAMPLES/"expected_outputs.json").read_text())
        for path in set(path for pair in pairs for path in pair):
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected["image_sha256"][path.name]:
                raise SystemExit(f"Demo image checksum differs: {path.name}")
        record = expected["models"][args.variant][args.backbone]
        for name, result in zip(names, results):
            verify(result, record[name], expected["tolerance"], name)
        print("PASS: demo outputs match the recorded reference within tolerance.")


if __name__ == "__main__":
    main()
