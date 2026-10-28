"""Validate all packaged checkpoints against the untouched research implementation."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import types

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
from PIL import Image
from transformers import CLIPProcessor

from alignface import AlignFace
from alignface.model import ATTRIBUTES, get_attr_text_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("assets/validation.json"))
    args = parser.parse_args()
    source = args.source.resolve()
    torch.set_num_threads(4)
    sys.path.insert(0, str(source))
    dataset = types.ModuleType("dataset")
    dataset.ATTRIBUTES = ATTRIBUTES
    sys.modules["dataset"] = dataset
    original = importlib.import_module("Model_Final_Version")
    records = []
    for variant, folder in [("overall", "Checkpoint_Final_Version_0602"), ("asian", "Checkpoint_Final_Version_Asian_0602"), ("white", "Checkpoint_Final_Version")]:
        for backbone, run in [("flip", "FLIP_P01son_FaceCLIP-base-32"), ("clip", "CLIP_openai_clip-vit-base-patch32")]:
            torch.manual_seed(16)
            package = AlignFace.from_pretrained(variant, backbone=backbone, device="cpu")
            root = source/folder/run
            reference, name, attrs, _ = original.load_stage1_from_checkpoint(root/"stage1/best_model/model.pt", "cpu")
            checkpoint = torch.load(root/"stage1/best_model/model.pt", map_location="cpu", weights_only=True)
            reference.load_state_dict(checkpoint["model_state_dict"], strict=True)
            reference.eval()
            c2 = torch.load(root/"stage2_monotonic_gam_attr_softmax/best_model.pt", map_location="cpu", weights_only=True)
            head = original.build_stage2_head_weight(c2["stage2_head_type"], len(attrs), dropout=c2["dropout"], n_knots=c2["n_knots"], x_min=c2["x_min"], x_max=c2["x_max"])
            head.load_state_dict(c2["model_state_dict"], strict=True)
            head.eval()
            processor = CLIPProcessor.from_pretrained(name, local_files_only=True)
            tokens = processor(text=get_attr_text_list(attrs), return_tensors="pt", padding=True, truncation=True)
            with torch.inference_mode():
                text = reference.encode_text(tokens["input_ids"], tokens["attention_mask"])
                torch.testing.assert_close(package.text, text, rtol=1e-5, atol=1e-6)
                errors = []
                for identity in ["1", "6"]:
                    images = sorted((source/"Dataset"/identity).rglob("*.jpg"))[:3]
                    assert len(images) == 3
                    pixels = []
                    for path in images:
                        with Image.open(path) as image:
                            tensor = processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"]
                        torch.testing.assert_close(package._pixels(path), tensor, rtol=0, atol=0)
                        pixels.append(tensor)
                    refout = reference(*pixels, text)
                    newout = package.stage1(*pixels, package.text)
                    for key in refout:
                        torch.testing.assert_close(newout[key], refout[key], rtol=1e-5, atol=1e-6)
                    raw = head(refout["delta_sim"])
                    delta = 2*torch.sigmoid(raw)-1
                    result = package.compare(images[0], images[1], alternative=images[2], details=True)
                    torch.testing.assert_close(torch.tensor(result.raw_score), raw[0], rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(torch.tensor(result.delta), delta[0], rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(torch.tensor(result.similarity), torch.sigmoid(delta)[0], rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(torch.tensor([v["similarity"] for v in result.concepts.values()]), refout["sim_ar"][0], rtol=1e-5, atol=1e-6)
                    assert abs(sum(v["contribution"] for v in result.concepts.values())-result.raw_score) < 1e-5
                    if identity == "1":
                        pair = package.compare(images[0], images[1], details=True)
                        expected_pair_raw = head(refout["sim_ar"])
                        torch.testing.assert_close(torch.tensor(pair.raw_score), expected_pair_raw[0], rtol=1e-5, atol=1e-6)
                        expected_percentile = float(package.score_calibrator.transform(expected_pair_raw.cpu().numpy())[0])
                        torch.testing.assert_close(torch.tensor(pair.similarity), torch.tensor(expected_percentile), rtol=1e-4, atol=1e-5)
                        assert pair.experimental
                        batch = package.compare_batch([(images[0],images[1]), (images[0],images[2])], batch_size=2)
                        torch.testing.assert_close(torch.tensor(batch[0].similarity), torch.tensor(pair.similarity), rtol=1e-4, atol=1e-5)
                        for item in batch:
                            assert all(set(value)=={"similarity", "weight"} for value in item.concepts.values())
                            assert abs(sum(value["weight"] for value in item.concepts.values())-1)<1e-6
                        for idx, expected in enumerate([refout["sim_ar"][0], refout["sim_br"][0]]):
                            torch.testing.assert_close(torch.tensor([v["similarity"] for v in batch[idx].concepts.values()]), expected, rtol=1e-4, atol=1e-5)
                        eye = package.compare(images[0], images[1], concept="eye color")
                        assert list(eye.concepts) == ["eye color"] and not eye.experimental
                        torch.testing.assert_close(torch.tensor(eye.similarity), refout["sim_ar"][0, attrs.index("eye_colour")], rtol=1e-5, atol=1e-6)
                        all_concepts = package.compare(images[0], images[1], level="concept")
                        assert all_concepts.similarity is None and len(all_concepts.concepts)==20
                        assert abs(sum(v["contribution"] for v in pair.concepts.values())-pair.raw_score)<1e-5
                    errors.append(float((newout["sim_ar"]-refout["sim_ar"]).abs().max()))
                records.append(dict(variant=variant, backbone=backbone, strict_loading=True, real_triplets=2, pixel_values_exact=True, pair_gam_formula=True, batch_order=True, single_concept=True, all_concepts=True, max_concept_error=max(errors), rtol=1e-5, atol=1e-6, batch_rtol=1e-4, batch_atol=1e-5))
                print(records[-1], flush=True)
            del package, reference, checkpoint, head, c2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"torch":torch.__version__, "device":"cpu", "results":records}, indent=2)+"\n")


if __name__ == "__main__":
    main()
