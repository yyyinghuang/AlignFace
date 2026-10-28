"""Check tensor/file parity and downstream gradients for every packaged model."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from alignface import AlignFace


def main():
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parent.parent
    a,b,c = [root/"examples/images"/f"face_{name}.png" for name in "abc"]
    records = []
    for variant in ["overall","asian","white"]:
        for backbone in ["flip","clip"]:
            model = AlignFace.from_pretrained(variant,backbone=backbone,device="cpu")
            refs = torch.cat([model._pixels(a),model._pixels(a)])
            candidates = torch.cat([model._pixels(b),model._pixels(c)]).requires_grad_(True)
            scores = model.score_tensors(refs,candidates)
            file_scores = model.compare_batch([(a,b),(a,c)])
            torch.testing.assert_close(scores,torch.tensor([r.similarity for r in file_scores]),rtol=1e-4,atol=1e-5)
            (1-scores.mean()).backward()
            assert candidates.grad is not None and bool(torch.isfinite(candidates.grad).all())
            assert float(candidates.grad.abs().sum())>0
            for selection in [{"concept":"eye color"},{"level":"concept"},{"calibration":"sigmoid"},{"calibration":"raw"}]:
                x = candidates.detach().clone().requires_grad_(True)
                values = model.score_tensors(refs,x,**selection)
                assert values.shape == ((2,20) if selection.get("level")=="concept" else (2,))
                values.sum().backward()
                assert bool(torch.isfinite(x.grad).all()) and float(x.grad.abs().sum())>0
            # Test differentiable resizing/normalization before the metric.
            with Image.open(c) as image:
                pixels = torch.from_numpy(np.asarray(image.resize((256,256))).copy()).permute(2,0,1).float()/255
            images = pixels.unsqueeze(0).requires_grad_(True)
            prepared = model.prepare_tensors(images)
            assert prepared.shape==(1,3,224,224)
            model.score_tensors(model.prepare_tensors(images.detach()),prepared,calibration="sigmoid").sum().backward()
            assert bool(torch.isfinite(images.grad).all()) and float(images.grad.abs().sum())>0
            assert all(not p.requires_grad and p.grad is None for module in [model.stage1,model.stage2] for p in module.parameters())
            record = dict(variant=variant,backbone=backbone,tensor_file_parity=True,percentile_input_gradients=True,sigmoid_input_gradients=True,concept_input_gradients=True,resize_input_gradients=True,metric_weights_frozen=True)
            records.append(record)
            print(record,flush=True)
            del model
    (root/"assets/training_validation.json").write_text(json.dumps(dict(device="cpu",torch=torch.__version__,results=records),indent=2)+"\n")


if __name__ == "__main__":
    main()
