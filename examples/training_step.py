"""Demonstrate a downstream optimizer step through a frozen AlignFace metric."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from alignface import AlignFace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["overall","asian","white"], default="overall")
    parser.add_argument("--backbone", choices=["flip","clip"], default="flip")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(7)
    model = AlignFace.from_pretrained(args.variant, backbone=args.backbone, device=args.device)
    directory = Path(__file__).resolve().parent/"images"
    def load(name):
        with Image.open(directory/name) as image:
            array = np.asarray(image.convert("RGB").resize((256,256),Image.Resampling.BICUBIC)).copy()
        return torch.from_numpy(array).permute(2,0,1).float()/255
    reference = load("face_a.png").unsqueeze(0).repeat(2,1,1,1).to(args.device)
    source = torch.stack([load("face_b.png"),load("face_c.png")]).to(args.device)
    # This small color adapter stands in for the user's image-generating model.
    generator = torch.nn.Conv2d(3,3,1).to(args.device)
    with torch.no_grad():
        generator.weight.copy_(torch.eye(3,device=args.device).reshape(3,3,1,1))
        generator.bias.zero_()
    optimizer = torch.optim.SGD(generator.parameters(),lr=1e-3)
    generated = torch.sigmoid(generator(source))
    score = model.score_tensors(model.prepare_tensors(reference),model.prepare_tensors(generated),calibration="sigmoid")
    loss = (1-score).mean()
    optimizer.zero_grad()
    loss.backward()
    norm = sum(float(p.grad.square().sum()) for p in generator.parameters())**.5
    assert norm>0 and all(torch.isfinite(p.grad).all() for p in generator.parameters())
    assert all(p.grad is None and not p.requires_grad for module in [model.stage1,model.stage2] for p in module.parameters())
    before = generator.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before,generator.weight.detach())
    print(f"PASS: batch={score.numel()}, loss={loss.item():.6f}, downstream gradient norm={norm:.6g}")
    print("The downstream model updated; AlignFace weights stayed frozen.")


if __name__ == "__main__":
    main()
