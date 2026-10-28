"""Dataset preparation before editing, followed by checkpoint CLIP processing."""
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image


def load_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB")


class FacePreprocessor:
    """Reproduce landmark alignment and the original white-background pipeline.

    Requires a user-supplied dlib predictor and the optional Facer dependencies.
    Already prepared dataset images should bypass this operation.
    """

    def __init__(self, predictor_path, device="cpu"):
        import dlib
        import facer
        self.device = device
        self.predictor = dlib.shape_predictor(str(predictor_path))
        self.detector = facer.face_detector("retinaface/mobilenet", device=device)
        self.parser = facer.face_parser("farl/lapa/448", device=device)

    def __call__(self, path):
        from ._alignment import align_face
        aligned = align_face(load_rgb(path), predictor=self.predictor, return_tensors=False)[0]
        return self.remove_background(aligned)

    def remove_background(self, image, target_size=1024):
        """Match resize_crop_mask.py, including its intermediate JPEG encoding."""
        import cv2
        import facer
        import torch
        bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        size = min(h, w)
        y, x = (h-size)//2, (w-size)//2
        bgr = cv2.resize(bgr[y:y+size, x:x+size], (target_size, target_size), interpolation=cv2.INTER_AREA)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory)/"parser_input.jpg")
            cv2.imwrite(path, bgr)
            tensor = facer.hwc2bchw(facer.read_hwc(path)).to(device=self.device)
            with torch.inference_mode():
                faces = self.detector(tensor)
                if faces is None or len(faces["rects"]) == 0:
                    raise ValueError("No face detected during background removal.")
                faces = self.parser(tensor, faces)
                labels = faces["seg"]["logits"].softmax(dim=1).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        mask = np.isin(labels, list(range(1, 13))).astype(np.uint8)*255
        if mask.shape != bgr.shape[:2]:
            mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        if np.mean(mask > 0) < 0.01:
            raise ValueError("Face mask covers less than one percent of the image.")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (9, 9), 0)
        alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(float)/255.0
        result = (bgr*alpha + np.ones_like(bgr)*255*(1-alpha)).astype(np.uint8)
        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
