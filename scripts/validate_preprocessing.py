"""Compare alignment and background removal with the original preparation code."""
import argparse
import ast
import json
from pathlib import Path
import types

import cv2
import dlib
import numpy as np
import PIL
from PIL import Image
import scipy.ndimage
import torch
from torchvision import transforms as T

from alignface._alignment import align_face
from alignface.preprocessing import FacePreprocessor, load_rgb


def functions(path, names=None):
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and (names is None or n.name in names)]
    return compile(tree, str(path.name), "exec", flags=__import__('__future__').annotations.compiler_flag)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True, help="Original dataset workspace")
    p.add_argument("--predictor", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("assets/preprocessing_validation.json"))
    args = p.parse_args()
    torch.set_num_threads(4)
    import facer
    import os
    import tempfile
    from pathlib import Path as PathClass
    env = dict(PIL=PIL, Image=Image, dlib=dlib, np=np, scipy=__import__('scipy'), torch=torch, T=T, Path=PathClass)
    if not hasattr(Image, "ANTIALIAS"):
        Image.ANTIALIAS = Image.Resampling.LANCZOS
    exec(functions(args.source/"HairFastGAN/utils/shape_predictor.py"), env)
    predictor = dlib.shape_predictor(str(args.predictor))
    image = load_rgb(args.image)
    expected = env["align_face"](image, predictor=predictor, return_tensors=False)[0]
    aligned = align_face(image, predictor=predictor, return_tensors=False)[0]
    np.testing.assert_array_equal(np.asarray(aligned), np.asarray(expected))
    pre = FacePreprocessor(args.predictor, "cpu")
    names = {"crop_to_square", "resize_to_target", "get_face_segmentation", "apply_mask_with_white_background"}
    env = dict(cv2=cv2, np=np, torch=torch, facer=facer, os=os, tempfile=tempfile, FACE_HAIR_IDS=list(range(1,13)), DEBUG=False)
    exec(functions(args.source/"resize_crop_mask.py", names), env)
    bgr = cv2.cvtColor(np.asarray(expected), cv2.COLOR_RGB2BGR)
    bgr = env["resize_to_target"](env["crop_to_square"](bgr), 1024)
    _, mask = env["get_face_segmentation"](bgr, pre.detector, pre.parser, "cpu")
    if mask is None:
        raise RuntimeError("Reference face parsing failed")
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), iterations=2)
    mask = cv2.GaussianBlur(mask, (9,9), 0)
    expected = env["apply_mask_with_white_background"](bgr, mask)
    actual = cv2.cvtColor(np.asarray(pre.remove_background(aligned)), cv2.COLOR_RGB2BGR)
    np.testing.assert_array_equal(actual, expected)
    args.output.write_text(json.dumps(dict(alignment_pixels_exact=True, background_pixels_exact=True, real_images=1, output_size=1024, torch=torch.__version__),indent=2)+"\n")
    print("Alignment and Facer white-background removal match exactly.", flush=True)


if __name__ == "__main__":
    main()
