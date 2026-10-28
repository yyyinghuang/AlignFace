from __future__ import annotations
"""Pair, batch, and concept inference with strict checkpoint loading."""
from dataclasses import dataclass
import inspect
import json
from pathlib import Path
import sys
from safetensors.torch import load_file

import torch
from transformers import CLIPConfig, CLIPProcessor

from .model import CLIPSharedAttributeModel, Stage2MonotonicGAMAttrSoftmaxHead, get_attr_text_list
from .preprocessing import load_rgb
from .calibration import PercentileCalibration


@dataclass
class ComparisonResult:
    similarity: float | None
    concepts: dict
    delta: float | None = None
    raw_score: float | None = None
    preferred: str | None = None
    experimental: bool = False
    level: str = "face"
    concept: str | None = None
    score_type: str = "concept_similarity"


class AlignFace:
    """Compare prepared faces; raw images require an explicit FacePreprocessor."""

    @classmethod
    def from_pretrained(cls, variant="overall", *, backbone="flip", device=None, root=None, preprocessor=None, calibration="percentile"):
        if root is None:
            root = Path(__file__).resolve().parent.parent
            if not (root/"configs/variants.json").exists():
                root = Path(sys.prefix)/"share/alignface"
        else:
            root = Path(root)
        variants = json.loads((root/"configs/variants.json").read_text())
        if variant not in variants:
            raise ValueError(f"Unknown variant: {variant}")
        if variants[variant] is None:
            raise FileNotFoundError(f"No verified final checkpoint mapping for {variant!r}.")
        if backbone not in variants[variant]:
            raise ValueError(f"Unknown backbone: {backbone}")
        info = variants[variant][backbone]
        directory = root/info["directory"]
        self = cls()
        if calibration not in ("percentile", "sigmoid"):
            raise ValueError("calibration must be percentile or sigmoid.")
        self.calibration = calibration
        self.score_calibrator = None
        if calibration == "percentile":
            self.score_calibrator = PercentileCalibration.from_dict(json.loads((directory/"calibration.json").read_text()))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        c1 = json.loads((directory/"stage1.json").read_text())
        c1["model_state_dict"] = load_file(str(directory/"stage1.safetensors"))
        c2 = json.loads((directory/"stage2.json").read_text())
        c2["model_state_dict"] = load_file(str(directory/"stage2.safetensors"))
        summary = c1["summary"]
        self.attributes = list(summary["selected_attributes"])
        if self.attributes != c2["stage1_selected_attributes"]:
            raise ValueError("Stage-1 and stage-2 concept order differs.")
        if c2["stage2_head_type"] != "monotonic_gam_attr_softmax":
            raise ValueError("Expected the final attribute-softmax monotonic GAM.")
        if c2["feature_type"] != "stage1_delta_sims" or c2["normalize_inputs"]:
            raise ValueError("Unsupported checkpoint feature transformation.")
        if c2.get("clip_delta") is not None:
            raise ValueError("Clipped output checkpoints require an explicit implementation.")
        if c2["final_delta_activation"] != "sigmoid_signed":
            raise ValueError("Unsupported final link function.")
        kwargs = {k:v for k,v in summary.items() if k in inspect.signature(CLIPSharedAttributeModel).parameters}
        kwargs.update(model_name=c1["model_name"], num_attributes=len(self.attributes), backbone_config=CLIPConfig.from_pretrained(root/info["config"], local_files_only=True))
        self.stage1 = CLIPSharedAttributeModel(**kwargs)
        self.stage1.load_state_dict(c1["model_state_dict"], strict=True)
        if not bool(self.stage1.sim_stats_initialized):
            raise ValueError("Checkpoint lacks initialized attribute normalization statistics.")
        self.stage2 = Stage2MonotonicGAMAttrSoftmaxHead(len(self.attributes), n_bins=c2["n_knots"], x_min=c2["x_min"], x_max=c2["x_max"], gam_dropout=c2["dropout"])
        self.stage2.load_state_dict(c2["model_state_dict"], strict=True)
        self.stage1.to(self.device).eval().requires_grad_(False)
        self.stage2.to(self.device).eval().requires_grad_(False)
        self.processor = CLIPProcessor.from_pretrained(root/info["config"], local_files_only=True)
        self.preprocessor = preprocessor
        with torch.inference_mode():
            tokens = self.processor(text=get_attr_text_list(self.attributes), return_tensors="pt", padding=True, truncation=True)
            self.text = self.stage1.encode_text(tokens["input_ids"].to(self.device), tokens["attention_mask"].to(self.device))
        self.text = self.text.clone()
        return self

    def _pixels(self, path):
        image = self.preprocessor(path) if self.preprocessor else load_rgb(path)
        return self.processor(images=image, return_tensors="pt")["pixel_values"].to(self.device)

    @property
    def concept_names(self):
        """Human-readable concept names accepted by compare and compare_batch."""
        return [name.replace("_", " ").replace("colour", "color") for name in self.attributes]

    def _concept_index(self, concept):
        normalized = concept.strip().lower().replace("-", "_").replace(" ", "_").replace("color", "colour")
        aliases = {"facial_proportions": "face_proportion", "distance_between_the_pupils": "pupillary_distance"}
        normalized = aliases.get(normalized, normalized)
        if normalized not in self.attributes:
            raise ValueError(f"Unknown concept {concept!r}. Available concepts: {', '.join(self.concept_names)}")
        return self.attributes.index(normalized)

    def prepare_tensors(self, images):
        """Normalize aligned BCHW RGB floats in [0,1] without detaching gradients.

        Torch resizing can differ slightly from the PIL file API. Inputs must
        already be aligned and masked; this method does not run dlib or Facer.
        """
        import torch.nn.functional as F
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have RGB shape [B,3,H,W].")
        if not torch.is_floating_point(images):
            raise TypeError("images must be floating-point RGB values in [0,1].")
        if images.numel() == 0 or not bool(torch.isfinite(images).all()):
            raise ValueError("images must be nonempty and finite.")
        if bool((images.detach()<0).any()) or bool((images.detach()>1).any()):
            raise ValueError("images must lie in [0,1]; rescale generator outputs first.")
        images = images.to(device=self.device, dtype=self.text.dtype)
        config = self.processor.image_processor
        size = config.size
        shortest = int(size.get("shortest_edge", 224)) if isinstance(size, dict) else int(size)
        crop = config.crop_size
        height, width = (int(crop["height"]),int(crop["width"])) if isinstance(crop,dict) else (int(crop),int(crop))
        h,w = images.shape[-2:]
        resized = (int(shortest*h/min(h,w)),int(shortest*w/min(h,w)))
        if (h,w) != resized:
            images = F.interpolate(images,size=resized,mode="bicubic",align_corners=False,antialias=True).clamp(0,1)
        top,left = (images.shape[-2]-height)//2,(images.shape[-1]-width)//2
        images = images[:,:,top:top+height,left:left+width]
        mean = images.new_tensor(config.image_mean).view(1,3,1,1)
        std = images.new_tensor(config.image_std).view(1,3,1,1)
        return (images-mean)/std

    def score_tensors(self, reference, candidate, *, level="face", concept=None, calibration=None):
        """Return tensor batch scores while retaining gradients to input pixels.

        Metric weights are frozen. Face output is [B]; all concepts are [B,D].
        Use calibration='sigmoid' for a smoother loss than reference percentiles.
        """
        if level not in ("face","concept"):
            raise ValueError("level must be 'face' or 'concept'.")
        if calibration not in (None,"percentile","sigmoid","raw"):
            raise ValueError("calibration must be percentile, sigmoid, raw, or None.")
        if not isinstance(reference,torch.Tensor) or not isinstance(candidate,torch.Tensor):
            raise TypeError("Inputs must be preprocessed torch tensors.")
        expected = self.stage1.clip.config.vision_config.image_size
        if (reference.ndim != 4 or reference.shape != candidate.shape
                or reference.shape[0] == 0 or tuple(reference.shape[1:]) != (3,expected,expected)):
            raise ValueError(f"Both inputs must have matching nonempty shape [B,3,{expected},{expected}].")
        if not torch.is_floating_point(reference) or not torch.is_floating_point(candidate):
            raise TypeError("Inputs must be floating-point normalized pixels.")
        if not bool(torch.isfinite(reference).all()) or not bool(torch.isfinite(candidate).all()):
            raise ValueError("Input tensors must be finite.")
        reference = reference.to(device=self.device,dtype=self.text.dtype)
        candidate = candidate.to(device=self.device,dtype=self.text.dtype)
        ref_embedding = self.stage1.encode_image(reference)
        candidate_embedding = self.stage1.encode_image(candidate)
        b,n = reference.shape[0],len(self.attributes)
        pair = self.stage1.build_pair(candidate_embedding,ref_embedding).unsqueeze(1).expand(b,n,-1)
        prefix = self.stage1.text_learnable_tokens.unsqueeze(0).expand(b,n,-1,-1)
        prompt = self.text.unsqueeze(0).expand(b,n,-1).unsqueeze(2)
        text = torch.cat([prefix,prompt],dim=2).reshape(b,n,-1)
        _,logits = self.stage1.head(pair,text)
        similarities,_ = self.stage1._normalized_sim_from_logits(logits,logits)
        out = {"sim_ar": similarities}
        if concept is not None:
            return out["sim_ar"][:,self._concept_index(concept)]
        if level == "concept":
            return out["sim_ar"]
        raw = self.stage2(out["sim_ar"])
        mode = calibration or self.calibration
        if mode == "raw":
            return raw
        if mode == "sigmoid":
            return torch.sigmoid(raw)
        if self.score_calibrator is None:
            raise ValueError("Load with percentile calibration to request percentile scores.")
        return self.score_calibrator.transform_tensor(raw)

    @torch.inference_mode()
    def compare_batch(self, pairs, *, level="face", concept=None, batch_size=8, details=False):
        """Compare pairs in batches, optionally selecting all concepts or one concept.

        Face-level pair scores apply the trained GAM to individual similarities.
        This user-selected inference extension is experimental: GAM training used
        differences between two pairs. A fixed reference percentile maps its raw
        score to [0, 1] without reversing the order. details exposes raw effects.
        """
        if level not in ("face", "concept"):
            raise ValueError("level must be 'face' or 'concept'.")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        index = self._concept_index(concept) if concept is not None else None
        if index is not None:
            level = "concept"
        pairs = list(pairs)
        if any(isinstance(pair, (str, bytes)) or len(pair) != 2 for pair in pairs):
            raise ValueError("Each pair must contain exactly two image paths.")
        results = []
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start+batch_size]
            ref = torch.cat([self._pixels(pair[0]) for pair in chunk])
            a = torch.cat([self._pixels(pair[1]) for pair in chunk])
            out = self.stage1(ref, a, a, self.text)
            if level == "face":
                raw, effects, weights, _ = self.stage2.forward_detailed(out["sim_ar"])
                linked = self.score_calibrator.transform(raw.detach().cpu().numpy()) if self.score_calibrator is not None else torch.sigmoid(raw).cpu().numpy()
            global_weights = self.stage2.attr_weights()
            for row in range(len(chunk)):
                selected = range(len(self.attributes)) if index is None else [index]
                concepts = {}
                for i in selected:
                    values = {"similarity": float(out["sim_ar"][row,i]), "weight": float(global_weights[i])}
                    if details:
                        values["logit"] = float(out["logits_ar"][row,i])
                        if level == "face":
                            values.update(effect=float(effects[row,i]), contribution=float((effects*weights)[row,i]))
                    concepts[self.concept_names[i]] = values
                score = float(linked[row]) if level == "face" else (concepts[self.concept_names[index]]["similarity"] if index is not None else None)
                results.append(ComparisonResult(score, concepts, raw_score=float(raw[row]) if level == "face" else None, experimental=level=="face", level=level, concept=self.concept_names[index] if index is not None else None, score_type=("reference_percentile" if self.score_calibrator is not None else "sigmoid_pair") if level=="face" else "concept_similarity"))
        return results

    @torch.inference_mode()
    def compare(self, face1, face2, *, level="face", concept=None, alternative=None, details=False):
        """Compare one pair or reproduce the original three-image 2AFC evaluation."""
        if alternative is None:
            return self.compare_batch([(face1, face2)], level=level, concept=concept, batch_size=1, details=details)[0]
        if level != "face" or concept is not None:
            raise ValueError("Concept selection applies to pair inference; omit alternative.")
        ref, a, b = self._pixels(face1), self._pixels(face2), self._pixels(alternative)
        out = self.stage1(ref, a, b, self.text)
        raw, u, weights, _ = self.stage2.forward_detailed(out["delta_sim"])
        delta = 2*torch.sigmoid(raw)-1
        concepts = {}
        for i,name in enumerate(self.concept_names):
            concepts[name] = dict(similarity=float(out["sim_ar"][0,i]), weight=float(weights[0,i]))
            if details:
                concepts[name].update(logit=float(out["logits_ar"][0,i]), alternative_similarity=float(out["sim_br"][0,i]), delta=float(out["delta_sim"][0,i]), effect=float(u[0,i]), contribution=float((u*weights)[0,i]))
        return ComparisonResult(float(torch.sigmoid(delta)[0]), concepts, float(delta[0]), float(raw[0]), "A" if delta[0]>0 else "B", score_type="triplet_preference")
