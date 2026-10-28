"""Small tests for interpolation, explanation additivity, and API boundaries."""
import unittest
import hashlib
import json
import numpy as np
import torch
from alignface import AlignFace
from alignface.model import ATTRIBUTES, Stage2MonotonicGAMAttrSoftmaxHead, _tree_interp_1d
from alignface.preprocessing import load_rgb
from alignface.calibration import PercentileCalibration
from PIL import Image
import tempfile
from pathlib import Path


class CoreTests(unittest.TestCase):
    def test_interpolation_clamps_and_interpolates(self):
        edges = torch.tensor([-1., 0., 1.])
        values = torch.tensor([2., 4., 8.])
        actual = _tree_interp_1d(torch.tensor([-2., -.5, .5, 2.]), edges, values)
        torch.testing.assert_close(actual, torch.tensor([2., 3., 6., 8.]), atol=1e-5, rtol=1e-5)

    def test_gam_monotonicity_and_explanation(self):
        torch.manual_seed(4)
        head = Stage2MonotonicGAMAttrSoftmaxHead(20).eval()
        x = torch.rand(4, 20)*2-1
        s, u, w, _ = head.forward_detailed(x)
        torch.testing.assert_close(s, (u*w).sum(-1))
        torch.testing.assert_close(w.sum(-1), torch.ones(4))
        self.assertTrue(bool((head(x+.1) >= s-1e-6).all()))

    def test_unknown_variant_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "Unknown variant"):
            AlignFace.from_pretrained("invented")

    def test_concept_names_aliases_and_invalid_inputs(self):
        model = AlignFace()
        model.attributes = list(ATTRIBUTES)
        self.assertIn("eye color", model.concept_names)
        self.assertEqual(model._concept_index("eye color"), model._concept_index("eye_colour"))
        self.assertEqual(model.compare_batch([]), [])
        with self.assertRaisesRegex(ValueError, "Unknown concept"):
            model.compare_batch([], concept="not a concept")
        with self.assertRaisesRegex(ValueError, "level"):
            model.compare_batch([], level="wrong")
        with self.assertRaisesRegex(ValueError, "positive integer"):
            model.compare_batch([], batch_size=0)
        with self.assertRaisesRegex(ValueError, "exactly two"):
            model.compare_batch([("one",)])

    def test_bundled_demo_assets_and_reference(self):
        examples = Path(__file__).resolve().parent.parent/"examples"
        reference = json.loads((examples/"expected_outputs.json").read_text())
        for name, expected_hash in reference["image_sha256"].items():
            path = examples/"images"/name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected_hash)
            with Image.open(path) as image:
                self.assertEqual(image.size, (1024, 1024))
                self.assertEqual(image.mode, "RGB")
        self.assertEqual(set(reference["models"]), {"overall", "asian", "white"})
        for backbones in reference["models"].values():
            self.assertEqual(set(backbones), {"flip", "clip"})
            for record in backbones.values():
                for name in ["ab", "ac"]:
                    result = record[name]
                    self.assertEqual(len(result["concepts"]), 20)
                    self.assertTrue(0 <= result["similarity"] <= 1)
                    self.assertTrue(all(0 <= value <= 1 for value in result["concepts"].values()))
                self.assertEqual(record["ab_higher_than_ac"], record["ab"]["similarity"] > record["ac"]["similarity"])
        self.assertEqual(set(reference["image_sha256"]), {"face_a.png", "face_b.png", "face_c.png"})

    def test_percentiles_preserve_order_ties_and_batch_independence(self):
        reference = np.array([3., 1., 2., 2., 5., 4.])
        calibration = PercentileCalibration.fit(reference)
        scores = np.array([0.9, 1., 1.5, 2., 2., 3., 4., 5., 5.1])
        mapped = calibration.transform(scores)
        self.assertTrue(np.all((mapped > 0) & (mapped < 1)))
        np.testing.assert_array_equal(np.sign(np.diff(mapped)), np.sign(np.diff(scores)))
        self.assertEqual(mapped[3], mapped[4])
        for score, percentile in zip(scores, mapped):
            self.assertEqual(float(calibration.transform([score])[0]), percentile)
        values = calibration.transform(reference)
        np.testing.assert_array_equal(np.argsort(reference, kind="stable"), np.argsort(values, kind="stable"))
        unique = PercentileCalibration.fit([1., 2., 3., 4.])
        np.testing.assert_allclose(unique.transform([1., 2., 3., 4.]), [.125, .375, .625, .875])
        np.testing.assert_allclose(PercentileCalibration.from_dict(calibration.to_dict()).transform(scores), mapped)
        with self.assertRaises(ValueError):
            PercentileCalibration.fit([1., 1.])

    def test_tensor_percentiles_match_numpy_and_keep_gradients(self):
        calibration = PercentileCalibration.fit([1.,2.,3.,4.])
        x = torch.tensor([.9,1.5,2.5,3.5,4.1],dtype=torch.float64,requires_grad=True)
        actual = calibration.transform_tensor(x)
        np.testing.assert_allclose(actual.detach().numpy(),calibration.transform(x.detach().numpy()),rtol=1e-12,atol=1e-12)
        self.assertTrue(torch.autograd.gradcheck(calibration.transform_tensor,(x,)))
        actual.sum().backward()
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        self.assertTrue(bool((x.grad>0).all()))

    def test_prepared_image_preserves_rgb_pixels(self):
        values = np.random.default_rng(4).integers(0, 256, (17, 21, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"image.png"
            Image.fromarray(values).save(path)
            np.testing.assert_array_equal(np.asarray(load_rgb(path)), values)


if __name__ == "__main__":
    unittest.main()
