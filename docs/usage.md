# AlignFace

Interpretable face comparison with 20 facial concepts and an attribute-weighted
monotonic GAM. This extraction preserves the final cross-attention / LoRA image
model and learned parameters. Every backbone and variant uses only
`monotonic_gam_attr_softmax`; experimental heads and datasets are excluded.

## Install

Use Python 3.10 and install PyTorch for your platform, then install this repository:

```bash
pip install -e .
```

The package includes backbone configurations, tokenizer files, and full model
weights in safetensors format. Inference works offline and does not need a
separate pretrained backbone download. The default device is CUDA when available,
otherwise CPU. Pass `device="cpu"` or `device="cuda:0"` explicitly if needed.

## Quick start

Inputs below are faces already aligned and placed on white backgrounds.

```python
from alignface import AlignFace

model = AlignFace.from_pretrained("overall", backbone="flip")
pair = model.compare("reference.jpg", "candidate_a.jpg")
print(pair.concepts["nose shape"])
# similarity: normalized concept similarity; logit: raw pair score;
# distance: negative raw pair logit, preserving stage-1 preference differences.

result = model.compare(
    "reference.jpg", "candidate_a.jpg", alternative="candidate_b.jpg"
)
print(result.similarity)
print(result.delta)
print(result.preferred)
print(result.concepts)
```

Run the pair CLI example with `python examples/quickstart.py a.jpg b.jpg`.

## Scores and explanations

Larger stage-1 logits mean greater concept similarity: training rewards positive
`logit(A,R)-logit(B,R)` when A is preferred. Stored training running means and
variances transform logits as `sigmoid(3*(logit-mean)/sqrt(var+eps))`. These values
are already similarities, not distances, and are not inverted. The original
evaluation additionally min/max-scales self-referenced concept logit differences
for its exported tables; that display transform is distinct from GAM input.
Neither normalization guarantees uniform distributions.

For pairs, the unchanged GAM receives the 20 individual concept similarities,
producing `raw_score = sum_i weight_i*effect_i`. The default face similarity is a
fixed reference percentile of this raw score, fitted separately for each variant
and backbone on its original training split. Each training triplet contributes
R–A and R–B pairs; no test rows are used. Metadata and curve knots are stored in
`checkpoints/<variant>/<backbone>/calibration.json`. Only aggregate score knots
and split/checkpoint hashes are shipped, not face images or study tables.

The calibration uses average ranks for tied reference scores, linear interpolation
between distinct raw scores, and monotonic exponential tails outside the reference
range. It does not invert or reorder scores, force distinct scores into coarse
percentile bins, or recalibrate on the current input batch. Exact ties stay tied.
The reference distribution is approximately uniform up to ties and finite sample
size; other datasets need not be uniform. As with any finite-precision computation,
extreme tails can approach machine-representable 0 or 1. Percentiles are relative
ranks, not identity-match probabilities or universally comparable scales across
models. `from_pretrained(..., calibration="sigmoid")` instead exposes
`sigmoid(raw_score)` without percentile calibration. Model weights and raw-score
ordering are unchanged by either display mapping.

Concept outputs contain `similarity` and the global positive GAM `weight`.
Weights sum to one and indicate global importance, not the percentage of a
particular sample's final score. `details=True` additionally exposes raw logits,
spline effects, and contributions; contributions sum to the raw GAM score before
the display transform. `level="concept"` returns all concepts;
`concept="eye color"` returns that concept's similarity as `result.similarity`.
Names such as `eye color` and `eye_colour` are both accepted.

`compare_batch(pairs, batch_size=8)` returns results in input order and supports the
same selectors. Each pair is ordered as reference, candidate; learned scores are
not guaranteed symmetric metrics. Pair-level GAM inference is an experimental
extension selected by the project owner. Calibration checks its numerical ordering,
not agreement with pair-level human judgments.

The original three-image API remains
`compare(reference, candidate_a, alternative=candidate_b)`. It still uses concept
similarity differences as GAM input, `delta=2*sigmoid(raw)-1`, and the original
evaluation AR score `sigmoid(delta)`. Positive delta selects A; otherwise B.
Pair percentile calibration never changes this triplet route.

## Checkpoints

| Variant | Source directory | Backbones |
| --- | --- | --- |
| `overall` | `Checkpoint_Final_Version_0602` | `flip`, `clip` |
| `asian` | `Checkpoint_Final_Version_Asian_0602` | `flip`, `clip` |
| `white` | `Checkpoint_Final_Version` (mapping confirmed by project owner) | `flip`, `clip` |

Backbones are `P01son/FaceCLIP-base-32` and `openai/clip-vit-base-patch32`.
Every combination has its own stage-1 parameters and final GAM parameters.
`assets/provenance.json` records source files and SHA-256 hashes. Export preserves
tensor values while separating sanitized metadata into JSON. Strict loading
rejects missing or mismatched parameters. No optimizer state or training images
are included. Calibration curves are additional per-model JSON files.

Each stage-1 weight file is approximately 626 MB; all six combinations total about
3.8 GB (3.5 GiB). Use Git LFS or a model hosting service for public weight storage;
`.gitattributes` is provided. This local delivery contains the actual weights.

## Preprocessing before editing

The original preparation is split across `align_dataset.py`,
`HairFastGAN/utils/shape_predictor.py`, and `resize_crop_mask.py`. The extracted
`alignface/preprocessing.py` connects these operations:

1. Detect the largest face with dlib and apply the original FFHQ landmark crop,
   reflected padding, 4096-pixel transform, and 1024-pixel output.
2. Center crop and resize to 1024 using OpenCV area interpolation.
3. Parse with Facer RetinaFace/MobileNet and `farl/lapa/448`; retain labels 1–12.
4. Close the mask twice with a 7×7 ellipse, blur it with a 9×9 Gaussian, and blend
   onto white. The original intermediate JPEG for parsing is preserved.
5. Apply the checkpoint's CLIPProcessor only when passing images to the model.

```bash
pip install -e '.[preprocessing]'
python scripts/preprocess.py raw.jpg prepared.jpg --predictor /path/to/shape_predictor_68_face_landmarks.dat
```

Supply the predictor separately; Facer also requires its detector/parser weights
and may download them on first use. Neither is redistributed here. Already
prepared CelebA/MultiPIE or edited images should bypass geometric alignment and
background removal to avoid changing the inputs. For raw images, preprocessing
can run inside comparison:

```python
from alignface import AlignFace, FacePreprocessor
preprocess = FacePreprocessor("/path/to/shape_predictor_68_face_landmarks.dat")
model = AlignFace.from_pretrained("overall", preprocessor=preprocess)
```

## Evaluation

Use a CSV with `reference`, `candidate_a`, `candidate_b`, and optional binary
`label` columns (1 selects A, 0 selects B). Paths are relative to `--image-root`.

```bash
python scripts/evaluate.py pairs.csv predictions.csv --image-root prepared_faces --variant overall
```

The script writes the original raw/linked scores and reports 2AFC agreement for
labeled rows. Inputs must use the prepared image pipeline described above.

## Bundled demo

`python examples/quickstart.py` uses prepared synthetic images in `examples/images`
without any user photos or optional preprocessing dependencies. Use `--concept
"eye color"` to compare that concept, `--demo-pair ac` to inspect only the
contrasting pair, and `--verify` to check image hashes and numerical
outputs against `examples/expected_outputs.json`. CPU is selected automatically
for verification unless overridden. All three images are fictional; see the
[image provenance notes](../README.md#license-and-third-party-notices).

## Validation

```bash
python -m unittest discover -s tests -v
python scripts/validate_reference.py --source /path/to/original-copy
python scripts/validate_preprocessing.py --source /path/to/original-workspace \
  --predictor /path/to/shape_predictor_68_face_landmarks.dat --image /path/to/face.jpg
```

Running the validation scripts generates local reports in `assets/`; generated
reports are excluded from version control. Model comparison uses two real image
triplets per available model, strict state loading, identical processor tensors,
concept outputs, final raw and linked scores, and explanation additivity.
A fresh combined dependency installation was not tested. Single-input tolerance is `rtol=1e-5, atol=1e-6`; batch-versus-single checks
use `rtol=1e-4, atol=1e-5`. this is a CPU equivalence check, not a complete
reproduction of paper-wide evaluation or GPU equivalence. Test images remain in
the original dataset and are not redistributed.

## Release status and citation

The code and weights are prepared locally for review. Public release still needs
the authors' project license choice, upstream weight/data redistribution review,
and finalized paper citation. See [license and third-party notices](../README.md#license-and-third-party-notices).

To refit calibration after an intentional model change, run
`python scripts/fit_reference_calibration.py --source /path/to/original-copy`.
This uses only the three specified training splits and records ordering checks in
`assets/calibration_validation.json`. Demo references can be regenerated for review
with `python scripts/record_demo_reference.py --output /path/to/new_reference.json`;
never update expected outputs simply to hide a failing verification.

## Downstream training

The file API (`compare`, `compare_batch`) deliberately runs in inference mode and
returns Python values. It is appropriate for evaluation, offline pseudo-labels,
reward-based methods without differentiating through the reward, and training a
student similarity predictor. It does not carry gradients to a student model.

For an embedding/recognition model, treat AlignFace scores as detached supervision
for your own pair predictor or embedding-similarity loss. Do not pass embedding
vectors to the image-input metric; `score_tensors` expects images, not features.
A tensor-based teacher step avoids path I/O and Python scalar conversion:

```python
with torch.no_grad():
    target = metric.score_tensors(
        metric.prepare_tensors(face_a), metric.prepare_tensors(face_b)
    )
prediction = student_model(face_a, face_b)  # Your model's [B] pair scores
loss = torch.nn.functional.mse_loss(prediction, target)
loss.backward()
```

Teacher similarity is not an identity label. Its value and bias on a target
population require validation before relying on it for training.

For a model that produces face images, `score_tensors` retains gradients to the
input pixel tensors while keeping both AlignFace stages frozen and in eval mode.
No `.backward()` graph is retained by the path-based API. A DataLoader loop can be:

```python
for source, reference in loader:
    optimizer.zero_grad()
    generated = generator(source)  # [B,3,H,W], aligned RGB in [0,1]
    scores = metric.score_tensors(
        metric.prepare_tensors(reference),
        metric.prepare_tensors(generated),
        calibration="sigmoid",
    )
    loss = (1-scores).mean()
    loss.backward()
    optimizer.step()
```

`prepare_tensors` performs differentiable resize, center crop and CLIP mean/std
normalization using the selected processor's configuration. It does not align
landmarks or remove the background. Those inputs must already be prepared, and
any upstream alignment/masking that requires gradients needs its own differentiable
implementation. A generator producing [-1,1] must first map its output to [0,1].
PIL and Torch bicubic interpolation may differ slightly; file/tensor parity is
checked using the exact same normalized pixel tensors, not by claiming the two
resize implementations are identical.

`score_tensors` accepts matching normalized `[B,3,224,224]` batches and returns
`[B]` face similarities, `[B,20]` for `level="concept"`, or `[B]` for one named
concept. Outputs remain on the metric device. Input-device transfers retain their
gradient connection. It does not internally microbatch training graphs; choose a
DataLoader batch size that fits device memory. The file API's `batch_size` option
chunks inference separately. Load the metric once and reuse it across steps.

The default tensor percentile follows the same fixed calibration as the file API.
It is piecewise differentiable with potentially uneven slopes. `calibration="sigmoid"`
uses `sigmoid(raw_score)` for a smoother loss; `calibration="raw"` exposes the
unbounded score. Neither re-fits normalization on the current batch. Derivatives
at interpolation knots are one-sided and extreme saturation can diminish gradients.
Use the score appropriate for the training objective, not simply the prettiest
reported distribution.

`python examples/training_step.py` performs a real optimizer step on a small color
adapter using the bundled images, checking nonzero finite downstream gradients
and unchanged metric weights. `python scripts/validate_training.py` checks all six
variants/backbones for file/tensor score parity, face and concept gradients, tensor
preprocessing gradients, and frozen parameters. Running this script writes CPU results to the local, ignored report
`assets/training_validation.json`. These integration checks establish API behavior;
they do not establish improved performance of a third-party model or GPU parity.
