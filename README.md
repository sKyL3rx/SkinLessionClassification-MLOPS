# Skin Lesion Classification

A deep learning project for classifying skin lesions from dermoscopic images using both image features and patient metadata.

The main focus of this project is experimenting with different model architectures and multimodal fusion strategies, comparing their performance, and taking the selected model through an inference and deployment pipeline.

The project is based on the HAM10000 dataset and predicts seven lesion classes:

- `akiec`
- `bcc`
- `bkl`
- `df`
- `mel`
- `nv`
- `vasc`


---

## Project Overview

I organized the project around two main parts:

1. **Model experiments**
2. **Deployment of the selected model**

The general workflow is:

```text
Dataset
   ↓
Experiment configs
   ↓
Train / evaluate candidate models
   ↓
Compare validation results
   ↓
Select a model
   ↓
Export to ONNX
   ↓
Calibration + selective prediction
   ↓
Error analysis
   ↓
Deployment bundle
   ↓
FastAPI / Gradio demo
```

Instead of focusing on a single architecture, I used the project to compare different image backbones, metadata fusion strategies, and preprocessing choices.

---

## Experiments

Experiment configurations are stored in:

```text
configs/experiments/
```

Each experiment keeps model architecture, image size, metadata settings, preprocessing, augmentation, optimizer, loss, and evaluation settings separate from the implementation code.

The experiments cover combinations of:

- ConvNeXtV2
- EfficientNetV2
- DINOv2
- image-only classification
- image + metadata classification
- concatenation fusion
- Gated Multimodal Unit (GMU)
- different image resolutions
- preprocessing and augmentation settings

The six main experiments are documented in detail in [EXPERIMENTS.md](EXPERIMENTS.md).

Full training runs, reports, and checkpoints are available on Google Drive:

**[Google Drive — S1–S6 Experiment Artifacts](https://drive.google.com/drive/folders/15fYdbFiFylaicqMJwVrwqC1ChsTEGOpM?usp=sharing)**

---

## Multimodal Learning

One of the main parts of this project is combining dermoscopic images with structured metadata.

The image branch produces visual features from an image backbone, while the metadata branch processes structured features with a small MLP.

```text
Image
  ↓
Image Backbone
  ↓
Image Features
        \
         → Fusion → Classifier → Prediction
        /
Metadata
  ↓
Metadata MLP
  ↓
Metadata Features
```

I implemented two fusion strategies.

### Concatenation

The baseline multimodal model concatenates image and metadata representations before classification.

```text
fused = concat(image_features, metadata_features)
```

This strategy is used in S2 as the first multimodal baseline.

### Gated Multimodal Unit

I also implemented a GMU-based fusion module.

Instead of always combining both modalities with the same importance, GMU learns a gate that controls the contribution from each branch.

Conceptually:

```text
image_hidden = image_projection(image_features)
metadata_hidden = metadata_projection(metadata_features)

gate = sigmoid(
    W [image_features, metadata_features]
)

fused =
    gate * image_hidden
    + (1 - gate) * metadata_hidden
```

The implementation is located in:

```text
src/lesion_ml/models/fusion.py
```

The model factory can switch between `concat` and `gmu` through the experiment configuration.

---

## Experiment Structure

The project contains six main experiment variants for comparing modeling choices.

| Experiment | Backbone | Resolution | Metadata | Fusion | Main Change |
|---|---|---:|---:|---|---|
| S1 | ConvNeXtV2-Tiny | 384 | No | Image only | Image-only baseline |
| S2 | ConvNeXtV2-Tiny | 384 | Yes | Concatenation | Add metadata |
| S3 | ConvNeXtV2-Tiny | 384 | Yes | GMU | Learned multimodal fusion |
| S4 | ConvNeXtV2-Tiny | 384 | Yes | GMU | Dark-border preprocessing |
| S5 | EfficientNetV2-S | 384 | Yes | GMU | Backbone comparison |
| S6 | DINOv2-Small | 518 | Yes | GMU | Frozen foundation backbone |

The experiments are meant to answer questions such as:

- Does metadata improve over an image-only model?
- Does GMU improve over simple concatenation?
- Does additional image preprocessing help?
- How much does the image backbone affect performance?
- Can a frozen foundation-model backbone compete with end-to-end training?

The selected model is **S3 — ConvNeXtV2-Tiny + Metadata + GMU**.

Its main results were:

| Metric | Score |
|---|---:|
| Best Validation Macro-F1 | **0.8471** |
| Test Macro-F1 | **0.8241** |
| Test Accuracy | **0.8842** |
| Test Balanced Accuracy | **0.8098** |

Model selection is based on validation performance rather than selecting directly on the test set.

For the complete S1–S6 results, comparisons, and artifact links, see [EXPERIMENTS.md](EXPERIMENTS.md).

---

## Evaluation

The evaluation pipeline reports several metrics instead of relying only on accuracy.

The main metrics include:

- accuracy
- macro F1
- weighted F1
- balanced accuracy
- per-class metrics
- confusion matrix

Macro-F1 is useful for this project because HAM10000 is class-imbalanced.

Prediction outputs can also be saved for later calibration and error analysis.

Main evaluation code:

```text
scripts/evaluate.py
scripts/evaluate_onnx.py
scripts/error_analysis.py
```

---

## Test-Time Augmentation

The evaluation pipeline supports test-time augmentation for selected experiments.

Typical views include:

```text
original
horizontal flip
vertical flip
```

Predictions from multiple views can be combined during inference without changing the trained checkpoint.

---

## ONNX Export

After selecting a model, it can be exported from PyTorch to ONNX.

```bash
make export
```

The exported model is then evaluated separately with ONNX Runtime:

```bash
make evaluate-onnx
```

This allows me to check whether the deployment version of the model behaves consistently with the original PyTorch implementation.

The ONNX-related code is located in:

```text
scripts/export_onnx.py
scripts/evaluate_onnx.py
src/lesion_ml/inference/
```

---

## Calibration

Classifier confidence is not always well calibrated.

For the selected model, I use temperature scaling on validation predictions:

```bash
make calibrate
```

Conceptually:

```text
calibrated_logits = logits / T
```

The learned temperature is reused by the inference pipeline.

---

## Selective Prediction

The project supports a confidence-based review policy.

Instead of automatically accepting every prediction:

```text
Prediction
    ↓
Confidence >= threshold
    ├── Yes → Accept
    └── No  → Review
```

The threshold is selected using validation data.

This allows the model to trade prediction coverage for higher confidence on accepted samples.

Related code:

```text
scripts/calibrate_temperature.py
scripts/evaluate_selective_prediction.py
```

Run:

```bash
make triage
```

---

## Error Analysis

After evaluating the selected model, I also inspect its failure cases instead of relying only on aggregate metrics.

The error-analysis stage looks at:

- difficult lesion classes
- common class confusions
- low-confidence predictions
- errors involving important classes such as `mel` and `akiec`
- metadata-related patterns

Run:

```bash
make error-analysis
```

The goal is to understand where the model fails and whether those failures follow consistent patterns.

---

## Deployment

After selecting a model, the project builds a deployment bundle containing the artifacts required for inference.

```bash
make bundle
```

The bundle contains the ONNX model together with the configuration and files required for calibration and prediction decisions.

The expected bundle can be validated with:

```bash
make check-bundle
```

The deployment pipeline focuses on the selected S3 model rather than all six experimental models.

---

## FastAPI

The inference API is implemented with FastAPI.

Run:

```bash
make serve-api
```

The API handles:

1. input preprocessing
2. metadata encoding
3. ONNX inference
4. confidence calibration
5. prediction output

The application code is located in:

```text
src/lesion_ml/api/
```

---

## Gradio Demo

A small Gradio application is included for interactive testing.

Run:

```bash
make serve-demo
```

The demo provides a simple interface around the inference service.

The code is located in:

```text
src/lesion_ml/demo/
```

---

## Docker

The inference service can run inside Docker.

Build:

```bash
make docker-build
```

Run:

```bash
make docker-up
```

Stop:

```bash
make docker-down
```

Deployment-related files are located under:

```text
deployment/
```

---

## Cloud Experiments

Some larger experiments were run on cloud GPU instances using SkyPilot.

DVC with Cloudflare R2 is used to store large experiment artifacts such as checkpoints and exported models.

---

## DVC Pipeline

DVC connects the main training, evaluation, and deployment stages.

Current stages include:

```text
prepare_data
make_splits
train
evaluate_val
evaluate_test
export_onnx
evaluate_onnx_val
evaluate_onnx_test
calibrate
triage_val
triage_test
error_analysis_test
build_deployment_bundle
benchmark_onnx
```

The pipeline is defined in:

```text
dvc.yaml
```

---

## Project Structure

```text
.
├── configs/
│   └── experiments/
│
├── deployment/
│   ├── docker/
│   ├── skypilot/
│   └── compose.yaml
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── evaluate_onnx.py
│   ├── export_onnx.py
│   ├── calibrate_temperature.py
│   ├── evaluate_selective_prediction.py
│   ├── error_analysis.py
│   ├── benchmark_inference.py
│   └── build_deployment_bundle.py
│
├── src/lesion_ml/
│   ├── api/
│   ├── data/
│   ├── demo/
│   ├── inference/
│   ├── models/
│   └── paths.py
│
├── tests/
├── EXPERIMENTS.md
├── dvc.yaml
├── params.yaml
└── Makefile
```

---

## Running the Main Checks

Run linting and fast tests:

```bash
make verify
```

Run the full test suite:

```bash
make test
```

---

## Main Deployment Commands

Export the selected model:

```bash
make export
```

Evaluate the ONNX model:

```bash
make evaluate-onnx
```

Calibrate confidence:

```bash
make calibrate
```

Evaluate the selective-prediction policy:

```bash
make triage
```

Run error analysis:

```bash
make error-analysis
```

Build the deployment bundle:

```bash
make bundle
```

Validate the deployment bundle:

```bash
make check-bundle
```

Run the API:

```bash
make serve-api
```

Run the demo:

```bash
make serve-demo
```

---

## Experiment Artifacts

The complete S1–S6 experiment report is available in:

**[EXPERIMENTS.md](EXPERIMENTS.md)**
