# Experiments

This document summarizes the six main experiments used to compare image-only and multimodal models for skin lesion classification.

The main questions were:

- How strong is the image-only baseline?
- Does metadata improve performance?
- Is GMU better than simple concatenation?
- Does dark-border cropping help?
- Does changing the backbone improve the multimodal model?
- Can frozen DINOv2 features compete with end-to-end trained CNNs?

For model selection, I use **Best Validation Macro-F1** as the main metric and report test metrics separately.

---

## Experiment Matrix

| ID | Backbone | Resolution | Metadata | Fusion | Preprocessing | Main Change |
|---|---|---:|---:|---|---|---|
| S1 | ConvNeXtV2-Tiny | 384 | No | — | ResizePad | Image-only baseline |
| S2 | ConvNeXtV2-Tiny | 384 | Yes | Concat | ResizePad | Add metadata |
| S3 | ConvNeXtV2-Tiny | 384 | Yes | GMU | Strict ResizePad | Learned multimodal fusion |
| S4 | ConvNeXtV2-Tiny | 384 | Yes | GMU | Dark-border crop + ResizePad | Preprocessing ablation |
| S5 | EfficientNetV2-S | 384 | Yes | GMU | ResizePad | Backbone comparison |
| S6 | DINOv2-Small | 518 | Yes | GMU | ResizePad | Frozen foundation backbone |

All experiments use the same seven-class HAM10000 classification task.

The main experiments use class-balanced focal loss.

---

## Results

| ID | Best Val Macro-F1 | Test Macro-F1 | Test Accuracy | Test Balanced Acc. |
|---|---:|---:|---:|---:|
| S1 | 0.8129 | 0.8229 | **0.8862** | 0.8077 |
| S2 | 0.8229 | 0.8087 | 0.8762 | 0.7872 |
| **S3** | **0.8471** | **0.8241** | 0.8842 | **0.8098** |
| S4 | 0.8151 | 0.7831 | 0.8663 | 0.7925 |
| S5 | 0.8335 | 0.7517 | 0.8663 | 0.7667 |
| S6 | 0.7189 | 0.6678 | 0.8184 | 0.6714 |

S3 achieved the highest **Best Validation Macro-F1** and the highest **Test Macro-F1**.

---

# S1 — ConvNeXtV2 Image-Only Baseline

**Run**

```text
convnextv2_tiny_384_image_only_resizepad
```

## Configuration

- ConvNeXtV2-Tiny
- 384 × 384 input
- image only
- no metadata
- resize-and-pad preprocessing
- class-balanced focal loss

## Results

| Metric | Score |
|---|---:|
| Best Validation Macro-F1 | 0.8129 |
| Test Accuracy | **0.8862** |
| Test Macro-F1 | 0.8229 |
| Test Balanced Accuracy | 0.8077 |

## Key insights

S1 provided a strong image-only baseline.

Its Test Macro-F1 of `0.8229` was already competitive with the later multimodal models, so metadata had to provide a real improvement rather than simply adding more complexity.

---

# S2 — ConvNeXtV2 + Metadata Concatenation

**Run**

```text
convnextv2_tiny_384_concat_metadata_resizepad
```

## Configuration

- ConvNeXtV2-Tiny
- 384 × 384 input
- image + metadata
- concatenation fusion
- resize-and-pad preprocessing
- class-balanced focal loss

## Results

| Metric | Score |
|---|---:|
| Best Validation Macro-F1 | 0.8229 |
| Test Accuracy | 0.8762 |
| Test Macro-F1 | 0.8087 |
| Test Balanced Accuracy | 0.7872 |

## Compared with S1

Best Validation Macro-F1:

```text
S1 Image Only:       0.8129
S2 Metadata Concat:  0.8229
```

Difference:

```text
+0.0100
```

Test Macro-F1:

```text
S1: 0.8229
S2: 0.8087
```

Difference:

```text
-0.0142
```

## Key insights

Adding metadata improved the best validation score, but simple concatenation did not improve test performance.

This suggested that metadata could be useful, but the fusion method mattered.

---

# S3 — ConvNeXtV2 + GMU Metadata Fusion

**Run**

```text
convnextv2_tiny_384_gmu_metadata_resizepad_strict
```

## Configuration

- ConvNeXtV2-Tiny
- 384 × 384 input
- image + metadata
- Gated Multimodal Unit
- strict resize-and-pad preprocessing
- class-balanced focal loss

## Results

| Metric | Score |
|---|---:|
| Best Validation Macro-F1 | **0.8471** |
| Test Accuracy | 0.8842 |
| Test Macro-F1 | **0.8241** |
| Test Balanced Accuracy | **0.8098** |

S3 produced the strongest overall result.

---

## GMU vs Concatenation

S2 and S3 use the same general ConvNeXtV2 + metadata setup.

The main difference is the fusion strategy.

Best Validation Macro-F1:

```text
S2 Concat:  0.8229
S3 GMU:     0.8471
```

Difference:

```text
+0.0242
```

Test Macro-F1:

```text
S2 Concat:  0.8087
S3 GMU:     0.8241
```

Difference:

```text
+0.0154
```

## Key insights

GMU performed better than simple concatenation on both validation and test Macro-F1.

This was the clearest experiment showing that the way image and metadata features are combined matters.

---

## S3 vs Image-Only Baseline

Best Validation Macro-F1:

```text
S1 Image Only:  0.8129
S3 GMU:         0.8471
```

Difference:

```text
+0.0342
```

Test Macro-F1:

```text
S1 Image Only:  0.8229
S3 GMU:         0.8241
```

Difference:

```text
+0.0012
```

## Key insights

The validation improvement from metadata + GMU was clear.

The improvement on the test set was much smaller because S1 was already a strong visual baseline.

---

# S4 — Dark-Border Crop Preprocessing

**Run**

```text
convnextv2_tiny_384_gmu_metadata_crop_dark_border
```

## Configuration

- ConvNeXtV2-Tiny
- 384 × 384 input
- image + metadata
- GMU fusion
- dark-border crop preprocessing
- resize-and-pad after cropping
- class-balanced focal loss

S4 tested whether removing dark borders around dermoscopic images before resizing would help the model focus more on the lesion area.

## Results

| Metric | Score |
|---|---:|
| Best Validation Macro-F1 | 0.8151 |
| Test Accuracy | 0.8663 |
| Test Macro-F1 | 0.7831 |
| Test Balanced Accuracy | 0.7925 |

## Compared with S3

Best Validation Macro-F1:

```text
S3 Strict ResizePad:    0.8471
S4 Dark-border crop:    0.8151
```

Difference:

```text
-0.0320
```

Test Macro-F1:

```text
S3 Strict ResizePad:    0.8241
S4 Dark-border crop:    0.7831
```

Difference:

```text
-0.0410
```

## Key insights

Dark-border cropping did not improve the model.

The simpler strict resize-and-pad preprocessing used by S3 performed better, so I kept that preprocessing strategy.

---

# S5 — EfficientNetV2-S + GMU

**Run**

```text
efficientnetv2_s_stage2_384_gmu_metadata_resizepad
```

## Configuration

- EfficientNetV2-S
- 384 × 384 input
- image + metadata
- GMU fusion
- resize-and-pad preprocessing
- class-balanced focal loss

S5 tested whether changing the image backbone while keeping the multimodal setup could improve performance.

## Results

| Metric | Score |
|---|---:|
| Best Validation Macro-F1 | 0.8335 |
| Test Accuracy | 0.8663 |
| Test Macro-F1 | 0.7517 |
| Test Balanced Accuracy | 0.7667 |

## Compared with S3

Best Validation Macro-F1:

```text
S3 ConvNeXtV2:      0.8471
S5 EfficientNetV2:  0.8335
```

Difference:

```text
-0.0136
```

Test Macro-F1:

```text
S3 ConvNeXtV2:      0.8241
S5 EfficientNetV2:  0.7517
```

Difference:

```text
-0.0724
```

## Key insights

EfficientNetV2-S was reasonably competitive on validation but generalized worse on the test set.

For this setup, changing the backbone did not improve over ConvNeXtV2.

---

# S6 — Frozen DINOv2 + GMU

**Run**

```text
dinov2_small_frozen_518_gmu_metadata_resizepad
```

## Configuration

- DINOv2-Small
- 518 × 518 input
- frozen image backbone
- image + metadata
- GMU fusion
- resize-and-pad preprocessing
- class-balanced focal loss

S6 tested whether pretrained DINOv2 representations could provide a strong image representation without fully fine-tuning the backbone.

## Results

| Metric | Score |
|---|---:|
| Best Validation Macro-F1 | 0.7189 |
| Test Accuracy | 0.8184 |
| Test Macro-F1 | 0.6678 |
| Test Balanced Accuracy | 0.6714 |

## Compared with S3

Best Validation Macro-F1:

```text
S3 ConvNeXtV2:  0.8471
S6 DINOv2:      0.7189
```

Difference:

```text
-0.1282
```

Test Macro-F1:

```text
S3 ConvNeXtV2:  0.8241
S6 DINOv2:      0.6678
```

Difference:

```text
-0.1563
```

## Key insights

The frozen DINOv2 setup underperformed the end-to-end trained ConvNeXtV2 models.

Using a pretrained foundation-model representation did not automatically lead to better task-specific performance in this setup.

---

# Final Ranking

## Best Validation Macro-F1

| Rank | Experiment | Model | Score |
|---:|---|---|---:|
| 1 | **S3** | **ConvNeXtV2 + GMU** | **0.8471** |
| 2 | S5 | EfficientNetV2 + GMU | 0.8335 |
| 3 | S2 | ConvNeXtV2 + Concat | 0.8229 |
| 4 | S4 | ConvNeXtV2 + GMU + Dark-Border Crop | 0.8151 |
| 5 | S1 | ConvNeXtV2 Image Only | 0.8129 |
| 6 | S6 | Frozen DINOv2 + GMU | 0.7189 |

---

## Test Macro-F1

| Rank | Experiment | Model | Score |
|---:|---|---|---:|
| 1 | **S3** | **ConvNeXtV2 + GMU** | **0.8241** |
| 2 | S1 | ConvNeXtV2 Image Only | 0.8229 |
| 3 | S2 | ConvNeXtV2 + Concat | 0.8087 |
| 4 | S4 | ConvNeXtV2 + GMU + Dark-Border Crop | 0.7831 |
| 5 | S5 | EfficientNetV2 + GMU | 0.7517 |
| 6 | S6 | Frozen DINOv2 + GMU | 0.6678 |

---

## Test Accuracy

| Rank | Experiment | Model | Score |
|---:|---|---|---:|
| 1 | S1 | ConvNeXtV2 Image Only | **0.8862** |
| 2 | S3 | ConvNeXtV2 + GMU | 0.8842 |
| 3 | S2 | ConvNeXtV2 + Concat | 0.8762 |
| 4 | S4 | ConvNeXtV2 + GMU + Dark-Border Crop | 0.8663 |
| 4 | S5 | EfficientNetV2 + GMU | 0.8663 |
| 6 | S6 | Frozen DINOv2 + GMU | 0.8184 |

---

## Test Balanced Accuracy

| Rank | Experiment | Model | Score |
|---:|---|---|---:|
| 1 | **S3** | **ConvNeXtV2 + GMU** | **0.8098** |
| 2 | S1 | ConvNeXtV2 Image Only | 0.8077 |
| 3 | S4 | ConvNeXtV2 + GMU + Dark-Border Crop | 0.7925 |
| 4 | S2 | ConvNeXtV2 + Concat | 0.7872 |
| 5 | S5 | EfficientNetV2 + GMU | 0.7667 |
| 6 | S6 | Frozen DINOv2 + GMU | 0.6714 |

---

# Selected Model

The selected model is **S3**:

```text
convnextv2_tiny_384_gmu_metadata_resizepad_strict
```

Configuration:

```text
ConvNeXtV2-Tiny
384 × 384 input
image + metadata
GMU fusion
strict resize-and-pad preprocessing
class-balanced focal loss
```

## Selected Model Metrics

| Metric | Score |
|---|---:|
| Best Validation Macro-F1 | **0.8471** |
| Test Macro-F1 | **0.8241** |
| Test Accuracy | 0.8842 |
| Test Balanced Accuracy | **0.8098** |

S3 was selected primarily because it achieved the strongest validation Macro-F1.

It also produced the highest Test Macro-F1 and Test Balanced Accuracy among the six experiments.

---

# Main Findings

## 1. The image-only baseline was already strong

S1 achieved:

```text
Test Macro-F1 = 0.8229
```

This made it a meaningful baseline rather than an intentionally weak comparison.

---

## 2. Metadata alone was not enough

S2 improved Best Validation Macro-F1:

```text
S1: 0.8129
S2: 0.8229
```

but Test Macro-F1 decreased:

```text
S1: 0.8229
S2: 0.8087
```

Simply adding metadata with concatenation did not guarantee better generalization.

---

## 3. GMU performed better than concatenation

Moving from S2 to S3 improved Best Validation Macro-F1:

```text
0.8229 → 0.8471
```

and Test Macro-F1:

```text
0.8087 → 0.8241
```

This was the clearest evidence that the multimodal fusion strategy mattered.

---

## 4. Additional preprocessing did not help

S4 tested dark-border cropping before resize-and-pad.

Its Best Validation Macro-F1 was:

```text
0.8151
```

compared with:

```text
0.8471
```

for S3.

The simpler preprocessing pipeline performed better.

---

## 5. EfficientNetV2 was competitive on validation but weaker on test

S5 had the second-highest Best Validation Macro-F1:

```text
0.8335
```

but its Test Macro-F1 was:

```text
0.7517
```

ConvNeXtV2 remained the stronger backbone for the multimodal setup.

---

## 6. Frozen DINOv2 was not competitive

S6 reached:

```text
Best Val Macro-F1 = 0.7189
Test Macro-F1     = 0.6678
```

The frozen representation did not outperform the end-to-end trained CNN models.

---

# After Model Selection

After selecting S3, the project moves from model comparison to deployment evaluation.

```text
S3 checkpoint
      ↓
ONNX export
      ↓
ONNX Runtime evaluation
      ↓
Temperature calibration
      ↓
Selective prediction
      ↓
Error analysis
      ↓
Deployment bundle
      ↓
FastAPI
      ↓
Gradio demo
```

The S1–S6 experiments are used for model selection, while the deployment stages focus only on the selected model.

---

# Summary

The main progression was:

```text
S1  Image-only baseline
      ↓
S2  Add metadata with concatenation
      ↓
S3  Replace concatenation with GMU
      ↓
S4  Test dark-border preprocessing
      ↓
S5  Compare EfficientNetV2
      ↓
S6  Compare frozen DINOv2
```

The selected configuration was:

```text
ConvNeXtV2-Tiny + Metadata + GMU
```

with:

```text
Best Validation Macro-F1: 0.8471
Test Macro-F1:            0.8241
Test Accuracy:            0.8842
Test Balanced Accuracy:   0.8098
```

The main result from these experiments was that **the fusion strategy mattered more than simply adding metadata**, while changing the backbone or adding extra preprocessing did not automatically improve performance.