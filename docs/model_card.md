# Model Card — ASD Screening Fusion Model

**Model name:** `best_fusion_model.pth`  
**Version:** 1.0  
**Date:** February 2026  
<!-- **Developed by:** Bonnie Baraka   -->
**Project:** Multimodal Fusion of Eye-Tracking and Facial Features for Early Autism Diagnosis  

---

## Model Summary

EfficientNet-B0 trained with class-conditional knowledge distillation from an eye-tracking SVM teacher. The model classifies facial photographs as indicative of ASD (Autism Spectrum Disorder) or TD (Typically Developing). At inference, only a facial photograph is required — eye-tracking data is not needed.

---

## Intended Use

### Primary use
Early ASD screening support for clinicians. The model accepts a frontal facial photograph of a child and returns an ASD risk level (HIGH / LOW) with a confidence score.

### Intended users
- Clinicians and researchers using the tool as one input in a broader diagnostic process
- Dissertation reviewers evaluating the multimodal AI approach

### Out-of-scope uses
- **Standalone diagnosis.** This model is a screening tool, not a diagnostic instrument. A HIGH result should trigger clinical referral and standardised assessment, not a diagnosis.
- **Adult subjects.** The training data consists of children. Performance on adults is unknown.
- **Non-frontal or low-quality images.** The model was trained on ~416×416px frontal facial images. Performance on occluded, rotated, or low-resolution images is not validated.
- **Real-time clinical deployment** without further validation on larger, independent datasets.

---

## Architecture

| Property | Value |
|---|---|
| Base architecture | EfficientNet-B0 |
| Pretrained weights | ImageNet (IMAGENET1K_V1) |
| Training strategy | Single-phase full fine-tuning with knowledge distillation |
| Classifier head | Dropout(0.3) → Linear(1280 → 2) |
| Parameters | ~4M (torchvision 0.25.0+cu128) |
| Input size | 224 × 224 × 3 |
| Output | Softmax probability over [TD, ASD] |
| Decision threshold | P(ASD) ≥ 0.5 → HIGH risk |

### Fusion mechanism
The eye-tracking SVM (trained on 57 participants, 21 gaze features) acts as a teacher during training. Its class-conditional mean probabilities become soft distillation targets:

| Target | Value |
|---|---|
| ASD distillation target | 0.6475 |
| TD distillation target | 0.4407 |

Combined loss: `0.6 × CrossEntropyLoss + 0.4 × MSELoss(distillation)`

At inference the eye-tracking model is absent — its clinical knowledge is encoded in the face model's weights.

---

## Training Data

| Dataset | Size | Source |
|---|---|---|
| Facial images | 3,398 images | YOLO format, train/val/test split |
| Eye-tracking | 57 participants | 25 CSV files, engineered to 21 features |

The two datasets come from **different participant populations**. There is no participant-level link between them. This determined the knowledge distillation fusion approach.

**Preprocessing:**
- YOLO classes 0 and 1 merged to ASD; class 2 remapped to TD
- Val/test class imbalance (~1:2 ASD:TD) handled with weighted cross-entropy
- ImageNet normalisation: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Augmentation during training: horizontal flip, rotation (±10°), colour jitter

---

## Evaluation

Evaluated on 473 held-out test images never seen during training or validation.

### Test set results

| Metric | Value |
|---|---|
| Accuracy | 86.26% |
| ASD Recall | 0.831 |
| ASD Precision | 0.778 |
| ASD F1 | 0.804 |
| TD Recall | 0.879 |
| TD F1 | 0.893 |
| False Negatives | 27 / 160 ASD cases missed |
| False Positives | 38 / 313 TD cases over-referred |

### Comparison against unimodal baselines

| Model | Accuracy | ASD Recall | ASD F1 |
|---|---|---|---|
| Face only (EfficientNet-B0) | 86.00% | 0.830 | 0.800 |
| Eye-tracking only (SVM) | 82.60% | 0.880 ± 0.160 | 0.827 ± 0.097 |
| **Fusion model** | **86.26%** | **0.831** | **0.804** |

The fusion model outperforms the face-only baseline on all metrics. The eye-tracking model's higher ASD recall (0.880) reflects the advantage of a balanced participant dataset — this signal was incorporated into the fusion model through distillation.

---

## Performance Considerations

**False negatives (27 / 160):** ASD children incorrectly classified as TD. In a clinical screening context this is the most critical error — a missed child receives no follow-up. The model's 0.831 ASD recall means 83.1% of true ASD cases are correctly flagged.

**False positives (38 / 313):** TD children incorrectly flagged as HIGH risk. These children would be referred for further clinical assessment. In screening, over-referral is the preferred error — false positives receive additional evaluation, not a diagnosis.

**Naive baseline:** A classifier always predicting TD would score 66% accuracy on this test set (313/473 samples are TD). The model's 86.26% represents genuine learned signal.

---

## Limitations

**Small eye-tracking dataset (n=57).** The SVM teacher was trained on 57 participants. High variance in CV metrics (± 0.160 on ASD recall) reflects genuine statistical uncertainty. The distillation targets derived from this dataset carry the same uncertainty.

**Unmatched populations.** The facial and eye-tracking datasets come from different participant groups. Class-conditional distillation approximates true joint multimodal learning — it cannot achieve the depth of fusion that matched datasets would enable.

**Domain shift.** Two-phase transfer learning (freeze backbone → fine-tune) was tested and achieved only 41% validation accuracy. Single-phase full fine-tuning was required, confirming significant domain shift from ImageNet to clinical ASD facial images.

**Atypical presentations.** All three eye-tracking classifiers scored 0.600 ASD recall on the same CV fold — suggesting a clinically meaningful subset of ASD participants whose gaze patterns are not distinguishable from TD on current features. The face model may similarly struggle with mild or atypical ASD presentations.

**Children only.** Training data consists entirely of children. The model has not been evaluated on adult subjects and should not be used for adult screening.

**Image quality sensitivity.** Performance is not validated on images that differ substantially from the training distribution: non-frontal angles, occlusion, low resolution, or images captured under poor lighting conditions.

---

## Ethical Considerations

**This model is a research prototype.** It was developed as a dissertation project and has not undergone clinical validation. It must not be used as the sole basis for any clinical decision.

**Bias and fairness.** The model was trained and evaluated on a specific pediatric dataset. Performance may differ across demographic groups (age, ethnicity, gender) that are underrepresented or absent from the training data. Fairness evaluation across subgroups was not conducted and is a necessary step before any clinical use.

**Consent and privacy.** Any deployment collecting facial images of children requires appropriate ethical oversight, informed consent from guardians, and compliance with applicable data protection regulations.

**Screening vs diagnosis.** A HIGH result indicates the model detected ASD-associated facial patterns. It does not constitute a clinical diagnosis. A LOW result does not exclude ASD. All model outputs must be reviewed by a qualified clinician before any action is taken.

---

## Technical Specifications

| Property | Value |
|---|---|
| Framework | PyTorch 2.x, torchvision 0.25.0+cu128 |
| Training hardware | NVIDIA A100-SXM4-40GB (Google Colab) |
| Training time | ~15–20 minutes (12 epochs) |
| Model file | `best_fusion_model.pth` |
| File location | `src/models/best_fusion_model.pth` |
| Inference runtime | CPU or CUDA |
| Deployment | Streamlit (`src/app/streamlit_app.py`) |

---

## How to Load

```python
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

MODEL_PATH = Path("src/models/best_fusion_model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Architecture must match training exactly
model = models.efficientnet_b0(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(1280, 2),
)

state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
```

<!-- --- -->

<!-- ## Citation

If referencing this model in academic work:

> Baraka, B. (2026). *Multimodal Fusion of Eye-Tracking and Facial Features Using AI for Early Autism Diagnosis*. Dissertation project. Available at: github.com/[your-username]/autism-multimodal-fusion -->

---

## Further Reading

- `docs/methodology.md` — full technical methodology including all preprocessing decisions, experiment logs, and justifications
- `notebooks/02_face_model1.ipynb` — EfficientNet-B0 training
- `notebooks/03_eye_tracking.ipynb` — SVM classifier and feature selection
- `notebooks/04_fusion.ipynb` — knowledge distillation fusion
- Tan, M. & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML*.
- Hinton, G., Vinyals, O. & Dean, J. (2015). Distilling the Knowledge in a Neural Network. *NIPS Deep Learning Workshop*.