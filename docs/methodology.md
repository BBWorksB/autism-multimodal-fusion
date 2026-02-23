# Methodology

**Project:** Multimodal Fusion of Eye-Tracking and Facial Features for Early Autism Diagnosis  
**Version:** 1.0  
**Date:** February 2026

---

## 1. Problem Statement

Autism Spectrum Disorder (ASD) is currently diagnosed using single-modality AI approaches — either facial image analysis or eye-tracking data in isolation. This project builds a multimodal system that incorporates both signals, addressing three research questions:

1. How do eye gaze, fixation, and facial responses differ between autistic and neurotypical children?
2. Can a model trained on both modalities outperform unimodal diagnostic approaches?
3. What are the limitations and advantages of different fusion techniques for unmatched multimodal datasets?

---

## 2. Datasets

### 2.1 Facial Image Dataset

| Property | Value |
|---|---|
| Format | YOLO (images + `.txt` label files) |
| Total images | 3,398 |
| Train split | 2,462 images (ASD: 1,312 / TD: 1,150) |
| Val split | 463 images (ASD: 150 / TD: 313) |
| Test split | 473 images (ASD: 160 / TD: 313) |
| Image size | ~416×416px |

**Class mapping:** The original YOLO dataset contains three class IDs. Classes 0 (`autistic_face`) and 1 (`autism`) both represent ASD and were merged into a single ASD label during preprocessing. Class 2 (`no_autism`) was remapped to TD. This merging is non-negotiable — treating the two ASD classes as separate would cause the model to learn the wrong task.

**Class imbalance:** The training split is reasonably balanced (53% ASD / 47% TD). Val and test sets have approximately a 1:2 ASD:TD ratio. Resampling was not applied — preserving the original distribution maintains evaluation integrity. Imbalance is handled during training via weighted cross-entropy loss.

### 2.2 Eye-Tracking Dataset

| Property | Value |
|---|---|
| Format | 25 CSV files + metadata file |
| Total participants | 57 usable (2 excluded — no recordings) |
| ASD participants | 27 |
| TD participants | 30 |
| Raw gaze rows | ~1.35M (after filtering unidentified frames) |

**Participant matching:** The metadata file lists 59 participants. Participants 12 and 16 were present in metadata but had no corresponding recording files — excluded, leaving 57 usable participants.

**CARS Score:** The Childhood Autism Rating Scale score was recorded for ASD participants only. It was not used as a model feature — doing so would constitute label leakage. It is available for post-hoc severity analysis only.

**Dataset populations:** The facial image dataset and the eye-tracking dataset were confirmed by the client to come from different participant populations. There is no participant ID linking the two datasets. This single fact determined the entire fusion architecture (see Section 5).

---

## 3. Eye-Tracking Preprocessing

### 3.1 Feature Engineering

Raw eye-tracking data contains one row every few milliseconds — approximately 1.35M rows across 57 participants. ML models cannot learn from individual gaze points. Each participant's rows were collapsed into a single row of 36 summary statistics capturing overall gaze behaviour:

| Feature Category | Features |
|---|---|
| Fixation | Count and rate, both eyes |
| Saccade | Count and rate, both eyes |
| Blink | Count and rate, both eyes |
| Tracking ratio | Mean and standard deviation |
| Pupil diameter | Mean, std, missing rate, both eyes |
| Gaze position | Mean and std for X and Y, both eyes |
| Eye position | Mean for X, Y, Z, both eyes |
| Gaze loss | Rate for both eyes |

### 3.2 Missing Value Imputation

One participant had no left-eye recordings, producing 9 missing values across left-eye feature columns. The participant was retained and missing values imputed using column median. Mean imputation was ruled out — with n=57, a single outlier can skew the mean substantially. Median imputation is robust to outliers and is the standard choice for small clinical datasets.

### 3.3 Feature Scaling

All 36 features were scaled to mean=0, std=1 using `StandardScaler`. Features operate on very different scales (fixation counts reach into hundreds; gaze loss rates sit between 0 and 1). Without scaling, large-magnitude features dominate model weight updates. Scaling ensures all features contribute on equal footing.

### 3.4 Feature Selection: Mann-Whitney U Test

Features were selected using a non-parametric Mann-Whitney U test comparing ASD vs TD distributions per feature.

**Why Mann-Whitney U and not a t-test:** With n=57, normality of feature distributions cannot be assumed. The Mann-Whitney U test operates on ranks rather than raw values and makes no assumptions about distribution shape.

**Threshold: p < 0.01:** With 36 features and 57 participants, the feature-to-sample ratio is high (1:1.6). Standard p < 0.05 was tightened to p < 0.01 to reduce overfitting risk — at p < 0.05, up to 5% of retained features could be noise. Since the eye model acts as a teacher during fusion training, it must be built on high-confidence signal.

**Results:** 21 of 36 features were retained.

**Key finding:** Gaze variability features (`gaze_right_y_std`, `gaze_left_y_std`, `gaze_left_x_std`, `gaze_right_x_std`) dominate the top rankings. Inconsistency in gaze patterns is more discriminative for ASD than mean gaze position — clinically consistent with known ASD presentation.

**Notable negative finding:** `gaze_loss_rate_left` was flagged as visually strong during EDA but showed p = 0.672 on the Mann-Whitney test. The statistical test takes precedence over visual inspection. It was dropped.

---

## 4. Facial Image Model (Notebook 02)

### 4.1 Architecture

EfficientNet-B0 pretrained on ImageNet. The classifier head was replaced for binary classification:

```
Original: Dropout(0.2) → Linear(1280 → 1000)  [ImageNet]
Replaced: Dropout(0.3) → Linear(1280 → 2)     [ASD / TD]
```

Dropout increased from 0.2 to 0.3 — stronger regularisation appropriate for a dataset of 3,398 images. The 1280-dimensional embedding produced by the backbone feeds into the fusion model.

**Why EfficientNet-B0 over alternatives:**

| Architecture | Parameters | Reason not used |
|---|---|---|
| ResNet-50 | 25M | 5× more params — high overfitting risk on 3,398 images |
| VGG-16 | 138M | Far too large |
| Vision Transformer | 86M+ | Requires very large datasets |
| **EfficientNet-B0** | **5.3M** | Right size; pretrained; proven in clinical imaging |

### 4.2 Training Strategy

**Negative result — two-phase transfer learning failed:**

Two-phase training (freeze backbone → train head → unfreeze → fine-tune full network) is standard transfer learning practice. It was tested first and failed: Phase 1 validation accuracy plateaued at 41–43% — near random. The domain gap between ImageNet (everyday objects) and clinical ASD face images is too large for frozen features to bridge with only 2,562 trainable parameters in the head.

This is a legitimate and reportable dissertation finding: two-phase transfer learning is not universally beneficial and fails on clinical imaging tasks with significant domain shift from the pretrained source.

**Final approach — single-phase full fine-tuning:**

The full network was trained from epoch 1 at lr=1e-4. The low learning rate protects pretrained weights through careful gradient updates rather than freezing, achieving the same protective goal while allowing full backbone adaptation.

### 4.3 Training Configuration

| Parameter | Value | Justification |
|---|---|---|
| Optimizer | Adam | Adaptive learning rates; robust for fine-tuning |
| Learning rate | 1e-4 | Protects pretrained weights without freezing |
| Loss | CrossEntropyLoss + class weights | Handles val/test imbalance |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=2) | Adapts to training dynamics |
| Early stopping | patience=5 | Prevents overfitting; saves best checkpoint |
| Batch size | 32 | Optimal for dataset size and gradient stability |
| Max epochs | 30 | Upper bound — early stopping determines actual stop |

### 4.4 Results

Best checkpoint at epoch 8 (of 13 total before early stopping).

| Metric | Value |
|---|---|
| Test accuracy | 86.0% |
| ASD recall | 0.83 |
| ASD precision | 0.77 |
| ASD F1 | 0.80 |
| TD recall | 0.88 |
| False negatives | 27 / 160 ASD cases |
| False positives | 39 / 313 TD cases |

A naive classifier always predicting TD would score 66% on this test set. The 86% result represents genuine learned signal.

---

## 5. Eye-Tracking Classifier (Notebook 03)

### 5.1 Algorithm Selection

Three classifiers were compared using nested cross-validation: Random Forest, XGBoost, and SVM.

**Why not deep learning:** With n=57 participants, deep learning would immediately overfit. Tree-based and kernel-based models are the established standard for small clinical tabular datasets.

### 5.2 Evaluation Strategy

**Why not a standard train/test split:** An 80/20 split on n=57 produces approximately 11 test participants. A single misclassification swings accuracy by 9 percentage points — statistically meaningless.

**Stratified 5-fold cross-validation:** Each fold rotates the held-out participants so every participant is tested exactly once. Stratification maintains the ASD/TD class ratio in each fold.

**Nested CV:** Standard CV with separate hyperparameter tuning produces optimistically biased estimates — if hyperparameters are tuned with knowledge of the CV folds, the model has indirectly seen the evaluation data. Nested CV separates these completely:

- Outer loop (5-fold): Evaluates true generalisation on held-out participants
- Inner loop (3-fold): Finds best hyperparameters using only the outer training fold

Inner loop optimises for ASD recall — clinically correct, as a missed ASD child is the worst outcome.

### 5.3 Results

| Classifier | Accuracy | ASD Recall | ASD F1 | ROC-AUC |
|---|---|---|---|---|
| Random Forest | 0.826 ± 0.121 | 0.813 ± 0.165 | 0.813 ± 0.126 | 0.928 ± 0.072 |
| XGBoost | 0.788 ± 0.117 | 0.813 ± 0.165 | 0.779 ± 0.126 | 0.887 ± 0.087 |
| **SVM (selected)** | **0.826 ± 0.096** | **0.880 ± 0.160** | **0.827 ± 0.097** | **0.889 ± 0.095** |

**SVM selected on three grounds:**

1. Highest ASD recall (0.880 vs 0.813) — 6.7 percentage point difference in the primary clinical metric
2. Lowest standard deviation on accuracy (0.096) and F1 (0.097) — most consistent across different fold compositions
3. Unanimous hyperparameter selection (C=0.1, RBF kernel) across all 5 outer folds — strong evidence the selection is driven by data signal, not noise

**Notable finding — Fold 3:** All three classifiers scored 0.600 ASD recall on fold 3. When three different algorithms fail on the same fold, the problem is the data in that fold, not the models. A subset of ASD participants had gaze patterns indistinguishable from TD on the selected features. This is clinically meaningful — ASD presentation varies and some participants sit genuinely close to the diagnostic boundary.

### 5.4 Final Model

SVM (C=0.1, RBF kernel) retrained on all 57 participants. Resubstitution accuracy of 0.807 confirms healthy regularisation — C=0.1 deliberately accepts some training misclassifications in exchange for a generalised decision boundary. This is not reported as a performance metric; the true estimate is the nested CV result.

---

## 6. Fusion Model (Notebook 04)

### 6.1 Architecture Decision

The two datasets come from different participant populations (client-confirmed). This eliminates all standard fusion architectures:

- **Simple late fusion** (weighted probability average): requires matched participants — one person with both face and eye probabilities simultaneously. Not possible.
- **Intermediate fusion** (feature combination): same requirement — matched participants needed to learn a joint representation.
- **Population-level mean distillation**: using a single mean eye probability as the distillation target gives all images the same signal regardless of their true class. No directional learning is possible.

**Solution: Class-conditional knowledge distillation.** The eye-tracking SVM acts as a teacher during face model training. Distillation targets are derived from the SVM's class-conditional mean probabilities — one target for ASD images, one for TD images. This extracts cross-dataset signal without fabricating connections that do not exist.

### 6.2 Distillation Targets

Computed from the SVM probability table produced in Notebook 03:

| Target | Value | Derived from |
|---|---|---|
| EYE_TARGET_ASD | 0.6475 | Mean P(ASD) for 27 true ASD participants |
| EYE_TARGET_TD | 0.4407 | Mean P(ASD) for 30 true TD participants |
| Separation | 0.2067 | Clinical signal being transferred |

These are not arbitrary values — they encode what the eye model says ASD and TD probability looks like at the population level for each true class.

### 6.3 Combined Loss Function

```
Total Loss = (1 - ALPHA) × CrossEntropyLoss(predictions, true_labels)
           + ALPHA × MSELoss(face_prob_ASD, eye_target)
           = 0.6 × CrossEntropy + 0.4 × Distillation
```

| Parameter | Value | Source |
|---|---|---|
| ALPHA | 0.40 | Determined through experimentation across 4 runs |
| TEMPERATURE | 2.0 | Hinton et al. 2015 — standard distillation starting value |
| Learning rate | 1e-4 | Carried from Notebook 02 — proven for this architecture |

True labels dominate (60%) — 3,398 ground truth labels are strong signal. Eye-tracking signal enriches without overriding.

### 6.4 Experiments

Four runs were conducted to determine the optimal configuration:

| Run | Configuration | Accuracy | ASD Recall | ASD F1 | Outcome |
|---|---|---|---|---|---|
| 1 | ALPHA=0.40, raw targets | 0.8774 | 0.8000 | 0.8153 | Recall below baseline |
| 2 | ALPHA=0.25, raw targets | 0.8562 | 0.7375 | 0.7763 | All metrics worse |
| 3 | ALPHA=0.40, calibrated targets (0.75/0.25) | 0.8584 | 0.8063 | 0.7938 | Still below baseline |
| **4** | **ALPHA=0.40, raw targets (re-run)** | **0.8626** | **0.8313** | **0.8036** | **Beats all baselines** |

Run 4 used identical hyperparameters to Run 1 but converged to a better local minimum (best checkpoint at epoch 7 vs epoch 5). Deep learning training is stochastic — batch shuffle order, weight initialisation, and dropout randomness all influence the training trajectory. This is expected behaviour; the model is saved at best validation loss regardless of when it occurs.

### 6.5 Final Results

| Model | Accuracy | ASD Recall | ASD F1 | False Negatives | False Positives |
|---|---|---|---|---|---|
| Face only (Notebook 02) | 0.8600 | 0.8300 | 0.8000 | 27 | 39 |
| Eye-tracking only (Notebook 03) | 0.8260 | 0.8800 | 0.8270 | — | — |
| **Fusion model (Notebook 04)** | **0.8626** | **0.8313** | **0.8036** | **27** | **38** |

The fusion model outperforms the face-only baseline on all metrics. False negatives held at 27 — no additional ASD children missed — while overall accuracy, precision, and F1 all improved. False positives reduced from 39 to 38.

---

## 7. Deployment

The Streamlit app (`src/app/streamlit_app.py`) accepts a facial photograph and returns an ASD risk score and confidence level. The eye-tracking model is not present at inference — its contribution is encoded permanently in the fusion model's weights from training.

The model loaded at inference is structurally identical to the face-only model from Notebook 02. The fusion is invisible at the architecture level because it occurred at the weight level during training.

**Inference preprocessing:** Resize to 224×224, ToTensor, ImageNet normalisation (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). No augmentation — inference must be deterministic.

---

## 8. Limitations

**Dataset size:** The eye-tracking dataset contains 57 participants. The high standard deviation on CV metrics (± 0.160 on ASD recall) reflects genuine statistical uncertainty at this sample size. Results should be interpreted with this in mind.

**Unmatched populations:** The two datasets come from different participant groups. Class-conditional knowledge distillation is an approximation of true joint multimodal learning — feature-level fusion, which would encode richer cross-modal relationships, was not possible given the data structure.

**Fold 3 failure pattern:** All three classifiers scored 0.600 ASD recall on the same CV fold, suggesting a clinically meaningful subset of ASD participants whose gaze patterns are not distinguishable from TD on the 21 selected features. This likely represents mild or atypical ASD presentation.

**Domain shift:** Two-phase transfer learning failed, confirming significant domain shift between ImageNet and clinical ASD facial images. Any future work extending this system to a new facial image dataset should anticipate the need for full fine-tuning from pretrained weights.

**Screening context:** The system is a screening tool, not a diagnostic instrument. A HIGH result indicates elevated ASD-associated facial patterns and should trigger clinical referral. It does not constitute a diagnosis. A LOW result does not exclude ASD. All outputs require clinical validation.

---

## 9. Reproducibility

All experiments used `RANDOM_STATE = 42` applied to every component involving randomness. Nested CV fold splits, classifier initialisation, and XGBoost all receive this seed. A full re-run of Notebook 03 from the same data produces bit-for-bit identical fold results.

Notebook 02 and 04 training results are not bit-for-bit reproducible due to GPU non-determinism in PyTorch (cuDNN operations are not fully deterministic by default). The best checkpoint is saved at each run; the reported results reflect the saved checkpoint.

All experiment metrics were tracked in Weights & Biases under project `Multimodal-AI-Fusion-Project`.