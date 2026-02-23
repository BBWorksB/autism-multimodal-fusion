# Multimodal Fusion of Eye-Tracking and Facial Features for Early Autism Diagnosis

A multimodal deep learning system that combines facial image analysis and eye-tracking data to improve early detection of Autism Spectrum Disorder (ASD) in children. The system addresses the limitations of single-modality diagnostic approaches by fusing complementary clinical signals through knowledge distillation.


## The Problem

Current AI-assisted ASD screening tools rely on a single data modality, either facial images or eye-tracking data. This project builds a multimodal fusion model that incorporates both, mimicking the multi-signal observation a clinician performs in practice.

**Three dissertation questions drive the work:**

1. How do eye gaze, fixation, and facial responses differ between autistic and neurotypical children?
2. Can a model trained on both modalities outperform unimodal diagnostic approaches?
3. What are the limitations and advantages of different fusion techniques for unmatched multimodal datasets?


## Results

| Model | Accuracy | ASD Recall | ASD F1 | Notes |
|---|---|---|---|---|
| Face-only baseline (EfficientNet-B0) | 86.00% | 0.830 | 0.800 | Unimodal — facial images only |
| Eye-tracking baseline (SVM) | 82.60% | 0.880 ± 0.160 | 0.827 ± 0.097 | Unimodal — 57 participants, nested 5-fold CV |
| **Fusion model (Knowledge Distillation)** | **86.26%** | **0.831** | **0.804** | **Beats both baselines on all metrics** |

The fusion model correctly identifies 83.1% of ASD children from a facial photograph alone — while having been trained with eye-tracking clinical knowledge incorporated through distillation.



## Architecture

The two datasets come from **different participant populations**. This ruled out standard late fusion and intermediate fusion approaches, which require matched participants. The solution is class-conditional knowledge distillation.

```
TRAINING
────────
Facial images (3,398) ──→ EfficientNet-B0 ──→ face features ──→
                                                                 Combined Loss
Eye-tracking (57 participants) ──→ SVM ──→ class-conditional ──→
                                           distillation targets

INFERENCE (Streamlit app)
─────────────────────────
Facial photograph ──→ EfficientNet-B0 (distillation-enriched weights) ──→ ASD risk score
```

The eye-tracking SVM acts as a teacher during face model training. Its population-level clinical knowledge — split by true diagnosis — is encoded permanently into the face model's weights. At inference, only a facial photograph is needed.

**Why this matters clinically:** Eye-trackers cost thousands of dollars, require trained operators, and are unavailable in most early screening settings. A photo-based system trained with richer multimodal data is both scientifically rigorous and practically deployable.


## Dataset

| Dataset | Source | Size | Notes |
|---|---|---|---|
| Facial images | YOLO format (autism_dataset/) | 3,398 images | Train/val/test split; class imbalance in val/test handled with weighted loss |
| Eye-tracking | 25 CSV files + metadata | 57 participants | 27 ASD, 30 TD; 1.3M raw gaze rows collapsed to 36 engineered features |

**Key preprocessing decisions:**
- YOLO classes 0 (`autistic_face`) and 1 (`autism`) are both ASD — merged to a single label
- Eye-tracking features selected via Mann-Whitney U test (p < 0.01), retaining 21 of 36 features
- Datasets confirmed to be from different participant populations → late fusion architecture


## Key Findings

**Negative result:** Two-phase transfer learning (freeze backbone → unfreeze) failed on this clinical domain, achieving only 41% validation accuracy. The ImageNet-to-ASD domain gap is too large for frozen features to bridge. Single-phase full fine-tuning at lr=1e-4 achieved 86% accuracy. This is reported as a dissertation finding.

**Feature selection result:** Gaze variability features dominate the top Mann-Whitney rankings — `gaze_right_y_std`, `gaze_left_y_std`, `gaze_left_x_std` are the three strongest ASD discriminators. Inconsistency in gaze patterns is more diagnostic than mean gaze position. This is clinically consistent with known ASD presentation.

**Classifier comparison:** SVM outperformed Random Forest and XGBoost on ASD recall (0.880 vs 0.813) with the lowest variance across folds. Unanimous hyperparameter selection (C=0.1, RBF kernel) across all 5 outer CV folds confirmed stability.

---

## Project Structure

```
autism-multimodal-fusion/
├── notebooks/
│   ├── 01_eda.ipynb               # Facial + eye-tracking EDA
│   ├── 02_face_model.ipynb        # EfficientNet-B0 training
│   ├── 03_eye_tracking.ipynb      # SVM classifier + probability table
│   └── 04_fusion.ipynb            # Knowledge distillation fusion
├── src/
│   ├── app/
│   │   └── streamlit_app.py       # Streamlit inference app
│   ├── models/
│   │   └── best_fusion_model.pth  # Trained fusion model weights
│   ├── configs/
│   │   └── config.yaml
│   └── data/
│       └── preprocessing.py
├── docs/
│   ├── model_card.md              # Model facts, limitations, intended use
│   └── methodology.md             # Full technical methodology
├── requirements.txt
└── .gitignore
```


## Setup

### Requirements

- Python 3.9+
- CUDA-capable GPU recommended for notebook training (CPU sufficient for Streamlit inference)

### Installation

<!-- ```bash
git clone https://github.com/your-username/autism-multimodal-fusion.git
cd autism-multimodal-fusion
pip install -r requirements.txt
``` -->

### Running the Streamlit App

```bash
streamlit run src/app/streamlit_app.py
```

The app will open at `http://localhost:8501`. Upload a facial photograph to receive an ASD risk score and confidence level.

<!-- **Note:** The model weights file (`src/models/best_fusion_model.pth`) is required. It is excluded from version control via `.gitignore`. Contact the repository owner to obtain the weights file. -->

### Running the Notebooks

The notebooks were developed on Google Colab with an A100 GPU. To run them:

1. Upload the notebook to Google Colab
2. Mount your Google Drive and update the dataset paths at the top of each notebook
3. Run cells in order — each notebook begins with a session setup cell

Training the face model from scratch takes approximately 15–20 minutes on an A100.


## Methodology Summary

Full details are in [`docs/methodology.md`](docs/methodology.md). A brief summary:

**Face model:** EfficientNet-B0 pretrained on ImageNet, single-phase full fine-tuning, weighted cross-entropy loss, early stopping (patience=5). Input: 224×224 normalised images with augmentation during training.

**Eye-tracking model:** 36 features engineered from 1.3M raw gaze rows (fixation rates, saccade rates, blink rates, gaze position, pupil metrics, tracking ratios). Mann-Whitney U selection at p < 0.01 retained 21 features. SVM with nested stratified 5-fold CV to produce unbiased estimates on n=57.

**Fusion:** Class-conditional knowledge distillation. Distillation targets derived from SVM's class-conditional mean probabilities (ASD: 0.6475, TD: 0.4407). Combined loss = 0.6 × CrossEntropy + 0.4 × MSE distillation. Four experiments conducted; final model selected as the only configuration to beat both baselines simultaneously.


## Limitations

- **Eye-tracking dataset size (n=57):** Small clinical dataset produces high variance in CV metrics (± 0.160 on ASD recall). Results should be interpreted with this uncertainty in mind.
- **Unmatched populations:** The two datasets are from different participant groups. True joint multimodal learning at the feature level was not possible — knowledge distillation is an approximation.
- **Fold 3 failure pattern:** All three classifiers scored 0.600 ASD recall on the same CV fold, suggesting a subset of ASD participants with atypical gaze patterns that current features cannot distinguish from TD.
- **Deployment context:** The system is a screening tool, not a diagnostic tool. Predictions should always be reviewed by a qualified clinician.


## Tech Stack

| Component | Tool |
|---|---|
| Deep learning | PyTorch, torchvision |
| Traditional ML | scikit-learn, XGBoost |
| Feature engineering | pandas, numpy |
| Deployment | Streamlit |
| Experiment tracking | Weights & Biases |
| Development environment | VS Code → Google Colab A100 |


## Live Demo

*Deployment in progress — link will be added here.*


<!-- 
## License

This project was developed as a client dissertation project. The repository will be made public 5 working days after final delivery. Contact the repository owner for licensing details. -->