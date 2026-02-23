"""
Multimodal ASD Diagnosis System — Streamlit App
Version: v1.0 — Fusion model connected

Architecture:
    EfficientNet-B0 trained with class-conditional knowledge distillation
    from eye-tracking SVM teacher. At inference: facial image only.
    Model: best_fusion_model.pth  (EfficientNet-B0, 4M params,
           Dropout(0.3) → Linear(1280→2), torchvision 0.25.0+cu128)

Preprocessing:
    Resize → 224×224, ImageNet normalisation
    mean=[0.485, 0.456, 0.406]  std=[0.229, 0.224, 0.225]

Model path (relative to project root):
    autism-multimodal-fusion/
    └── src/
        └── models/
            └── best_fusion_model.pth

To download from Google Drive (run once in Colab, then copy to local):
    from google.colab import drive
    drive.mount('/content/drive')
    import shutil, os
    os.makedirs('src/models', exist_ok=True)
    shutil.copy(
        '/content/drive/MyDrive/best_fusion_model.pth',
        'src/models/best_fusion_model.pth'
    )
"""

import os
import io
import time
from pathlib import Path

import streamlit as st
from PIL import Image

# Torch imports — guarded so the app shows a clear error if PyTorch is missing
# rather than a cryptic import stack trace
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── PATHS ───────────────────────────
# All paths relative to this file — works from any machine after cloning repo.
# app/streamlit_app.py → parent = app/ → parent = project root
APP_DIR    = Path(__file__).parent
MODEL_PATH = APP_DIR.parent / "models" / "best_fusion_model.pth"

# ── PAGE CONFIG ─────────────────────
st.set_page_config(
    page_title="ASD Screening Tool",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ─────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Global ────────────────────── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background-color: #07111f;
    color: #d0dde8;
}
.stApp {
    background:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(0,150,130,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(30,80,160,0.10) 0%, transparent 55%),
        linear-gradient(180deg, #07111f 0%, #0a1628 50%, #07111f 100%);
    min-height: 100vh;
}

/* ── Container ─────────────────── */
.main .block-container {
    max-width: 860px !important;
    padding: 0 1.5rem 4rem 1.5rem !important;
}

/* ── Header ────────────────────── */
.app-header {
    text-align: center;
    padding: 3.5rem 2rem 2.5rem 2rem;
    position: relative;
}
.header-glow {
    position: absolute;
    top: 0; left: 50%; transform: translateX(-50%);
    width: 420px; height: 2px;
    background: linear-gradient(90deg, transparent, #00c9b1, #4f8ef7, transparent);
}
.header-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #00c9b1;
    margin-bottom: 1rem;
}
.header-title {
    font-size: 3rem;
    font-weight: 700;
    color: #eaf2ff;
    margin-bottom: 0.6rem;
    letter-spacing: -0.02em;
    line-height: 1.05;
}
.header-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #4a6a84;
    margin-bottom: 1.8rem;
}
.header-badges {
    display: flex;
    gap: 0.6rem;
    justify-content: center;
    flex-wrap: wrap;
}
.hbadge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.3rem 0.9rem;
    border-radius: 3px;
}
.hbadge-teal  { background: rgba(0,201,177,0.10);  color: #00c9b1; border: 1px solid rgba(0,201,177,0.30); }
.hbadge-blue  { background: rgba(79,142,247,0.10); color: #6da0f8; border: 1px solid rgba(79,142,247,0.30); }
.hbadge-green { background: rgba(0,210,100,0.10);  color: #00d264; border: 1px solid rgba(0,210,100,0.30); }

/* ── Section divider ───────────── */
.section-divider {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 2rem 0 1rem 0;
}
.section-divider-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, transparent, #1a2d45, transparent);
}
.section-divider-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #6da0f8;
    font-weight: 500;
}

/* ── File uploader ─────────────── */
section[data-testid="stFileUploader"] {
    background: rgba(10,16,32,0.85) !important;
    border: 1.5px dashed #1d3050 !important;
    border-radius: 14px !important;
    transition: border-color 0.2s !important;
}
section[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,201,177,0.45) !important;
}

/* ── Uploaded image ────────────── */
[data-testid="stImage"] img {
    border-radius: 10px;
    border: 1px solid #1a2d45;
    box-shadow: 0 8px 40px rgba(0,0,0,0.55);
}

/* ── Primary result block ──────── */
.result-block { border-radius: 12px; padding: 1.4rem 1.6rem 1.2rem 1.6rem; margin-bottom: 1rem; position: relative; overflow: hidden; }
.result-block::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px; }
.result-block-high { background: linear-gradient(135deg, rgba(28,10,10,0.95), rgba(18,8,8,0.95)); border: 1px solid rgba(255,112,112,0.25); }
.result-block-high::before { background: linear-gradient(90deg, transparent, rgba(255,112,112,0.5), transparent); }
.result-block-low  { background: linear-gradient(135deg, rgba(6,22,20,0.95), rgba(5,16,14,0.95)); border: 1px solid rgba(0,201,177,0.22); }
.result-block-low::before  { background: linear-gradient(90deg, transparent, rgba(0,201,177,0.45), transparent); }
.result-outcome-label { font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; letter-spacing: 0.2em; text-transform: uppercase; color: #3d5a78; margin-bottom: 0.7rem; }
.result-main-row { display: flex; align-items: baseline; gap: 0.9rem; margin-bottom: 0.5rem; }
.result-risk-high { font-family: 'JetBrains Mono', monospace; font-size: 2.8rem; font-weight: 700; color: #ff7070; letter-spacing: 0.04em; line-height: 1; }
.result-risk-low  { font-family: 'JetBrains Mono', monospace; font-size: 2.8rem; font-weight: 700; color: #00c9b1; letter-spacing: 0.04em; line-height: 1; }
.result-pct-high  { font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 500; color: rgba(255,112,112,0.6); line-height: 1; }
.result-pct-low   { font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 500; color: rgba(0,201,177,0.55); line-height: 1; }
.result-sub { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #3d5a78; margin-top: 0.2rem; }

/* ── Clinical alert ────────────── */
.clinical-alert-high { background: rgba(255,112,112,0.05); border: 1px solid rgba(255,112,112,0.2); border-left: 3px solid #ff7070; border-radius: 0 8px 8px 0; padding: 0.85rem 1rem; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #c07070; line-height: 1.6; margin-bottom: 0.8rem; }
.clinical-alert-low  { background: rgba(79,142,247,0.05);  border: 1px solid rgba(79,142,247,0.2);  border-left: 3px solid #4f8ef7; border-radius: 0 8px 8px 0; padding: 0.85rem 1rem; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #6090c0; line-height: 1.6; margin-bottom: 0.8rem; }

/* ── Progress bar ──────────────── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #006b60, #00c9b1) !important;
    border-radius: 99px !important;
}
[data-testid="stProgress"] > div {
    background: #0a1020 !important;
    border: 1px solid #1a2d45 !important;
    border-radius: 99px !important;
    height: 8px !important;
}

/* ── Caption ───────────────────── */
[data-testid="stCaptionContainer"] p {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.67rem !important;
    color: #2a4060 !important;
    line-height: 1.7 !important;
}

/* ── Model error / warning box ─── */
.model-error {
    background: rgba(251,188,5,0.05);
    border: 1px solid rgba(251,188,5,0.2);
    border-left: 3px solid #fbc005;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #c8a030;
    line-height: 1.8;
    margin: 1rem 0;
}

/* ── Empty state ───────────────── */
.empty-state {
    background: linear-gradient(135deg, rgba(14,26,46,0.6), rgba(10,18,32,0.6));
    border: 1px dashed #1a2d45;
    border-radius: 16px;
    padding: 4.2rem 2.4rem;
    text-align: center;
}
.empty-step {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #2a4060;
}
.empty-icon { font-size: 2.4rem; opacity: 0.15; margin: 1rem 0; }
.empty-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #3b5672;
}

/* ── Cleanup ───────────────────── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── MODEL LOADER ────────────────────
# @st.cache_resource: loads once per session, survives reruns.
# Without this the model reloads on every file upload or page interaction.

@st.cache_resource(show_spinner=False)
def load_model(model_path: Path):
    """
    Load best_fusion_model.pth.
    Returns (model, device) on success, (None, error_str) on failure.

    Architecture must match Notebook 04 exactly:
        efficientnet_b0 with classifier replaced by
        Sequential(Dropout(0.3), Linear(1280, 2))
    """
    if not TORCH_AVAILABLE:
        return None, "PyTorch not installed. Run: pip install torch torchvision"

    if not model_path.exists():
        msg = (
            "Model weights not found.<br>"
            "Expected: <code>" + str(model_path) + "</code><br><br>"
            "Ensure <code>best_fusion_model.pth</code> is present at the path above."
        )
        return None, msg

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Rebuild architecture — must match training in Notebook 04
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(1280, 2),
        )

        # weights_only=True: safe loading, avoids arbitrary code execution
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()   # disables dropout + batchnorm training behaviour

        return model, device

    except Exception as exc:
        return None, "Failed to load model: " + str(exc)


# ── INFERENCE TRANSFORM ─────────────
# Identical to val/test transform in Notebooks 02 and 04.
# No augmentation — inference must be deterministic.

def get_transform():
    if not TORCH_AVAILABLE:
        return None
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def predict(model, device, pil_image: Image.Image) -> float:
    """
    Run inference on a PIL image. Returns face_prob: float in [0, 1].
    face_prob = P(ASD) from softmax output, class index 1.
    Decision threshold: >= 0.5 → HIGH risk.
    """
    transform = get_transform()
    tensor = transform(pil_image)        # [3, 224, 224]
    tensor = tensor.unsqueeze(0)         # [1, 3, 224, 224]
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)           # [1, 2]  raw scores
        probs  = torch.softmax(logits, dim=1)  # [1, 2]  probabilities
        face_prob = probs[0, 1].item()   # P(ASD)

    return face_prob


# ── HEADER ──────────────────────────
st.markdown("""
<div class="app-header">
    <div class="header-glow"></div>
    <div class="header-eyebrow">Research Prototype &middot; v1.0</div>
    <div class="header-title">&#129504; ASD Screening System</div>
    <div class="header-sub">Multimodal AI Diagnostic Tool &mdash; Facial Analysis</div>
    <div class="header-badges">
        <span class="hbadge hbadge-teal">EfficientNet-B0</span>
        <span class="hbadge hbadge-blue">86.26% Fusion Accuracy</span>
        <span class="hbadge hbadge-teal">0.83 ASD Recall</span>
        <span class="hbadge hbadge-green">Fusion Model &middot; v1.0</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── LOAD MODEL (once, cached) ───────
model_result, device_or_err = load_model(MODEL_PATH)
model_ready = model_result is not None

if not model_ready:
    st.markdown(
        '<div class="model-error">'
        '<strong>Model weights not loaded</strong><br><br>'
        + device_or_err +
        '</div>',
        unsafe_allow_html=True,
    )


# ── IMAGE INPUT ─────────────────────
st.markdown("""
<div class="section-divider">
    <div class="section-divider-line"></div>
    <div class="section-divider-label">Image Input</div>
    <div class="section-divider-line"></div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload facial image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)


# ── MAIN: post-upload ───────────────
if uploaded_file:

    # Detect whether this is a new file or a rerun of the same session
    is_new_file = (
        "last_filename" not in st.session_state
        or st.session_state.last_filename != uploaded_file.name
    )

    image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")

    if is_new_file:
        if model_ready:
            with st.spinner("Running inference…"):
                result_prob = predict(model_result, device_or_err, image)
        else:
            # Model not loaded — show UI with placeholder so layout is visible
            with st.spinner("Preprocessing image…"):
                time.sleep(0.6)
            result_prob = None

        st.session_state.last_filename = uploaded_file.name
        st.session_state.last_prob     = result_prob

    # Always read from session_state for rendering — handles both new and rerun
    face_prob = st.session_state.get("last_prob", None)

    col_l, col_r = st.columns([5, 4], gap="large")

    # ── LEFT: image ─────────────────
    with col_l:
        st.image(image, use_container_width=True)

    # ── RIGHT: results ──────────────
    with col_r:

        st.markdown("""
        <div class="section-divider">
            <div class="section-divider-line"></div>
            <div class="section-divider-label">Screening Result</div>
            <div class="section-divider-line"></div>
        </div>
        """, unsafe_allow_html=True)

        if face_prob is not None:
            # ── Real model output 
            asd_risk = "HIGH" if face_prob >= 0.5 else "LOW"

            # confidence_pct = confidence in the predicted class:
            #   HIGH → P(ASD) as %
            #   LOW  → P(TD) = 1 - P(ASD) as %
            confidence_pct = (
                int(face_prob * 100)
                if face_prob >= 0.5
                else int((1 - face_prob) * 100)
            )
        else:
            # Model unavailable — neutral placeholder, UI still renders
            asd_risk       = "—"
            confidence_pct = 0
            face_prob      = 0.0

        is_high   = asd_risk == "HIGH"
        block_cls = "result-block-high" if is_high else "result-block-low"
        risk_cls  = "result-risk-high"  if is_high else "result-risk-low"
        pct_cls   = "result-pct-high"   if is_high else "result-pct-low"
        alert_cls = "clinical-alert-high" if is_high else "clinical-alert-low"
        sub_text  = (
            "Elevated ASD indicators detected"
            if is_high else "Within typical developmental range"
        )
        alert_msg = (
            "Clinical review and standardised assessment required."
            if is_high else
            "No strong ASD indicators detected. A LOW result does not "
            "exclude ASD — clinical judgement remains essential."
        )

        # Primary result block — HIGH/LOW large, confidence % to its right
        st.markdown(
            '<div class="result-block ' + block_cls + '">'
            '<div class="result-outcome-label">ASD Risk Level</div>'
            '<div class="result-main-row">'
            '<span class="' + risk_cls + '">' + asd_risk + '</span>'
            '<span class="' + pct_cls + '">' + str(confidence_pct) + '%</span>'
            '</div>'
            '<div class="result-sub">' + sub_text + '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Confidence bar
        st.progress(confidence_pct / 100)
        st.markdown(
            '<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.68rem;'
            'color:#3d5a78;margin-top:-0.3rem;margin-bottom:0.8rem;letter-spacing:0.06em;">'
            'Model confidence: ' + str(confidence_pct) + '%'
            '</p>',
            unsafe_allow_html=True,
        )

        # Clinical alert
        st.markdown(
            '<div class="' + alert_cls + '">' + alert_msg + '</div>',
            unsafe_allow_html=True,
        )

        # Disclaimer
        st.caption(
            "Research prototype — not a diagnostic instrument. "
            "Confidence reflects the model's probability estimate for ASD-associated "
            "facial patterns. All outputs require clinical validation before any "
            "clinical action is taken."
        )


# ── EMPTY STATE ─────────────────────
else:
    # Clear session cache when file is removed so next upload reruns inference
    st.session_state.pop("last_filename", None)
    st.session_state.pop("last_prob", None)

    st.markdown("""
    <div class="section-divider">
        <div class="section-divider-line"></div>
        <div class="section-divider-label">Screening Result</div>
        <div class="section-divider-line"></div>
    </div>
    <div class="empty-state">
        <div class="empty-step">Step 1 of 2</div>
        <div class="empty-icon">&#128247;</div>
        <div class="empty-text">Upload a clear frontal facial image to begin ASD screening</div>
    </div>
    """, unsafe_allow_html=True)