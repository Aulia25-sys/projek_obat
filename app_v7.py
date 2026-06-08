import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import numpy as np
import cv2
from pathlib import Path

st.set_page_config(
    page_title="PillScan — Identifikasi Obat",
    page_icon="",
    layout="centered",
    initial_sidebar_state="collapsed",
)

CLASS_NAMES    = ["alpara", "amoxcillin", "lecozinc", "simvastatin(QL)", "simvastatin(selvim)"]
IMG_SIZE       = 224
CONF_THRESHOLD = 0.80
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]

DRUG_INFO = {
    "alpara": {
        "nama"      : "Alpara",
        "sub"       : "Paracetamol + CTM + Phenylpropanolamine",
        "indikasi"  : "Flu, demam, hidung tersumbat, nyeri kepala",
        "dosis"     : "1 tablet 3–4x/hari (dewasa), sesudah makan",
        "peringatan": "Jangan diberikan pada anak di bawah 2 tahun. Hindari alkohol.",
        "color"     : "#F59E0B",
        "dot"       : "#FDE68A",
    },
    "amoxcillin": {
        "nama"      : "Amoxicillin",
        "sub"       : "500mg — Antibiotik Penisilin",
        "indikasi"  : "Infeksi bakteri saluran napas, telinga, kulit, saluran kemih",
        "dosis"     : "500mg setiap 8 jam, 5–7 hari (sesuai resep dokter)",
        "peringatan": "HANYA dengan resep dokter. Habiskan seluruh antibiotik!",
        "color"     : "#EF4444",
        "dot"       : "#FECACA",
    },
    "lecozinc": {
        "nama"      : "Lecozinc",
        "sub"       : "Suplemen Zinc",
        "indikasi"  : "Suplementasi zinc, mendukung imunitas dan penyembuhan",
        "dosis"     : "1 tablet/hari sesudah makan",
        "peringatan": "Konsumsi bersama makanan untuk menghindari mual.",
        "color"     : "#3B82F6",
        "dot"       : "#BFDBFE",
    },
    "simvastatin(QL)": {
        "nama"      : "Simvastatin",
        "sub"       : "Merek QL — Statin Kolesterol",
        "indikasi"  : "Menurunkan kadar kolesterol LDL dan trigliserida",
        "dosis"     : "10–40mg, 1x/hari malam hari (sesuai resep dokter)",
        "peringatan": "Resep dokter wajib. Laporkan nyeri otot tidak biasa segera.",
        "color"     : "#F97316",
        "dot"       : "#FED7AA",
    },
    "simvastatin(selvim)": {
        "nama"      : "Simvastatin",
        "sub"       : "Merek Selvim — Statin Kolesterol",
        "indikasi"  : "Menurunkan kadar kolesterol LDL dan trigliserida",
        "dosis"     : "10–40mg, 1x/hari malam hari (sesuai resep dokter)",
        "peringatan": "Resep dokter wajib. Jangan konsumsi bersamaan dengan grapefruit.",
        "color"     : "#10B981",
        "dot"       : "#A7F3D0",
    },
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Clash+Display:wght@500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #060E1C !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] { background: transparent !important; box-shadow: none !important; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── HERO ── */
.ps-hero {
    background: #0A1628;
    padding: 4rem 2rem 5rem;
    position: relative;
    overflow: hidden;
}
.ps-hero::before {
    content: '';
    position: absolute;
    top: -120px; right: -120px;
    width: 480px; height: 480px;
    background: radial-gradient(circle, rgba(51,78,172,0.4) 0%, transparent 70%);
    pointer-events: none;
}
.ps-hero::after {
    content: '';
    position: absolute;
    bottom: -80px; left: -80px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(16,185,129,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.ps-nav {
    display: flex; align-items: center; justify-content: space-between;
    max-width: 800px; margin: 0 auto 3.5rem;
}
.ps-logo {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.1rem; font-weight: 600;
    color: #fff; letter-spacing: -0.01em;
    display: flex; align-items: center; gap: 0.5rem;
}
.ps-logo-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #334EAC; display: inline-block;
}
.ps-badge {
    background: rgba(51,78,172,0.2);
    border: 1px solid rgba(51,78,172,0.4);
    color: #BAD6EB; font-size: 0.7rem; font-weight: 500;
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 0.3rem 0.8rem; border-radius: 100px;
}
.ps-hero-body { max-width: 800px; margin: 0 auto; }
.ps-eyebrow {
    font-size: 0.75rem; font-weight: 500; letter-spacing: 0.15em;
    text-transform: uppercase; color: #BAD6EB; margin-bottom: 1.2rem;
    display: flex; align-items: center; gap: 0.6rem;
}
.ps-eyebrow::before {
    content: ''; display: inline-block;
    width: 24px; height: 1px; background: #334EAC;
}
.ps-title {
    font-family: 'DM Sans', sans-serif;
    font-size: clamp(2.8rem, 7vw, 4.5rem);
    font-weight: 300; line-height: 1.05;
    color: #fff; letter-spacing: -0.03em;
    margin-bottom: 1.5rem;
}
.ps-title strong {
    font-weight: 600;
    background: linear-gradient(135deg, #BAD6EB, #334EAC);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.ps-desc {
    font-size: 1rem; color: rgba(186,214,235,0.7);
    font-weight: 300; max-width: 440px; line-height: 1.7;
    margin-bottom: 2.5rem;
}
.ps-stats {
    display: flex; gap: 2.5rem; flex-wrap: wrap;
}
.ps-stat-val {
    font-size: 1.8rem; font-weight: 600; color: #fff;
    line-height: 1; margin-bottom: 0.2rem;
}
.ps-stat-label {
    font-size: 0.72rem; color: rgba(186,214,235,0.5);
    text-transform: uppercase; letter-spacing: 0.08em;
}

/* ── MAIN CONTENT ── */
.ps-main {
    max-width: 800px; margin: 0 auto;
    padding: 2.5rem 2rem 4rem;
    background: #060E1C;
}

/* ── SECTION LABEL ── */
.ps-section-label {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.15em;
    text-transform: uppercase; color: #64748B;
    margin-bottom: 0.8rem; margin-top: 2rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.ps-section-label::after {
    content: ''; flex: 1; height: 1px; background: #E2E8F0;
}
.ps-section-label.dark-bg {
    color: rgba(186,214,235,0.5);
}
.ps-section-label.dark-bg::after {
    background: rgba(51,78,172,0.3);
}

/* ── CONTROLS ── */
.ps-controls {
    background: #0A1628;
    border: 1px solid rgba(51,78,172,0.35);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 0 20px rgba(51,78,172,0.08), inset 0 1px 0 rgba(186,214,235,0.06);
    position: relative;
    overflow: hidden;
}
.ps-controls::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(51,78,172,0.6), transparent);
}

/* ── UPLOAD ZONE ── */
.ps-upload {
    background: #0A1628;
    border: 1px solid rgba(51,78,172,0.35);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 0 20px rgba(51,78,172,0.06), inset 0 1px 0 rgba(186,214,235,0.04);
    position: relative; overflow: hidden;
}
.ps-upload::before {
    content: '';
    position: absolute; bottom: 0; right: 0;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(51,78,172,0.12) 0%, transparent 70%);
    pointer-events: none;
}
[data-testid="stFileUploaderDropzone"] {
    background: rgba(13,31,60,0.8) !important;
    border: 1.5px dashed rgba(51,78,172,0.5) !important;
    border-radius: 12px !important;
    transition: all 0.2s !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #334EAC !important;
    background: rgba(51,78,172,0.08) !important;
    box-shadow: 0 0 20px rgba(51,78,172,0.15) !important;
}
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span {
    color: #BAD6EB !important;
}
[data-testid="stTabs"] [role="tablist"] {
    background: rgba(13,31,60,0.8) !important;
    border-radius: 10px !important;
    padding: 3px !important;
    gap: 2px !important;
    border: 1px solid rgba(51,78,172,0.25) !important;
}
[data-testid="stTabs"] button[role="tab"] {
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important; font-weight: 500 !important;
    color: rgba(186,214,235,0.5) !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: rgba(51,78,172,0.4) !important; color: #BAD6EB !important;
    box-shadow: 0 0 12px rgba(51,78,172,0.3), inset 0 1px 0 rgba(186,214,235,0.1) !important;
}
/* ── SELECTBOX sci-fi style ── */
[data-testid="stSelectbox"] > div > div {
    background: #0D1F3C !important;
    border: 1px solid rgba(51,78,172,0.6) !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    color: #BAD6EB !important;
    box-shadow: 0 0 0 1px rgba(51,78,172,0.2), inset 0 1px 0 rgba(186,214,235,0.05) !important;
}
[data-testid="stSelectbox"] > div > div:hover {
    border-color: rgba(51,78,172,0.9) !important;
    box-shadow: 0 0 12px rgba(51,78,172,0.25) !important;
}
[data-testid="stSelectbox"] > div > div > div {
    color: #BAD6EB !important;
}
[data-testid="stSelectbox"] svg {
    fill: #BAD6EB !important;
}
/* Dropdown list */
[data-testid="stSelectbox"] ul {
    background: #0D1F3C !important;
    border: 1px solid rgba(51,78,172,0.5) !important;
    border-radius: 10px !important;
}
[data-testid="stSelectbox"] li {
    color: #BAD6EB !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
}
[data-testid="stSelectbox"] li:hover {
    background: rgba(51,78,172,0.3) !important;
}

/* ── RESULT CARDS ── */
.ps-result {
    border-radius: 16px; padding: 2rem;
    margin: 1.2rem 0; position: relative; overflow: hidden;
}
.ps-result-verified {
    background: #0A1628;
    border: 1px solid rgba(51,78,172,0.3);
}
.ps-result-verified::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #334EAC, #10B981);
}
.ps-result-warn {
    background: #fff;
    border: 1px solid #FDE68A;
}
.ps-result-warn::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: #F59E0B;
}
.ps-result-danger {
    background: #fff;
    border: 1px solid #FECACA;
}
.ps-result-danger::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: #EF4444;
}
.ps-result-tag {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.15em;
    text-transform: uppercase; padding: 0.25rem 0.7rem;
    border-radius: 100px; display: inline-flex; align-items: center;
    gap: 0.35rem; margin-bottom: 1rem;
}
.ps-result-tag.verified {
    background: rgba(51,78,172,0.2); color: #BAD6EB;
}
.ps-result-tag.warn { background: #FEF3C7; color: #92400E; }
.ps-result-tag.danger { background: #FEE2E2; color: #991B1B; }
.ps-result-tag-dot {
    width: 5px; height: 5px; border-radius: 50%;
    background: currentColor; display: inline-block;
}
.ps-result-name {
    font-size: 2rem; font-weight: 300; letter-spacing: -0.02em;
    margin-bottom: 0.2rem; line-height: 1;
}
.ps-result-verified .ps-result-name { color: #fff; }
.ps-result-warn .ps-result-name, .ps-result-danger .ps-result-name { color: #0A1628; }
.ps-result-sub {
    font-size: 0.85rem; margin-bottom: 1rem; font-weight: 400;
}
.ps-result-verified .ps-result-sub { color: rgba(186,214,235,0.6); }
.ps-result-warn .ps-result-sub, .ps-result-danger .ps-result-sub { color: #64748B; }
.ps-conf-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    font-size: 0.8rem; font-weight: 600;
}
.ps-result-verified .ps-conf-badge { color: #BAD6EB; }
.ps-conf-num {
    font-size: 1.4rem; font-weight: 600; letter-spacing: -0.02em;
}
.ps-result-verified .ps-conf-num { color: #fff; }
.ps-result-warn .ps-conf-num { color: #F59E0B; }
.ps-result-danger .ps-conf-num { color: #EF4444; }

/* ── INFO CARD ── */
.ps-info {
    background: #0A1628; border: 1px solid rgba(51,78,172,0.3);
    border-radius: 16px; padding: 1.5rem;
    margin: 1rem 0; box-shadow: 0 0 20px rgba(51,78,172,0.08);
}
.ps-info-header {
    display: flex; align-items: center; gap: 0.8rem;
    margin-bottom: 1.2rem; padding-bottom: 1rem;
    border-bottom: 1px solid rgba(51,78,172,0.2);
}
.ps-info-dot {
    width: 10px; height: 10px; border-radius: 50%;
    flex-shrink: 0; box-shadow: 0 0 8px currentColor;
}
.ps-info-drug-name {
    font-size: 1rem; font-weight: 600; color: #E2EEFF;
}
.ps-info-drug-sub {
    font-size: 0.78rem; color: rgba(186,214,235,0.5); margin-top: 0.1rem;
}
.ps-info-row {
    display: grid; grid-template-columns: 90px 1fr;
    gap: 0.5rem 1rem; margin-bottom: 0.7rem; align-items: start;
}
.ps-info-key {
    font-size: 0.68rem; font-weight: 700; color: rgba(186,214,235,0.3);
    text-transform: uppercase; letter-spacing: 0.1em; padding-top: 2px;
}
.ps-info-val {
    font-size: 0.85rem; color: #BAD6EB; line-height: 1.5; font-weight: 400;
}
.ps-warning-strip {
    background: rgba(249,115,22,0.1); border-left: 2px solid #F97316;
    padding: 0.6rem 0.8rem; border-radius: 0 8px 8px 0;
    font-size: 0.82rem; color: #FDBA74; font-weight: 500; line-height: 1.4;
}

/* ── CONFIDENCE BARS ── */
.ps-bars {
    background: #0A1628; border: 1px solid rgba(51,78,172,0.3);
    border-radius: 16px; padding: 1.5rem;
    margin: 1rem 0; box-shadow: 0 0 20px rgba(51,78,172,0.06);
}
.ps-bars-title {
    font-size: 0.68rem; font-weight: 700; color: rgba(186,214,235,0.3);
    text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 1.2rem;
}
.ps-bar-item { margin-bottom: 0.9rem; }
.ps-bar-row {
    display: flex; justify-content: space-between; align-items: baseline;
    margin-bottom: 0.3rem;
}
.ps-bar-name { font-size: 0.83rem; font-weight: 500; color: #BAD6EB; }
.ps-bar-pct { font-size: 0.83rem; font-weight: 600; color: #E2EEFF; }
.ps-bar-track {
    background: rgba(51,78,172,0.15); border-radius: 100px; height: 5px; overflow: hidden;
}
.ps-bar-fill {
    height: 100%; border-radius: 100px;
    transition: width 0.8s cubic-bezier(.4,0,.2,1);
}
.ps-bar-fill.top { background: linear-gradient(90deg, #334EAC, #BAD6EB); box-shadow: 0 0 8px rgba(51,78,172,0.5); }
.ps-bar-fill.other { background: rgba(51,78,172,0.25); }

/* ── FOOTER ── */
.ps-footer {
    border-top: 1px solid rgba(51,78,172,0.2);
    padding: 2rem 2rem;
    max-width: 800px; margin: 0 auto;
    display: flex; justify-content: space-between; align-items: center;
    flex-wrap: wrap; gap: 0.5rem;
}
.ps-footer-left {
    font-size: 0.78rem; color: rgba(186,214,235,0.35); font-weight: 400;
}
.ps-footer-right {
    font-size: 0.72rem; color: rgba(186,214,235,0.2);
}

/* ── UPLOAD LABEL ── */
.ps-upload-label {
    font-size: 0.82rem; font-weight: 500; color: #BAD6EB;
    margin-bottom: 0.4rem; display: block;
}
.ps-upload-hint {
    font-size: 0.75rem; color: rgba(186,214,235,0.45); margin-bottom: 0.8rem;
}

/* ── IMAGE PREVIEW (kecil, tidak dominan) ── */
.ps-img-preview-wrap {
    margin-bottom: 0.5rem;
}
.ps-img-preview-wrap [data-testid="stImage"] {
    max-width: 220px !important;
}
.ps-img-preview-wrap img {
    max-height: 180px !important;
    object-fit: cover !important;
    border-radius: 12px !important;
    border: 1px solid rgba(51,78,172,0.25) !important;
    box-shadow: 0 4px 20px rgba(10,22,40,0.12), 0 0 0 1px rgba(51,78,172,0.1) !important;
}

/* ── IMAGE CAPTION ── */
.ps-img-wrap {
    border-radius: 12px; overflow: hidden;
    border: 1px solid #E8EDF5; margin-bottom: 1rem;
}
.ps-img-cap {
    font-size: 0.72rem; color: #94A3B8; text-align: center;
    padding: 0.5rem 0; background: #F8FAFF;
    border-top: 1px solid #E8EDF5;
}

/* Streamlit overrides */
.stButton > button {
    background: #0A1628 !important; color: #fff !important;
    border: 1px solid rgba(51,78,172,0.5) !important; border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important; font-weight: 500 !important;
    padding: 0.7rem 1.5rem !important; width: 100% !important;
    letter-spacing: 0.01em !important;
    transition: all 0.2s !important;
    box-shadow: 0 0 12px rgba(51,78,172,0.15) !important;
}
.stButton > button:hover {
    background: #334EAC !important;
    box-shadow: 0 0 20px rgba(51,78,172,0.4) !important;
}
/* Checkbox in dark controls area */
[data-testid="stCheckbox"] label {
    color: #BAD6EB !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
}
[data-testid="stCheckbox"] > label > span {
    color: #BAD6EB !important;
}
.stSpinner > div { color: #334EAC !important; }
hr { border-color: #E8EDF5 !important; }

/* animation */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.ps-result, .ps-info, .ps-bars {
    animation: fadeUp 0.35s ease both;
}
</style>
""", unsafe_allow_html=True)


# ─── MODEL LOADER ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model...")
def load_model(model_type: str):
    device = torch.device("cpu")
    model_path = Path(f"models/{model_type}_5kelas.pth")
    if not model_path.exists():
        st.error(f"File model tidak ditemukan: {model_path}")
        st.stop()
    if model_type == "mobilenetv3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier = nn.Sequential(
            nn.Linear(576, 256), nn.Hardswish(),
            nn.Dropout(p=0.5), nn.Linear(256, 5),
        )
    else:
        model = models.vgg11(weights=None)
        model.classifier = nn.Sequential(
            nn.Linear(25088, 256), nn.ReLU(inplace=True),
            nn.Dropout(p=0.5), nn.Linear(256, 5),
        )
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, device


# ─── PREPROCESSING ────────────────────────────────────────────────────────────
def resize_with_pad(img, target=IMG_SIZE):
    img_r = img.copy()
    img_r.thumbnail((target, target), Image.LANCZOS)
    canvas = Image.new("RGB", (target, target), (128, 128, 128))
    canvas.paste(img_r, ((target - img_r.width) // 2, (target - img_r.height) // 2))
    return canvas

def clahe_enhance(img, clip_limit=1.5, tile=4):
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(l)
    rgb = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb, mode="RGB")

def preprocess(img, use_rembg=True):
    img = img.convert("RGB")
    if use_rembg:
        try:
            from rembg import remove as rembg_remove
            out = rembg_remove(img)
            canvas = Image.new("RGB", out.size, (128, 128, 128))
            canvas.paste(out, mask=out.split()[3])
            img = canvas
        except Exception:
            pass
    img = resize_with_pad(img)
    img = clahe_enhance(img)
    return img.convert("RGB")

def predict(pil_img, model, device, use_rembg=True):
    processed = preprocess(pil_img, use_rembg)
    arr  = np.array(processed, dtype=np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    arr  = (arr - mean) / std
    tensor = torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    pred_idx = int(probs.argmax())
    return CLASS_NAMES[pred_idx], float(probs[pred_idx]), probs, processed


# ─── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ps-hero">
  <div class="ps-nav">
    <div class="ps-logo">
      <span class="ps-logo-dot"></span>
      PillScan
    </div>
    <div class="ps-badge">AI · v2.0</div>
  </div>
  <div class="ps-hero-body">
    <div class="ps-eyebrow">Sistem Identifikasi Obat</div>
    <h1 class="ps-title">Kenali obat Anda<br><strong>dengan presisi.</strong></h1>
    <p class="ps-desc">
      Unggah atau foto obat Anda. Model AI kami akan menganalisis dan mengidentifikasi
      jenisnya secara instan — dirancang untuk keamanan pasien.
    </p>
    <div class="ps-stats">
      <div>
        <div class="ps-stat-val">5</div>
        <div class="ps-stat-label">Kelas Obat</div>
      </div>
      <div>
        <div class="ps-stat-val">100%</div>
        <div class="ps-stat-label">Akurasi Uji</div>
      </div>
      <div>
        <div class="ps-stat-val">2</div>
        <div class="ps-stat-label">Model Tersedia</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── MAIN ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="ps-main">', unsafe_allow_html=True)

# Controls
st.markdown('<div class="ps-section-label">Konfigurasi</div>', unsafe_allow_html=True)
st.markdown('<div class="ps-controls">', unsafe_allow_html=True)
col1, col2 = st.columns([3, 1])
with col1:
    model_choice = st.selectbox(
        "Model",
        options=["mobilenetv3_small", "vgg11"],
        format_func=lambda x: "MobileNetV3 Small — Utama (Cepat)" if x == "mobilenetv3_small" else "VGG11 — Pembanding (Akurat)",
        label_visibility="collapsed",
    )
with col2:
    use_rembg = st.checkbox("Hapus BG", value=True, help="Aktifkan background removal")
st.markdown('</div>', unsafe_allow_html=True)

model, device = load_model(model_choice)

# Upload
st.markdown('<div class="ps-section-label">Foto Obat</div>', unsafe_allow_html=True)
st.markdown('<div class="ps-upload">', unsafe_allow_html=True)
tab_upload, tab_camera = st.tabs(["Upload Foto", "Kamera"])
pil_image = None

with tab_upload:
    st.markdown('<span class="ps-upload-label">Pilih foto dari galeri</span>', unsafe_allow_html=True)
    st.markdown('<span class="ps-upload-hint">JPG, PNG, WEBP · Pastikan obat terlihat jelas di tengah frame</span>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Foto obat", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")
    if uploaded:
        pil_image = Image.open(uploaded).convert("RGB")

with tab_camera:
    st.markdown('<span class="ps-upload-label">Ambil foto langsung</span>', unsafe_allow_html=True)
    st.markdown('<span class="ps-upload-hint">Arahkan kamera ke obat dengan pencahayaan cukup</span>', unsafe_allow_html=True)
    cam = st.camera_input("Kamera", label_visibility="collapsed")
    if cam:
        pil_image = Image.open(cam).convert("RGB")

st.markdown('</div>', unsafe_allow_html=True)

# ─── RESULT ───────────────────────────────────────────────────────────────────
if pil_image is not None:
    # Tampilkan gambar kecil di kanan atas, bukan dominan
    st.markdown("""
    <div class="ps-section-label">Foto Teranalisis</div>
    <div class="ps-img-preview-wrap">
    """, unsafe_allow_html=True)
    col_img, col_spacer = st.columns([1, 2])
    with col_img:
        st.image(pil_image, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("Menganalisis..."):
        pred_name, conf, probs, processed_img = predict(pil_image, model, device, use_rembg)

    is_verified = conf >= CONF_THRESHOLD
    info = DRUG_INFO.get(pred_name, {})

    st.markdown('<div class="ps-section-label">Hasil Analisis</div>', unsafe_allow_html=True)

    if is_verified:
        st.markdown(f"""
        <div class="ps-result ps-result-verified">
          <div class="ps-result-tag verified">
            <span class="ps-result-tag-dot"></span>Terverifikasi
          </div>
          <div class="ps-result-name">{info.get('nama', pred_name)}</div>
          <div class="ps-result-sub">{info.get('sub', '')}</div>
          <div class="ps-conf-badge">
            <span class="ps-conf-num">{conf:.1%}</span>
            <span>keyakinan</span>
          </div>
        </div>""", unsafe_allow_html=True)
        st.markdown("""<script>
        var m=new SpeechSynthesisUtterance("Obat terverifikasi. "+arguments[0]);
        m.lang="id-ID"; window.speechSynthesis.speak(m);
        </script>""", unsafe_allow_html=True)

    elif conf < 0.4:
        st.markdown(f"""
        <div class="ps-result ps-result-danger">
          <div class="ps-result-tag danger">
            <span class="ps-result-tag-dot"></span>Tidak Dikenali
          </div>
          <div class="ps-result-name">Obat Tidak Teridentifikasi</div>
          <div class="ps-result-sub">Keyakinan terlalu rendah untuk identifikasi</div>
          <div class="ps-conf-badge">
            <span class="ps-conf-num">{conf:.1%}</span>
            <span>di bawah ambang 80%</span>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="ps-result ps-result-warn">
          <div class="ps-result-tag warn">
            <span class="ps-result-tag-dot"></span>Keyakinan Rendah
          </div>
          <div class="ps-result-name">{info.get('nama', pred_name)}</div>
          <div class="ps-result-sub">Kemungkinan — perlu konfirmasi</div>
          <div class="ps-conf-badge">
            <span class="ps-conf-num">{conf:.1%}</span>
            <span>dari 80% yang dibutuhkan</span>
          </div>
        </div>""", unsafe_allow_html=True)

    # Info card
    if is_verified and info:
        dot_color = info.get('color', '#334EAC')
        st.markdown(f"""
        <div class="ps-info">
          <div class="ps-info-header">
            <div class="ps-info-dot" style="background:{dot_color}"></div>
            <div>
              <div class="ps-info-drug-name">{info['nama']}</div>
              <div class="ps-info-drug-sub">{info.get('sub','')}</div>
            </div>
          </div>
          <div class="ps-info-row">
            <span class="ps-info-key">Indikasi</span>
            <span class="ps-info-val">{info['indikasi']}</span>
          </div>
          <div class="ps-info-row">
            <span class="ps-info-key">Dosis</span>
            <span class="ps-info-val">{info['dosis']}</span>
          </div>
          <div class="ps-info-row">
            <span class="ps-info-key">Perhatian</span>
            <span class="ps-info-val">
              <div class="ps-warning-strip">{info['peringatan']}</div>
            </span>
          </div>
        </div>""", unsafe_allow_html=True)

    # Confidence bars
    bars_html = '<div class="ps-bars"><div class="ps-bars-title">Distribusi Keyakinan Model</div>'
    for i in np.argsort(probs)[::-1]:
        name = CLASS_NAMES[i]
        pct  = probs[i] * 100
        cls  = "top" if i == int(probs.argmax()) else "other"
        info_i = DRUG_INFO.get(name, {})
        display = info_i.get('nama', name)
        bars_html += f"""
        <div class="ps-bar-item">
          <div class="ps-bar-row">
            <span class="ps-bar-name">{display}</span>
            <span class="ps-bar-pct">{pct:.1f}%</span>
          </div>
          <div class="ps-bar-track">
            <div class="ps-bar-fill {cls}" style="width:{pct:.1f}%"></div>
          </div>
        </div>"""
    bars_html += "</div>"
    st.markdown(bars_html, unsafe_allow_html=True)

    with st.expander("Lihat hasil preprocessing"):
        st.image(processed_img, caption="Setelah background removal + CLAHE", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ps-footer">
  <div class="ps-footer-left">
    PillScan AI &nbsp;·&nbsp; Teknik Informatika Universitas Lampung &nbsp;·&nbsp; 2026
  </div>
  <div class="ps-footer-right">
    Hanya alat bantu — selalu konsultasikan dengan tenaga medis
  </div>
</div>
""", unsafe_allow_html=True)
