import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import numpy as np
import cv2
from pathlib import Path

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PillScan AI — Pemeriksa Obat",
    page_icon="💊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
CLASS_NAMES    = ["alpara", "amoxcillin", "lecozinc", "simvastatin(QL)", "simvastatin(selvim)"]
IMG_SIZE       = 224
CONF_THRESHOLD = 0.80
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]

DRUG_INFO = {
    "alpara": {
        "nama"      : "Alpara",
        "indikasi"  : "Flu, demam, hidung tersumbat, nyeri kepala",
        "dosis"     : "1 tablet 3-4x/hari (dewasa), sesudah makan",
        "peringatan": "Jangan diberikan pada anak <2 tahun. Hindari alkohol.",
        "emoji"     : "🟡",
    },
    "amoxcillin": {
        "nama"      : "Amoxicillin 500mg",
        "indikasi"  : "Infeksi bakteri saluran napas, telinga, kulit, saluran kemih",
        "dosis"     : "500mg setiap 8 jam selama 5-7 hari (sesuai resep dokter)",
        "peringatan": "HANYA dengan resep dokter. Habiskan seluruh antibiotik!",
        "emoji"     : "🔴",
    },
    "lecozinc": {
        "nama"      : "Lecozinc (Suplemen Zinc)",
        "indikasi"  : "Suplemen zinc, mendukung imunitas dan pertumbuhan",
        "dosis"     : "1 tablet/hari sesudah makan",
        "peringatan": "Konsumsi bersama makanan untuk menghindari mual.",
        "emoji"     : "🔵",
    },
    "simvastatin(QL)": {
        "nama"      : "Simvastatin (Merek QL)",
        "indikasi"  : "Menurunkan kadar kolesterol darah",
        "dosis"     : "1 tablet malam hari sebelum tidur",
        "peringatan": "Merek QL — pastikan sesuai jadwal dokter Anda.",
        "emoji"     : "🟠",
    },
    "simvastatin(selvim)": {
        "nama"      : "Simvastatin (Merek Selvim)",
        "indikasi"  : "Menurunkan kadar kolesterol darah",
        "dosis"     : "1 tablet malam hari sebelum tidur",
        "peringatan": "Merek Selvim — pastikan sesuai jadwal dokter Anda.",
        "emoji"     : "🟢",
    },
}

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Syne:wght@700;800&display=swap');
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [data-testid="stAppViewContainer"] { background: #f0f4ff !important; }
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #e8eeff 0%, #f5f0ff 50%, #e8f8ff 100%) !important;
    font-family: 'Plus Jakarta Sans', sans-serif; min-height: 100vh;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }
.block-container { padding: 2rem 1.5rem 4rem !important; max-width: 780px !important; }
.hero { text-align: center; padding: 2.5rem 1rem 1.5rem; }
.hero-badge {
    display: inline-block; background: rgba(99,102,241,0.12); color: #6366f1;
    font-size: 0.75rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
    padding: 0.4rem 1rem; border-radius: 100px; margin-bottom: 1.2rem;
    border: 1px solid rgba(99,102,241,0.2);
}
.hero-title {
    font-family: 'Syne', sans-serif; font-size: clamp(2.2rem, 6vw, 3.2rem);
    font-weight: 800; line-height: 1.1; color: #0f172a;
    letter-spacing: -0.02em; margin-bottom: 0.8rem;
}
.hero-title span {
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #06b6d4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-sub { font-size: 1rem; color: #64748b; font-weight: 400; max-width: 440px; margin: 0 auto 0.5rem; line-height: 1.6; }
.upload-card {
    background: rgba(255,255,255,0.85); backdrop-filter: blur(20px);
    border: 1px solid rgba(99,102,241,0.15); border-radius: 24px; padding: 2rem 1.5rem; margin: 1.5rem 0;
    box-shadow: 0 8px 32px rgba(99,102,241,0.08), 0 1px 0 rgba(255,255,255,0.8) inset;
}
.upload-label { font-size: 1rem; font-weight: 600; color: #1e293b; margin-bottom: 0.5rem; display: block; }
.upload-hint { font-size: 0.82rem; color: #94a3b8; margin-bottom: 1rem; }
[data-testid="stFileUploaderDropzone"] {
    background: linear-gradient(135deg, rgba(99,102,241,0.04), rgba(139,92,246,0.04)) !important;
    border: 2px dashed rgba(99,102,241,0.3) !important; border-radius: 16px !important;
    padding: 2rem !important; transition: all 0.2s ease !important;
}
.stButton > button {
    width: 100% !important; background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important; border: none !important; border-radius: 14px !important;
    padding: 0.9rem 2rem !important; font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 1rem !important; font-weight: 700 !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.3) !important;
}
.result-safe {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7); border: 2px solid #22c55e;
    border-radius: 20px; padding: 1.8rem; margin: 1rem 0; animation: pulseGreen 2s ease-in-out 3;
}
.result-danger {
    background: linear-gradient(135deg, #fff1f2, #ffe4e6); border: 2px solid #f43f5e;
    border-radius: 20px; padding: 1.8rem; margin: 1rem 0;
}
.result-warning {
    background: linear-gradient(135deg, #fffbeb, #fef3c7); border: 2px solid #f59e0b;
    border-radius: 20px; padding: 1.8rem; margin: 1rem 0;
}
@keyframes pulseGreen {
    0%,100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); }
    50% { box-shadow: 0 0 0 12px rgba(34,197,94,0.15); }
}
.result-icon { font-size: 2.5rem; margin-bottom: 0.6rem; }
.result-status { font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 800; margin-bottom: 0.3rem; }
.result-safe .result-status { color: #15803d; }
.result-danger .result-status { color: #be123c; }
.result-warning .result-status { color: #b45309; }
.result-drug-name { font-size: 1.2rem; font-weight: 700; color: #1e293b; margin-bottom: 0.3rem; }
.result-conf { font-size: 0.9rem; color: #64748b; font-weight: 500; }
.info-card { background: rgba(255,255,255,0.9); border: 1px solid rgba(0,0,0,0.07); border-radius: 16px; padding: 1.4rem; margin: 1rem 0; }
.info-row { display: flex; gap: 0.8rem; margin-bottom: 0.8rem; align-items: flex-start; }
.info-label { font-size: 0.75rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; min-width: 80px; padding-top: 2px; }
.info-value { font-size: 0.92rem; color: #1e293b; font-weight: 500; line-height: 1.5; }
.warning-text { background: rgba(239,68,68,0.08); border-left: 3px solid #ef4444; padding: 0.6rem 0.8rem; border-radius: 0 8px 8px 0; font-size: 0.85rem; color: #b91c1c; font-weight: 600; }
.conf-bar-wrap { background: rgba(255,255,255,0.9); border: 1px solid rgba(0,0,0,0.07); border-radius: 16px; padding: 1.4rem; margin: 1rem 0; }
.conf-bar-title { font-size: 0.78rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 1rem; }
.conf-item { margin-bottom: 0.75rem; }
.conf-item-label { display: flex; justify-content: space-between; margin-bottom: 0.3rem; }
.conf-item-name { font-size: 0.85rem; font-weight: 600; color: #1e293b; }
.conf-item-pct  { font-size: 0.85rem; font-weight: 700; color: #6366f1; }
.conf-bar-bg { background: #e2e8f0; border-radius: 100px; height: 8px; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 100px; transition: width 1s cubic-bezier(.4,0,.2,1); }
.conf-bar-fill.top { background: linear-gradient(90deg,#6366f1,#8b5cf6); }
.conf-bar-fill.other { background: #cbd5e1; }
.footer { text-align: center; padding: 2rem 0 1rem; color: #94a3b8; font-size: 0.78rem; }
hr { border-color: rgba(99,102,241,0.1) !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ─── MODEL LOADER ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Memuat model AI...")
def load_model(model_type: str):
    device = torch.device("cpu")
    model_path = Path(f"models/{model_type}_5kelas.pth")
    if not model_path.exists():
        st.error(f"❌ File model tidak ditemukan: {model_path}")
        st.stop()

    if model_type == "mobilenetv3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier = nn.Sequential(
            nn.Linear(576, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 5),
        )
    else:
        model = models.vgg11(weights=None)
        model.classifier = nn.Sequential(
            nn.Linear(25088, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 5),
        )

    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, device


# ─── PREPROCESSING ────────────────────────────────────────────────────────────
def resize_with_pad(img: Image.Image, target: int = IMG_SIZE) -> Image.Image:
    img_r = img.copy()
    img_r.thumbnail((target, target), Image.LANCZOS)
    canvas = Image.new("RGB", (target, target), (128, 128, 128))
    canvas.paste(img_r, ((target - img_r.width) // 2, (target - img_r.height) // 2))
    return canvas


def clahe_enhance(img: Image.Image, clip_limit: float = 1.5, tile: int = 4) -> Image.Image:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(l)
    rgb = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb, mode="RGB")


def preprocess(img: Image.Image, use_rembg: bool = True) -> Image.Image:
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


# ─── INFERENCE ────────────────────────────────────────────────────────────────
def predict(pil_img: Image.Image, model, device, use_rembg: bool = True):
    processed = preprocess(pil_img, use_rembg)

    arr  = np.array(processed, dtype=np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    arr  = (arr - mean) / std
    tensor = torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

    pred_idx  = int(probs.argmax())
    return CLASS_NAMES[pred_idx], float(probs[pred_idx]), probs, processed


# ─── UI ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">💊 AI Pemeriksa Obat — 5 Kelas</div>
  <h1 class="hero-title">Periksa Obat Anda<br><span>dengan Kamera</span></h1>
  <p class="hero-sub">Foto obat Anda dan AI kami akan memverifikasi jenisnya secara instan — dirancang untuk keamanan lansia.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    model_choice = st.selectbox(
        "Model AI",
        options=["mobilenetv3_small", "vgg11"],
        format_func=lambda x: "🚀 MobileNetV3 Small (Utama)" if x == "mobilenetv3_small" else "🔬 VGG11 (Pembanding)",
    )
with col2:
    use_rembg = st.checkbox("Hapus Background", value=True)

model, device = load_model(model_choice)

st.markdown('<div class="upload-card">', unsafe_allow_html=True)
tab_upload, tab_camera = st.tabs(["📁  Upload Foto", "📷  Kamera Langsung"])
pil_image = None

with tab_upload:
    st.markdown('<span class="upload-label">Pilih foto obat dari galeri</span>', unsafe_allow_html=True)
    st.markdown('<span class="upload-hint">Format: JPG, PNG, WEBP · Pastikan obat terlihat jelas</span>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    if uploaded:
        pil_image = Image.open(uploaded).convert("RGB")

with tab_camera:
    st.markdown('<span class="upload-label">Foto langsung menggunakan kamera</span>', unsafe_allow_html=True)
    st.markdown('<span class="upload-hint">Arahkan kamera ke obat, pastikan pencahayaan cukup</span>', unsafe_allow_html=True)
    cam = st.camera_input("", label_visibility="collapsed")
    if cam:
        pil_image = Image.open(cam).convert("RGB")

st.markdown('</div>', unsafe_allow_html=True)

if pil_image is not None:
    col_img, col_blank = st.columns([1, 1])
    with col_img:
        st.image(pil_image, caption="Foto yang Anda ambil", use_column_width=True)

    with st.spinner("🔍 AI sedang menganalisis obat..."):
        pred_name, conf, probs, processed_img = predict(pil_image, model, device, use_rembg)

    is_verified = conf >= CONF_THRESHOLD
    info        = DRUG_INFO.get(pred_name, {})

    if is_verified:
        st.markdown(f"""
        <div class="result-safe">
          <div class="result-icon">✅</div>
          <div class="result-status">Obat Terverifikasi!</div>
          <div class="result-drug-name">{info.get('emoji','💊')} {info.get('nama', pred_name)}</div>
          <div class="result-conf">Tingkat keyakinan: <b>{conf:.1%}</b> (di atas ambang 80%)</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("""<script>
        var m=new SpeechSynthesisUtterance("Obat terverifikasi benar, silakan diminum.");
        m.lang="id-ID";m.rate=0.9;window.speechSynthesis.speak(m);
        </script>""", unsafe_allow_html=True)
    elif conf < 0.4:
        st.markdown(f"""
        <div class="result-danger">
          <div class="result-icon">🚨</div>
          <div class="result-status">Peringatan! Obat Tidak Dikenali</div>
          <div class="result-drug-name">Keyakinan terlalu rendah — ambil ulang foto</div>
          <div class="result-conf">Keyakinan tertinggi: <b>{conf:.1%}</b> (minimum 80%)</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("""<script>
        var m=new SpeechSynthesisUtterance("Peringatan! Obat ini tidak jelas atau salah. Tolong periksa kembali.");
        m.lang="id-ID";m.rate=0.9;window.speechSynthesis.speak(m);
        </script>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-warning">
          <div class="result-icon">⚠️</div>
          <div class="result-status">Keyakinan Kurang</div>
          <div class="result-drug-name">Kemungkinan: {info.get('nama', pred_name)} — tapi belum pasti</div>
          <div class="result-conf">Keyakinan: <b>{conf:.1%}</b> · Diperlukan minimal 80%</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("""<script>
        var m=new SpeechSynthesisUtterance("Peringatan! Obat ini tidak jelas atau salah. Tolong periksa kembali.");
        m.lang="id-ID";m.rate=0.9;window.speechSynthesis.speak(m);
        </script>""", unsafe_allow_html=True)

    if is_verified and info:
        st.markdown(f"""
        <div class="info-card">
          <div class="conf-bar-title">📋 Informasi Obat</div>
          <div class="info-row"><span class="info-label">Indikasi</span><span class="info-value">{info['indikasi']}</span></div>
          <div class="info-row"><span class="info-label">Dosis</span><span class="info-value">{info['dosis']}</span></div>
          <div class="info-row" style="margin-bottom:0">
            <span class="info-label">⚠️ Perhatian</span>
            <span class="info-value"><div class="warning-text">{info['peringatan']}</div></span>
          </div>
        </div>""", unsafe_allow_html=True)

    bars_html = '<div class="conf-bar-wrap"><div class="conf-bar-title">📊 Kepercayaan AI per Kelas</div>'
    for i in np.argsort(probs)[::-1]:
        name   = CLASS_NAMES[i]
        pct    = probs[i] * 100
        cls    = "top" if i == int(probs.argmax()) else "other"
        bars_html += f"""
        <div class="conf-item">
          <div class="conf-item-label">
            <span class="conf-item-name">{DRUG_INFO.get(name,{}).get('emoji','💊')} {name}</span>
            <span class="conf-item-pct">{pct:.1f}%</span>
          </div>
          <div class="conf-bar-bg"><div class="conf-bar-fill {cls}" style="width:{pct:.1f}%"></div></div>
        </div>"""
    bars_html += "</div>"
    st.markdown(bars_html, unsafe_allow_html=True)

    with st.expander("🔬 Lihat Hasil Preprocessing"):
        st.image(processed_img, caption="Setelah rembg → resize_with_pad → CLAHE", use_column_width=True)

st.markdown("""
<div class="footer">
  <hr/>
  <p>PillScan AI · MobileNetV3 Small & VGG11 · Akurasi 100% pada data uji</p>
  <p style="margin-top:4px">Program Studi Teknik Informatika — Universitas Lampung 2026</p>
  <p style="margin-top:8px;color:#cbd5e1">⚠️ Aplikasi ini hanya alat bantu. Selalu konsultasikan dengan tenaga medis.</p>
</div>
""", unsafe_allow_html=True)
