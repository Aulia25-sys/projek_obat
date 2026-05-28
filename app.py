import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from rembg import remove as rembg_remove
import io

# ─────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────
CLASS_NAMES = [
    'alpara',
    'amoxcillin',
    'lecozinc',
    'simvastatin(QL)',
    'simvastatin(selvim)'
]
IMG_SIZE       = 224
CONF_THRESHOLD = 0.80
DROPOUT_RATE   = 0.5
HIDDEN_NEURONS = 256
NUM_CLASSES    = len(CLASS_NAMES)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

DRUG_INFO = {
    'alpara': {
        'nama'      : 'Alpara (Paracetamol + CTM + Phenylpropanolamine)',
        'bentuk'    : 'Kaplet/tablet, bentuk LONJONG (oval), warna kuning keemasan',
        'imprint'   : 'Tercetak teks ALPARA atau kode produsen pada permukaan',
        'lekukan'   : 'Terdapat garis tengah (scoring) untuk pembagian dosis',
        'indikasi'  : 'Flu, demam, hidung tersumbat, nyeri kepala',
        'dosis'     : '1 tablet 3-4x/hari (dewasa), sesudah makan',
        'peringatan': 'Jangan diberikan pada anak <2 tahun. Hindari alkohol.',
    },
    'amoxcillin': {
        'nama'      : 'Amoxicillin 500mg (Antibiotik Penisilin)',
        'bentuk'    : 'KAPSUL, bentuk PANJANG silindris, warna merah-kuning atau putih',
        'imprint'   : 'Tercetak AMOXICILLIN 500 atau kode produsen pada badan kapsul',
        'lekukan'   : 'Tidak ada lekukan (kapsul dua bagian)',
        'indikasi'  : 'Infeksi bakteri saluran napas, telinga, kulit, saluran kemih',
        'dosis'     : '500mg setiap 8 jam selama 5-7 hari (sesuai resep dokter)',
        'peringatan': 'HANYA dengan resep dokter. Habiskan seluruh antibiotik!',
    },
    'lecozinc': {
        'nama'      : 'Lecozinc (Suplemen Zinc)',
        'bentuk'    : 'Tablet/kaplet, biasanya BULAT atau oval kecil',
        'imprint'   : 'Tercetak LECOZINC atau kode lot pada permukaan tablet',
        'lekukan'   : 'Bervariasi sesuai merek dan ukuran',
        'indikasi'  : 'Suplementasi zinc, mendukung sistem imun dan penyembuhan luka',
        'dosis'     : '1 tablet/hari sesudah makan',
        'peringatan': 'Konsultasi dokter jika dikonsumsi bersamaan dengan antibiotik.',
    },
    'simvastatin(QL)': {
        'nama'      : 'Simvastatin 10/20mg merek QL',
        'bentuk'    : 'Tablet BULAT kecil, warna putih atau krem',
        'imprint'   : 'Tercetak kode QL + dosis pada permukaan',
        'lekukan'   : 'Terdapat garis tengah (scoring) untuk pembagian dosis',
        'indikasi'  : 'Menurunkan kolesterol LDL dan trigliserida',
        'dosis'     : '10-40mg 1x/hari malam hari (sesuai resep dokter)',
        'peringatan': 'RESEP DOKTER. Laporkan nyeri otot tidak biasa segera!',
    },
    'simvastatin(selvim)': {
        'nama'      : 'Simvastatin 10/20mg merek Selvim',
        'bentuk'    : 'Tablet BULAT kecil, warna putih atau krem',
        'imprint'   : 'Tercetak kode SELVIM + dosis pada permukaan tablet',
        'lekukan'   : 'Terdapat garis tengah (scoring) untuk pembagian dosis',
        'indikasi'  : 'Menurunkan kolesterol LDL dan trigliserida',
        'dosis'     : '10-40mg 1x/hari malam hari (sesuai resep dokter)',
        'peringatan': 'RESEP DOKTER. Laporkan nyeri otot tidak biasa segera!',
    },
}

# ─────────────────────────────────────────────────────────
# PREPROCESSING (sama persis dengan training)
# ─────────────────────────────────────────────────────────
def remove_bg_and_replace(img: Image.Image) -> Image.Image:
    output = rembg_remove(img)
    canvas = Image.new('RGB', output.size, (128, 128, 128))
    canvas.paste(output, mask=output.split()[3])
    return canvas

def resize_with_pad(img: Image.Image, target=IMG_SIZE) -> Image.Image:
    img_r = img.copy()
    img_r.thumbnail((target, target), Image.LANCZOS)
    canvas = Image.new('RGB', (target, target), (128, 128, 128))
    x_off  = (target - img_r.width)  // 2
    y_off  = (target - img_r.height) // 2
    canvas.paste(img_r, (x_off, y_off))
    return canvas

def clahe_enhance(img: Image.Image) -> Image.Image:
    img_np  = np.array(img)
    lab     = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    l_eq    = clahe.apply(l)
    lab_eq  = cv2.merge([l_eq, a, b])
    result  = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    return Image.fromarray(result)

def preprocess(img: Image.Image) -> torch.Tensor:
    img = remove_bg_and_replace(img)
    img = resize_with_pad(img)
    img = clahe_enhance(img)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(img).unsqueeze(0)

# ─────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────
def build_mobilenetv3_small():
    model = models.mobilenet_v3_small(weights=None)
    in_f  = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_f, HIDDEN_NEURONS),
        nn.Hardswish(),
        nn.Dropout(p=DROPOUT_RATE),
        nn.Linear(HIDDEN_NEURONS, NUM_CLASSES)
    )
    return model

def build_vgg11():
    model = models.vgg11(weights=None)
    in_f  = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_f, HIDDEN_NEURONS),
        nn.ReLU(inplace=True),
        nn.Dropout(p=DROPOUT_RATE),
        nn.Linear(HIDDEN_NEURONS, NUM_CLASSES)
    )
    return model

@st.cache_resource
def load_model(model_path: str, model_type: str):
    if model_type == 'MobileNetV3':
        model = build_mobilenetv3_small()
    else:
        model = build_vgg11()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# ─────────────────────────────────────────────────────────
# PREDIKSI
# ─────────────────────────────────────────────────────────
def predict(model, img_tensor: torch.Tensor):
    with torch.no_grad():
        logits = model(img_tensor)
        probs  = torch.softmax(logits, dim=1)[0]
    conf, idx = probs.max(0)
    return CLASS_NAMES[idx.item()], conf.item(), probs.numpy()

# ─────────────────────────────────────────────────────────
# UI STREAMLIT
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Klasifikasi Obat',
    page_icon='💊',
    layout='wide'
)

st.title('💊 Klasifikasi Obat 5 Kelas')
st.caption('MobileNetV3-Small & VGG11 | Preprocessing: rembg + CLAHE')

# Sidebar — upload model
with st.sidebar:
    st.header('⚙️ Pengaturan Model')
    model_type = st.radio('Pilih Model', ['MobileNetV3', 'VGG11'])
    model_file = st.file_uploader('Upload file .pth', type=['pth'])
    conf_thresh = st.slider('Confidence Threshold', 0.0, 1.0, CONF_THRESHOLD, 0.05)
    st.divider()
    st.markdown('**5 Kelas Obat:**')
    for c in CLASS_NAMES:
        st.markdown(f'- {c}')

# Load model
model = None
if model_file:
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
        f.write(model_file.read())
        tmp_path = f.name
    try:
        model = load_model(tmp_path, model_type)
        st.sidebar.success(f'✅ Model {model_type} berhasil dimuat!')
    except Exception as e:
        st.sidebar.error(f'❌ Gagal load model: {e}')
    finally:
        os.unlink(tmp_path)
else:
    st.info('👈 Upload file model `.pth` di sidebar untuk memulai.')

# Upload gambar
uploaded = st.file_uploader('📸 Upload gambar obat', type=['jpg', 'jpeg', 'png', 'bmp', 'webp'])

if uploaded and model:
    img = Image.open(uploaded).convert('RGB')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader('📷 Gambar Asli')
        st.image(img, use_container_width=True)

    with st.spinner('Memproses gambar (rembg + CLAHE)...'):
        img_nobg    = remove_bg_and_replace(img)
        img_padded  = resize_with_pad(img_nobg)
        img_clahe   = clahe_enhance(img_padded)
        img_tensor  = preprocess(img)

    with col2:
        st.subheader('🔲 Setelah Hapus BG')
        st.image(img_nobg, use_container_width=True)

    with col3:
        st.subheader('✨ Setelah CLAHE')
        st.image(img_clahe, use_container_width=True)

    st.divider()

    # Prediksi
    pred_class, confidence, all_probs = predict(model, img_tensor)

    col_res, col_info = st.columns([1, 2])

    with col_res:
        st.subheader('🎯 Hasil Prediksi')
        if confidence >= conf_thresh:
            st.success(f'**{pred_class}**')
            st.metric('Confidence', f'{confidence*100:.1f}%')
        else:
            st.warning(f'Prediksi: **{pred_class}** (confidence rendah)')
            st.metric('Confidence', f'{confidence*100:.1f}%', delta=f'< threshold {conf_thresh*100:.0f}%')

        st.subheader('📊 Probabilitas Semua Kelas')
        for i, (cls, prob) in enumerate(zip(CLASS_NAMES, all_probs)):
            bar_color = '🟩' if cls == pred_class else '⬜'
            st.write(f'{bar_color} `{cls[:20]:20s}` {prob*100:5.1f}%')
            st.progress(float(prob))

    with col_info:
        if confidence >= conf_thresh and pred_class in DRUG_INFO:
            info = DRUG_INFO[pred_class]
            st.subheader(f'ℹ️ Informasi Obat')
            st.markdown(f"**Nama:** {info['nama']}")
            st.markdown(f"**Bentuk:** {info['bentuk']}")
            st.markdown(f"**Imprint:** {info['imprint']}")
            st.markdown(f"**Lekukan:** {info['lekukan']}")
            st.markdown(f"**Indikasi:** {info['indikasi']}")
            st.markdown(f"**Dosis:** {info['dosis']}")
            st.error(f"⚠️ **Peringatan:** {info['peringatan']}")
        else:
            st.info('Informasi obat ditampilkan jika confidence mencukupi threshold.')

elif uploaded and not model:
    st.warning('⚠️ Gambar sudah diupload, tapi model belum dimuat. Upload file `.pth` di sidebar.')
