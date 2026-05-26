import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# 1. Set Halaman Utama
st.set_page_config(page_title="Klasifikasi Obat PyTorch", layout="centered")
st.title("💊 Identifikasi Obat (MobileNetV3 PyTorch)")
st.write("Klik tombol di bawah untuk mengambil foto obat Alpara / Amoxcilin.")

# 2. Definisikan Ulang Arsitektur Model (Harus Sama dengan Saat Training)
@st.cache_resource
def load_pytorch_model():
    # Inisialisasi arsitektur MobileNetV3 Small
    # weights=None karena kita akan me-load bobot dari file .pth sendiri
    model = models.mobilenet_v3_small(weights=None)
    
    # Sesuaikan bagian classifier/layer terakhir dengan jumlah kelasmu (2 kelas: Alpara & Amoxcilin)
    # Catatan: Sesuaikan bagian ini jika saat training kamu memodifikasi struktur linear layer-nya
    num_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_features, 2) 
    
    # Load state_dict (bobot) yang sudah kamu simpan di Drive
    # map_location='cpu' memastikan model bisa berjalan meski device tidak punya GPU/CUDA
    state_dict = torch.load('mobilenet_alpara.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    
    # Ubah model ke mode evaluasi
    model.eval()
    
    # Daftar Label
    labels = ["Alpara", "Amoxcilin"]
    
    return model, labels

try:
    model, labels = load_pytorch_model()
except Exception as e:
    st.error(f"Gagal memuat model PyTorch: {e}")

# 3. Transformasi Gambar (Pre-processing)
# Sederhanakan agar sama dengan transformasi data uji/validasi saat training
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Jika saat training kamu pakai normalisasi ImageNet, aktifkan baris di bawah:
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 4. Fitur Kamera Streamlit
img_file_buffer = st.camera_input("Arahkan obat ke kamera dan ambil foto")

if img_file_buffer is not None:
    # Mengubah gambar menjadi format PIL Image
    image = Image.open(img_file_buffer).convert('RGB')
    
    with st.spinner('Sedang mengidentifikasi obat...'):
        # Pre-processing gambar menggunakan transformasi PyTorch
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0) # Tambah dimensi batch (1, 3, 224, 224)

        # 5. Jalankan Inference (Prediksi)
        with torch.no_grad():
            output = model(input_batch)
            
            # Mengitung Probabilitas Menggunakan Softmax
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # Mendapatkan indeks kelas dengan probabilitas tertinggi
            confidence, prediction_idx = torch.max(probabilities, 0)
            
            confidence_percentage = confidence.item() * 100
            predicted_class = labels[prediction_idx.item()]

        # 6. Tampilkan Hasil Ke User
        st.success("### Hasil Identifikasi:")
        st.metric(label="Nama Obat", value=f"{predicted_class}")
        st.info(f"Tingkat Keyakinan (Confidence): {confidence_percentage:.2f}%")
