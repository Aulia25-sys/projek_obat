import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# 1. Set Halaman Utama
st.set_page_config(page_title="Klasifikasi Obat PyTorch", layout="centered")
st.title("💊 Identifikasi Obat (MobileNetV3 PyTorch)")
st.write("Arahkan obat ke kamera untuk mengidentifikasi jenis obat.")

# 2. Definisikan Struktur Model Sesuai Checkpoint Kamu
@st.cache_resource
def load_pytorch_model():
    # Load arsitektur dasar MobileNetV3 Small
    model = models.mobilenet_v3_small(weights=None)
    
    # KUSTOMISASI CLASSIFIER: Disamakan dengan isi file .pth kamu 
    # untuk memperbaiki error "size mismatch" ([576 -> 256 -> 5 kelas])
    model.classifier = nn.Sequential(
        nn.Linear(576, 256),       # Lapisan pertama: input 576, output 256
        nn.Hardswish(),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(256, 5)          # Lapisan terakhir: output 5 kelas (sesuai jumlah kelasmu)
    )
    
    # Load state_dict (bobot) dari file modelmu
    # map_location='cpu' memastikan model bisa jalan di server Streamlit Cloud tanpa GPU
    state_dict = torch.load('mobilenetv3_small_5kelas.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    
    model.eval() # Mengubah model ke mode evaluasi
    return model

# Memuat model ke dalam aplikasi
try:
    model = load_pytorch_model()
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file 'mobilenetv3_small_5kelas.pth' ada di folder yang sama. Error: {e}")

# 3. Daftar Label (Sesuaikan urutannya dengan indeks kelas saat training dulu!)
# PENTING: Ganti 'Obat C', 'Obat D', 'Obat E' dengan nama obat asli kamu
labels = ['Alpara', 'Amoxcilin', 'Obat C', 'Obat D', 'Obat E']

# 4. Pipeline Transformasi Gambar
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Jika saat training kamu menggunakan normalisasi standar ImageNet, aktifkan baris di bawah:
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 5. Fitur Kamera Streamlit
img_file_buffer = st.camera_input("Arahkan obat ke kamera dan ambil foto")

if img_file_buffer is not None:
    # Mengubah gambar menjadi format PIL Image
    image = Image.open(img_file_buffer).convert('RGB')
    
    # Menampilkan preview gambar yang diambil oleh user
    st.image(image, caption="Foto yang diambil", use_column_width=True)
    
    with st.spinner('Sedang mengidentifikasi obat...'):
        # Pre-processing gambar menggunakan transformasi PyTorch
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0) # Menambah dimensi batch: menjadi (1, 3, 224, 224)

        # 6. Jalankan Inference (Prediksi)
        with torch.no_grad():
            output = model(input_batch)
            
            # Menghitung Probabilitas Menggunakan Softmax
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # Mendapatkan indeks kelas dengan probabilitas tertinggi
            confidence, prediction_idx = torch.max(probabilities, 0)
            
            confidence_percentage = confidence.item() * 100
            predicted_class = labels[prediction_idx.item()]

    # 7. Tampilkan Hasil Prediksi ke Layar
    st.success("### Hasil Identifikasi:")
    st.write(f"**Nama Obat:** {predicted_class}")
    st.write(f"**Tingkat Keyakinan (Confidence):** {confidence_percentage:.2f}%")
    
    # Menampilkan progress bar visual untuk tingkat keyakinan
    st.progress(int(confidence_percentage))
