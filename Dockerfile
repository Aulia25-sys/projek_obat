FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install streamlit torch torchvision \
    numpy==2.0.2 Pillow==10.4.0 \
    opencv-python-headless scipy \
    onnxruntime==1.26.0 PyMatting \
    pooch rembg==2.0.57

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
