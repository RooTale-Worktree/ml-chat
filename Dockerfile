# ===== 베이스 이미지 (PyTorch + CUDA 12.4 + cuDNN9) =====
FROM --platform=linux/amd64 pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

# ===== 필수 유틸 =====
RUN apt-get update && apt-get install -y \
    git wget curl vim build-essential \
    && rm -rf /var/lib/apt/lists/*

# ===== 작업 디렉토리 =====
WORKDIR /workspace

# ===== Python 의존성 설치 =====
COPY requirements.txt /workspace/
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# ===== 소스코드 복사 =====
COPY . /workspace

# ===== 환경 변수 =====
ENV PYTHONPATH=/workspace \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/workspace/.cache/sentence_transformers