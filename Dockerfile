# ===== 베이스 이미지 =====
FROM vllm/vllm-openai:latest

# PyTorch 베이스로 시작하는 경우
# FROM --platform=linux/amd64 pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

# ===== 필수 유틸 =====
USER root
RUN apt-get update && apt-get install -y \
    git wget curl vim build-essential \
    && rm -rf /var/lib/apt/lists/*

# ===== 작업 디렉토리 =====
WORKDIR /workspace

# ===== Python 의존성 설치 =====
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ===== 모델 사전 다운로드 =====
# Cold start 시간을 줄이기 위해
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('openai/gpt-oss-20b', cache_dir='/workspace/.cache/huggingface'); \
    print('Model downloaded to cache')"

# ===== 소스코드 복사 =====
COPY . /workspace

# ===== 환경 변수 =====
ENV PYTHONPATH=/workspace \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/workspace/.cache/sentence_transformers \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    CUDA_VISIBLE_DEVICES=0

# ===== 사전 처리: Neo4j에서 인덱스 빌드 =====
# Neo4j 접근이 빌드 타임에 가능하다면 활성화
# RUN python scripts/build_index.py || echo "Index build skipped (no DB access)"

# ===== RunPod Handler 설정 =====
# RunPod serverless는 handler.py의 handler 함수를 자동 호출
CMD ["python", "-u", "handler.py"]