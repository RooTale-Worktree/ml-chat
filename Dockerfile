# Base image with vLLM pre-installed (includes PyTorch and CUDA)
# provides the basic runtime environment (~12GB)
FROM vllm/vllm-openai:latest

# Install essential utilities (git, curl, etc ...)
USER root
RUN apt-get update && apt-get install -y \
    git wget curl vim build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory for subsequent commands
WORKDIR /workspace

# Set environment variables for Python and model caching
ENV PYTHONPATH=/workspace \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/workspace/.cache/sentence_transformers \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    CUDA_VISIBLE_DEVICES=0

# Download python dependencies (~650MB)
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Download LLM model to specified cache directory
# model_name: openai/gpt-oss-20b (~51GB)
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('openai/gpt-oss-20b', cache_dir='/workspace/.cache/huggingface'); \
    print('Model downloaded to cache')"

# Download embedding model to specified cache directory
# model_name: jhgan/ko-sbert-nli (~500MB)
RUN python3 -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('jhgan/ko-sbert-nli', cache_folder='/workspace/.cache/sentence_transformers'); \
    print('Embedding model downloaded to cache')"

# Copy source code into the container
# It is placed at the end to leverage Docker layer caching
COPY . /workspace

# If required: build the index from Neo4j
# RUN python scripts/build_index.py || echo "Index build skipped (no DB access)"

# Commands implemented when the container starts
CMD ["python", "-u", "handler.py"]