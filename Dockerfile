# --- 階段 1：構建環境 (Builder) ---
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 升級 pip 並安裝 PyTorch (務必加上 --no-cache-dir)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# 安裝 SAM3 及其依賴
COPY . .
RUN pip install --no-cache-dir ".[train,notebooks,dev]"

# --- 階段 2：最終運行環境 (Final) ---
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/local/bin:$PATH"
# 1. 除了 python3-dev，補上 libpython3.10-dev 與 pkg-config，確保編譯器能找到路徑
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev libpython3.10-dev \
    git libgl1 libglib2.0-0 build-essential pkg-config ffmpeg\
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python
# 2. 修正 libcuda 連結問題 (Triton 必備)
RUN ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so || true
# ... 搬運套件指令保持不變 ...
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/local/share/jupyter /usr/local/share/jupyter
# 修正 Jupyter 路徑問題
RUN mkdir -p /usr/share/jupyter && \
    ln -s /usr/local/share/jupyter/lab /usr/share/jupyter/lab
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .
EXPOSE 8888
CMD ["/bin/bash"]