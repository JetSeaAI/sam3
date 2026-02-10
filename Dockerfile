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

# 只安裝運行時必備的系統庫
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# 從 Builder 階段把安裝好的 Python 套件全部搬過來
# 這會直接捨棄掉所有過程中的快取與暫存檔
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app
COPY . .

# 重新以不可編輯模式安裝，確保進入路徑正確
RUN pip install --no-cache-dir .

EXPOSE 8888
CMD ["/bin/bash"]