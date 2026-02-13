# SAM3 Image Streaming 效能優化報告

## 1. 測試結果摘要
本報告基於在 **NVIDIA RTX 5090** 上運行的 **SAM3-Huge (Resolution 1008x1008)** 模型 Streaming 測試結果。

- **總幀數**: 1500 Frames
- **總耗時**: 93.41 秒
- **平均 FPS**: **16.06 FPS** (優化前約 8 FPS，提升幅度 **~100%**)
- **VRAM 佔用**: Peak ~13.6 GB / Current ~13.2 GB

### 詳細 Profiling (每幀平均耗時)
| 階段 | 耗時 (ms) | 佔比 | 說明 |
| :--- | :--- | :--- | :--- |
| **I/O & Convert** | **0.10 ms** | < 1% | **極致優化**。透過多執行緒讀取 + GPU Preprocessing，幾乎完全消除了 I/O 延遲。 |
| **Backbone (+Pre)** | **41.79 ms** | 69% | **主要瓶頸**。這是 ViT-Huge 在 1008x1008 解析度下的硬體計算極限。 |
| **Decoder (Fast)** | **18.63 ms** | 30% | **顯著改善**。透過跳過高解析度插值，耗時從原始的 ~46ms 大幅下降。 |

---

## 2. 實作的優化策略
我們透過以下三個關鍵技術手段，成功將 FPS 翻倍：

### A. Decoder 加速 (FastSam3Processor)
- **原狀**: SAM3 預設會將 Mask (256x256) 強制插值放大回原圖尺寸 (1920x1080)，在 GPU 上耗時約 46ms。
- **優化**: 實作 `FastSam3Processor`，**跳過插值步驟**，直接回傳模型原始輸出的低解析度 Mask。
- **成效**: Decoder 耗時從 46ms 降至 **18ms**。

### B. I/O 隱藏 (Threaded Video Reader)
- **原狀**: 每次迴圈的主執行緒需等待硬碟讀取 (`cv2.read`)，導致 GPU 空轉 (約 7ms)。
- **優化**: 實作 `ThreadedVideoReader`，在背景執行緒持續預讀影像至 Queue。
- **成效**: I/O 等待時間從 7ms 降至 **0.1ms** (幾乎為零)。

### C. GPU 前處理 (Preprocessing Fusion)
- **原狀**: 使用 CPU (OpenCV/PIL) 進行 Resize、RGB 轉換與 Normalize，佔用 CPU 資源且慢。
- **優化**: 直接將原始 BGR 數據送入 GPU，利用 CUDA 核心進行高效的 Resize 與 Normalize。
- **成效**: 前處理時間從 5ms 被融合進 Backbone 計算中，幾乎無感。

---

## 3. 未來效能提升方向 (FPS)
目前的瓶頸為 **Backbone (41.79 ms)**，佔總時間 70%。若需進一步提升 FPS，建議方向如下：

1.  **模型輕量化 (最有效)**
    *   改用 **SAM3-Large** 或 **SAM3-Base**。參數量減少可直接提升推論速度至 30-40 FPS。
2.  **量化 (Quantization)**
    *   將模型從 `BFloat16` 轉為 **INT8** (需使用 TensorRT)。預期 Backbone 速度可翻倍。
3.  **Batching (犧牲延遲換吞吐量)**
    *   若允許非即時處理，可累積 4 張圖 (`batch_size=4`) 一起推論。雖延遲增加，但整體 FPS 可望提升至 30+。

---

## 4. VRAM 使用量優化指南
目前 RTX 5090 (32GB) 僅使用了約 **13.6 GB (42%)**。若您希望降低 VRAM 使用量 (例如為了在較小的顯卡如 4090 或 3090 上運行，或為了跑多個實例)，可以採取以下措施：

### A. 關鍵優化：降低輸入解析度 (Resolution Reduction)
這是最直接有效的方法，VRAM 使用量與解析度的平方成正比。
*   **方法**: 在 `run_stream_processing.py` 中將 `resolution` 從 `1008` 改為 `640` 或 `512`。
    ```python
    # 範例修改
    resolution = 640  # 原為 1008
    processor = FastSam3Processor(model, resolution=resolution)
    ```
    *注意：這需要模型 Checkpoint 支援動態 Position Embedding 插值，否則可能會報錯或準確度下降。*

### B. 清理 Cache (Empty Cache)
PyTorch 會快取記憶體以加速分配。若看到 VRAM 佔用高但實際使用低，可手動清理。
*   **方法**:
    ```python
    torch.cuda.empty_cache()
    ```
    *注意：這會稍微影響速度，建議僅在 frame 之間偶爾執行。*

### C. 混合精度優化 (Mixed Precision)
目前已使用 `BFloat16`。確認沒有意外轉回 `Float32`。
*   **檢查**: 確保所有 inference 代碼都在 `with torch.autocast(...)` 區塊內。

### D. 模型卸載 (Offloading) - 極端手段
若 VRAM 極其吃緊，可以只將當前運算的層 (Layer) 搬到 GPU，算完搬回 CPU。
*   **工具**: 使用 `accelerate` 庫的 `cpu_offload` 功能。這會顯著降低 FPS，但能在大模型上運作於小顯存卡。

---
**總結**: 對於即時串流應用，目前的配置 (16 FPS, 13GB VRAM) 是一個在 RTX 5090 上非常平衡且高效的結果。