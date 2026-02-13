import os
import cv2
import matplotlib.pyplot as plt
import torch
import glob
import time
from PIL import Image
from tqdm import tqdm
import threading
import queue
import numpy as np
# Imports from sam3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results

class ThreadedVideoReader:
    def __init__(self, path, queue_size=4):
        self.cap = cv2.VideoCapture(path)
        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.t = threading.Thread(target=self.update, daemon=True)
        self.t.start()
    
    def update(self):
        while not self.stopped:
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    break
                self.q.put((ret, frame))
            else:
                time.sleep(0.001)
                
    def read(self):
        if self.q.empty() and self.stopped:
            return False, None
        try:
            # Short timeout to allow check for stopped
            return self.q.get(timeout=1.0)
        except queue.Empty:
            return False, None
            
    def release(self):
        self.stopped = True
        # Read remaining items to allow thread to exit if blocked on put
        try:
            while not self.q.empty():
                self.q.get_nowait()
        except:
            pass
        self.t.join()
        self.cap.release()

class FastSam3Processor(Sam3Processor):
    def __init__(self, model, resolution=1024):
        super().__init__(model, resolution=resolution)
        # Standard ImageNet mean/std
        self.pixel_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(model.device)
        self.pixel_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(model.device)

    @torch.inference_mode()
    def preprocess_on_gpu(self, frame_bgr_np):
        # 1. To Tensor (H, W, C) -> (C, H, W)
        # Doing this on CPU is still often faster than moving huge int8 array then casting
        # taking ~1-2ms for 1080p
        frame_tensor = torch.from_numpy(frame_bgr_np).permute(2, 0, 1).contiguous()
        
        # 2. Move to GPU (Async)
        frame_tensor = frame_tensor.to(self.device, non_blocking=True)
        
        # 3. Cast to float & Normalize (fuse)
        # BGR -> RGB (flip channel 0 and 2)
        frame_tensor = frame_tensor[[2, 1, 0], :, :]
        frame_tensor = frame_tensor.float().div(255.0).unsqueeze(0) # (1, 3, H, W)
        
        # 4. Resize if needed
        if frame_tensor.shape[-1] != self.resolution or frame_tensor.shape[-2] != self.resolution:
            frame_tensor = torch.nn.functional.interpolate(
                frame_tensor, 
                size=(self.resolution, self.resolution), 
                mode='bilinear', 
                align_corners=False
            )
            
        # 5. Normalize
        frame_tensor = (frame_tensor - self.pixel_mean) / self.pixel_std
        return frame_tensor

    @torch.inference_mode()
    def _forward_grounding(self, state: dict):
        # Override to skip expensive interpolation to original resolution
        outputs = self.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=self.find_stage,
            geometric_prompt=state["geometric_prompt"],
            find_target=None,
        )

        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"] # Low res masks (likely 256x256)
        out_probs = out_logits.sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)

        keep = out_probs > self.confidence_threshold
        out_probs = out_probs[keep]
        out_masks = out_masks[keep]
        out_bbox = out_bbox[keep]

        # Convert boxes
        from sam3.model import box_ops
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        img_h = state["original_height"]
        img_w = state["original_width"]
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(self.device)
        boxes = boxes * scale_fct[None, :]

        # OPTIMIZATION: Skip F.interpolate to original size!
        # Just keep low-res masks. Logits > 0 is equivalent to Sigmoid > 0.5
        state["masks_logits"] = out_masks
        state["masks"] = out_masks > 0.0 
        state["boxes"] = boxes
        state["scores"] = out_probs
        return state

def main():
    # Setup
    video_path = "./assets/videos/S1_ch1234_20251029_1413_1min.mp4"
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        mp4s = glob.glob("./assets/videos/*.mp4")
        if mp4s:
            video_path = mp4s[0]
            print(f"Using alternative video: {video_path}")
        else:
            print("No video found.")
            return

    output_dir = "./assets/output_frames"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Build the image model (for per-frame processing)
    # OPTIMIZATION: Enable compilation!
    print("Building SAM3 Image Model (with torch.compile enabled)...")
    try:
        model = build_sam3_image_model(compile=True)
    except Exception as e:
        print(f"Compilation failed or not supported: {e}. Falling back to standard mode.")
        model = build_sam3_image_model(compile=False)
    
    # Reverting to default resolution 1008 as the model checkpoint requires fixed positional embeddings
    resolution = 1008 
    print(f"Initializing FastProcessor with resolution={resolution} (Skipping High-Res Interpolation)...")
    
    # OPTIMIZATION: Use the Fast Processor
    processor = FastSam3Processor(model, resolution=resolution)
    
    # OPTIMIZATION: Disable compilation as it proved slower (12 FPS vs 16 FPS)
    # due to complex operators fallback.
    print("Building SAM3 Image Model (Standard Eager Mode)...")
    model = build_sam3_image_model(compile=False)
    
    resolution = 1008 
    print(f"Initializing FastProcessor with resolution={resolution} (Skipping High-Res Interpolation)...")
    processor = FastSam3Processor(model, resolution=resolution)

    prompt = "boat"
    print(f"Pre-computing text embedding for prompt: '{prompt}'...")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        cached_text_features = model.backbone.forward_text([prompt], device=processor.device)
    
    # WARM UP (Just to allocate GPU memory buffers)
    print("Warming up model with ACTUAL video frame...")
    
    # Use distinct capture for warmup to avoid threading race conditions
    temp_cap = cv2.VideoCapture(video_path)
    total_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, warmup_frame = temp_cap.read()
    temp_cap.release()
    time.sleep(0.5) # Wait for FFmpeg to cleanup
    
    if not ret:
        print("Could not read video for warmup.")
        return
    
    # Run a full pass for warmup
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = {}
        processed_tensor = processor.preprocess_on_gpu(warmup_frame)
        state["original_height"], state["original_width"] = warmup_frame.shape[:2]
        state["backbone_out"] = processor.model.backbone.forward_image(processed_tensor)
        state["backbone_out"].update(cached_text_features)
        state["geometric_prompt"] = model._get_dummy_prompt()
        state = processor._forward_grounding(state)
        
    print("Warmup complete. Starting Measurement...")
    time.sleep(0.5) # Extra buffer before starting threads

    # OPTIMIZATION: Use Threaded Video Reader
    print(f"Starting optimized streaming processing with threaded I/O...")
    cap = ThreadedVideoReader(video_path)

    start_time = time.time()
    frame_count = 0
    
    # Profiling variables
    t_io = 0
    t_pre = 0
    t_inf = 0
    
    # Use a loop to process frames one by one
    pbar = tqdm(total=total_frames)
    
    try:
        import subprocess
        while True:
            # 1. IO Time
            # Note: With threaded reader, read() pulls from queue, essentially hiding disk I/O latency
            torch.cuda.synchronize()
            t0 = time.time()
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break
            
            # OPTIMIZATION: REMOVED CPU Convert (cv2.cvtColor) and PIL creation
            # Direct BGR numpy -> GPU Tensor
            torch.cuda.synchronize()
            t1 = time.time()
            t_io += (t1 - t0)
            
            # 3. Process the image using AMP with GPU Preprocessing
            state = {}
            state["original_height"], state["original_width"] = frame_bgr.shape[:2]
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                
                # 2. Backbone (+Pre) Time
                torch.cuda.synchronize()
                t2 = time.time()
                
                # A. GPU Preprocessing (Resize, Normalize, ToTensor)
                processed_tensor = processor.preprocess_on_gpu(frame_bgr)
                
                # B. Backbone Forward
                # Note: processor.set_image would do transform (CPU) then forward_image.
                # We replaced transform with preprocess_on_gpu.
                state["backbone_out"] = processor.model.backbone.forward_image(processed_tensor)
                
                torch.cuda.synchronize()
                t3 = time.time()
                
                state["backbone_out"].update(cached_text_features)
                if "geometric_prompt" not in state:
                    state["geometric_prompt"] = model._get_dummy_prompt()
                
                # 3. Decoder Inference Time
                # Run grounding (Fast Decoder)
                state = processor._forward_grounding(state)
                torch.cuda.synchronize()
                t4 = time.time()
                
                t_pre += (t3 - t2)
                t_inf += (t4 - t3)
            
            frame_count += 1
            pbar.update(1)
            
            # Monitor GPU every 30 frames
            if frame_count % 30 == 0:
                # Get GPU stats
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], 
                                         capture_output=True, text=True)
                    gpu_info = result.stdout.strip().split(',')
                    util = gpu_info[0].strip()
                    mem = gpu_info[1].strip()
                    pbar.set_description(f"GPU: {util}% / {mem}MiB")
                except:
                    pass
            
    finally:
        cap.release()
        pbar.close()
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    print(f"\n--- Fast Image Streaming Statistics (Threaded I/O + Compile + GPU Preprocess) ---")
    print(f"Total Frames: {frame_count}")
    print(f"Total Time: {elapsed_time:.2f} s")
    print(f"Average FPS: {fps:.2f}")
    
    if frame_count > 0:
        avg_io = (t_io / frame_count) * 1000
        avg_backbone = (t_pre / frame_count) * 1000
        avg_decoder = (t_inf / frame_count) * 1000
        print(f"\n--- Detailed Profiling (Avg per frame) ---")
        print(f"I/O & Convert : {avg_io:.2f} ms")
        print(f"Backbone (+Pre): {avg_backbone:.2f} ms")
        print(f"Decoder (Fast) : {avg_decoder:.2f} ms")
        print(f"----------------------------------------")
    
    # Final Memory Stats via Torch
    max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    cur_mem = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"Peak GPU Memory: {max_mem:.2f} MiB")
    print(f"Current GPU Memory: {cur_mem:.2f} MiB")

if __name__ == "__main__":
    main()
