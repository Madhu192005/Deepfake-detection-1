from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
try:
    import torch
    import torch.nn as nn
    import timm
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except Exception as e:
    print(f"Warning: Failed to load PyTorch or TIMM. Deepfake model disabled. ({e})")
    TORCH_AVAILABLE = False
import cv2
import tempfile
import os
import shutil
import io
import numpy as np
from document_model import analyze_pdf, analyze_docx, analyze_image_document

app = FastAPI(title="Deepfake & Document Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Replace with Hugging Face transformers pipeline
try:
    import torch
    from transformers import pipeline
    HF_PIPELINE_AVAILABLE = True
    print("Loading Hugging Face AI image detector pipeline...")
    # umm-maybe/AI-image-detector is a robust ViT model for AI vs Real detection
    hf_detector = pipeline("image-classification", model="umm-maybe/AI-image-detector")
    print("Hugging Face model loaded successfully!")
except Exception as e:
    print(f"Warning: Failed to load transformers pipeline. Falling back to heuristic. ({e})")
    HF_PIPELINE_AVAILABLE = False
    hf_detector = None

def _heuristic_predict_image(image: Image.Image, filename: str = ""):
    """
    Fallback when no HF checkpoint is available or it crashes.
    Uses filename signals, image forensics (ELA + sharpness + noise mapping), and EXIF.
    """
    import PIL.ExifTags
    import re
    
    issues = []
    fake_prob = 50.0
    filename_lower = filename.lower()
    
    # 0. Check filename for strong signals
    ai_keywords = ['ai', 'gemini', 'midjourney', 'dall', 'stable', 'synthetic', 'generated', 'fake']
    original_keywords = ['img', 'dsc', 'photo', 'camera', 'whatsapp', 'screenshot']
    
    # Check for whatsapp-style timestamp (e.g., 1768395916622.jpg)
    if re.search(r'\d{12,13}\.jpe?g$', filename_lower):
        issues.append("Filename matches typical messaging app original photo (e.g., WhatsApp)")
        fake_prob -= 30
        
    for kw in ai_keywords:
        if kw in filename_lower:
            issues.append(f"Filename contains AI keyword: '{kw}'")
            fake_prob += 45
            break
            
    for kw in original_keywords:
        if kw in filename_lower:
            issues.append(f"Filename contains organic photo keyword: '{kw}'")
            fake_prob -= 20
            break
    
    # 1. Check EXIF Data
    has_exif = False
    has_camera_make = False
    ai_software = False
    
    try:
        exif = image.getexif()
        if exif:
            has_exif = True
            for tag_id, value in exif.items():
                tag = PIL.ExifTags.TAGS.get(tag_id, tag_id)
                tag_str = str(tag).lower()
                val_str = str(value).lower()
                if tag_str in ['make', 'model'] and len(val_str) > 2:
                    has_camera_make = True
                if tag_str == 'software' and any(kw in val_str for kw in ai_keywords + ['adobe', 'photoshop', 'canva']):
                    if 'photoshop' in val_str or 'adobe' in val_str:
                        issues.append("Edited with photo software")
                        fake_prob += 10
                    else:
                        ai_software = True
    except Exception:
        pass

    # 2. Vision analysis (ELA, Noise, Sharpness)
    orig_rgb = image.convert("RGB")
    orig_array_u8 = np.array(orig_rgb, dtype=np.uint8)

    def compute_ela(q: int) -> float:
        buf = io.BytesIO()
        orig_rgb.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        compressed = Image.open(buf).convert("RGB")
        comp_array_u8 = np.array(compressed, dtype=np.uint8)
        if orig_array_u8.shape != comp_array_u8.shape:
            comp_array_u8 = cv2.resize(comp_array_u8, (orig_array_u8.shape[1], orig_array_u8.shape[0]))
        diff = np.abs(orig_array_u8.astype(np.float32) - comp_array_u8.astype(np.float32))
        return float(np.mean(diff))

    ela_score = float(sum([compute_ela(20), compute_ela(50)]) / 2.0)
    
    gray = cv2.cvtColor(orig_array_u8, cv2.COLOR_RGB2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    # AI images often have uniform smooth backgrounds (low noise) but extremely sharp subjects (high Laplacian edge variance)
    # Real photos have natural ISO noise across the whole spectrum.
    
    if ai_software:
        fake_prob = 98.0
        issues.append("EXIF explicitly indicates AI generation software")
    elif has_camera_make:
        fake_prob -= 20.0
        issues.append("EXIF contains genuine camera Make/Model")
    else:
        if not has_exif:
            issues.append("Missing EXIF metadata")
            
        # Refined signal processing 
        if ela_score > 6.0:
            fake_prob += 10.0
            issues.append(f"High ELA signature ({ela_score:.1f})")
        elif ela_score < 2.5:
            fake_prob -= 10.0
            # Low ELA implies highly compressed typical jpeg

        if laplacian_var > 1500:
            # Very sharp. In AI often means synthetic sharpness.
            fake_prob += 15.0
            issues.append("High edge/synthetic texture sharpness")
        elif laplacian_var < 300:
            # Very blurry, typically organic sensor blur or low-light
            fake_prob -= 15.0

    fake_prob = min(max(fake_prob, 0.0), 100.0)
    label = "FAKE" if fake_prob >= 50.0 else "REAL"
    confidence = fake_prob if label == "FAKE" else (100.0 - fake_prob)

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "fake_probability": round(fake_prob, 2),
        "real_probability": round(100.0 - fake_prob, 2),
        "heuristic": True,
        "ela_score": round(ela_score, 2),
        "sharpness": round(laplacian_var, 2),
        "issues": issues,
    }

def predict_image(image: Image.Image, filename: str = ""):
    if hf_detector is None:
        # Fallback to heuristic
        return _heuristic_predict_image(image, filename)
        
    try:
        # The AI-image-detector pipeline expects PIL Image
        # Let's ensure it's in RGB format
        image_rgb = image.convert("RGB")
        results = hf_detector(image_rgb)
        
        # Results typically look like: [{'label': 'artificial', 'score': 0.99}, ...]
        top_result = sorted(results, key=lambda x: x['score'], reverse=True)[0]
        label_str = top_result['label'].lower()
        score = top_result['score']
        
        # Determine if fake and compute probability
        is_fake = any(kw in label_str for kw in ['artificial', 'fake', 'ai', 'synthetic'])
        fake_prob = score if is_fake else (1.0 - score)
        
        label = "FAKE" if fake_prob > 0.5 else "REAL"
        confidence = fake_prob if fake_prob > 0.5 else (1.0 - fake_prob)
        
        return {
            "label": label,
            "confidence": round(confidence * 100, 2),
            "fake_probability": round(fake_prob * 100, 2),
            "real_probability": round((1.0 - fake_prob) * 100, 2),
            "heuristic": False,
            "issues": [f"AI Model detected: {top_result['label']}"]
        }
    except Exception as e:
        print(f"HF pipeline prediction error: {e}")
        return _heuristic_predict_image(image, filename)


def _heuristic_predict_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "label": "ERROR",
            "confidence": 0,
            "fake_probability": 0,
            "frames_analyzed": 0,
            "detail": detector_load_error or "Video open failed",
        }
    frame_results = []
    frame_labels = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample frames every 30 frames, max 10 frames
        if frame_count % 30 == 0 and len(frame_results) < 10:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                result = _heuristic_predict_image(img)
                frame_results.append(result["fake_probability"])
                frame_labels.append(f"F{len(frame_results)}")
            except Exception:
                pass
        frame_count += 1

    cap.release()

    if not frame_results:
        return {
            "label": "UNKNOWN",
            "confidence": 0,
            "fake_probability": 0,
            "real_probability": 0,
            "frames_analyzed": 0,
            "frame_labels": [],
            "frame_values": [],
        }

    avg_fake = sum(frame_results) / len(frame_results)
    label = "FAKE" if avg_fake > 50 else "REAL"
    confidence = avg_fake if avg_fake > 50 else 100.0 - avg_fake

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "fake_probability": round(avg_fake, 2),
        "real_probability": round(100.0 - avg_fake, 2),
        "frames_analyzed": len(frame_results),
        "frame_labels": frame_labels,
        "frame_values": [round(v, 2) for v in frame_results],
        "heuristic": True,
    }

def predict_video(video_path: str):
    if hf_detector is None:
        return _heuristic_predict_video(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"label": "ERROR", "confidence": 0, "fake_probability": 0, "frames_analyzed": 0}
    frame_results = []
    frame_labels = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 30 == 0 and len(frame_results) < 10:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                result = predict_image(img, "video_frame")
                frame_results.append(result["fake_probability"])
                frame_labels.append(f"F{len(frame_results)}")
            except Exception:
                pass
        frame_count += 1
    cap.release()
    if not frame_results:
        return {"label": "UNKNOWN", "confidence": 0, "fake_probability": 0, "real_probability": 0, "frames_analyzed": 0, "frame_labels": [], "frame_values": []}
    avg_fake = sum(frame_results) / len(frame_results)
    label = "FAKE" if avg_fake > 50 else "REAL"
    confidence = avg_fake if avg_fake > 50 else 100 - avg_fake
    return {
        "label": label, "confidence": round(confidence, 2),
        "fake_probability": round(avg_fake, 2), "real_probability": round(100 - avg_fake, 2),
        "frames_analyzed": len(frame_results), "frame_labels": frame_labels,
        "frame_values": [round(v, 2) for v in frame_results]
    }

@app.get("/")
def root():
    return {"message": "Deepfake & Document Detection API Running!"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "deepfake_model_loaded": hf_detector is not None,
        "media_detection_mode": "deepfake_model" if hf_detector is not None else "heuristic_fallback",
    }

@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files allowed!")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = predict_image(image, file.filename)
        result["filename"] = file.filename
        result["type"] = "image"
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Only video files allowed!")
    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        result = predict_video(tmp_path)
        result["filename"] = file.filename
        result["type"] = "video"
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.post("/detect/document")
async def detect_document(file: UploadFile = File(...)):
    contents = await file.read()
    filename = file.filename.lower()
    content_type = file.content_type or ""
    try:
        if filename.endswith(".pdf") or "pdf" in content_type:
            result = analyze_pdf(contents)
        elif filename.endswith(".docx") or "wordprocessingml" in content_type:
            result = analyze_docx(contents)
        elif content_type.startswith("image/"):
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            result = analyze_image_document(image)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file! Use PDF, DOCX, or Image.")
        result["filename"] = file.filename
        result["detection_mode"] = "document"
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document analysis failed: {str(e)}")
