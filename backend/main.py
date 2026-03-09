from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn as nn
import timm
import torchvision.transforms as transforms
import cv2
import tempfile
import os
import shutil
import io
from document_model import analyze_pdf, analyze_docx, analyze_image_document
app = FastAPI(title="Deepfake & Document Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=2
        )
    def forward(self, x):
        return self.model(x)
print("Loading AI model...")
detector = DeepfakeDetector().to(device)
detector.eval()
print("Model loaded!")
def predict_image(image: Image.Image) -> dict:
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = detector(tensor)
        probs = torch.softmax(output, dim=1)
        fake_prob = probs[0][1].item()

    label = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = fake_prob if fake_prob > 0.5 else 1 - fake_prob

    return {
        "label": label,
        "confidence": round(confidence * 100, 2),
        "fake_probability": round(fake_prob * 100, 2),
        "real_probability": round((1 - fake_prob) * 100, 2),
    }
def predict_video(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "label": "ERROR",
            "confidence": 0,
            "frames_analyzed": 0,
            "frame_labels": [],
            "frame_values": [],
        }
    frame_results = []
    frame_labels = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count >= 300:
            break
        if frame_count % 30 == 0 and len(frame_results) < 10:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                result = predict_image(img)
                frame_results.append(result["fake_probability"])
                frame_labels.append(f"F{len(frame_results)}")
            except Exception:
                pass  # skip unreadable frames silently
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
    confidence = avg_fake if avg_fake > 50 else 100 - avg_fake
    return {
        "label": label,
        "confidence": round(confidence, 2),
        "fake_probability": round(avg_fake, 2),
        "real_probability": round(100 - avg_fake, 2),
        "frames_analyzed": len(frame_results),
        "frame_labels": frame_labels,
        "frame_values": [round(v, 2) for v in frame_results],
    }
@app.get("/")
def root():
    return {"message": "Deepfake & Document Detection API Running"}

@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files allowed")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        result = predict_image(image)
        result["filename"] = file.filename
        result["type"] = "image"
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    content_type = file.content_type or ""
    if not content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Only video files allowed")
    suffix = os.path.splitext(file.filename or "video")[1] or ".mp4"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        result = predict_video(tmp_path)
        result["filename"] = file.filename
        result["type"] = "video"
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
@app.post("/detect/document")
async def detect_document(file: UploadFile = File(...)):
    content_type = file.content_type or ""
    filename = (file.filename or "").lower()
    contents = await file.read()
    try:
        if filename.endswith(".pdf") or "pdf" in content_type:
            result = analyze_pdf(contents)
        elif filename.endswith(".docx") or "wordprocessingml" in content_type:
            result = analyze_docx(contents)
        elif content_type.startswith("image/"):
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            result = analyze_image_document(image)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: '{file.filename}'. "
                       "Accepted: .pdf, .docx, or an image file.",
            )
        result["filename"] = file.filename
        result["detection_mode"] = "document"
        return result
    except HTTPException:
        raise 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
