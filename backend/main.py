from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import timm
import torchvision.transforms as transforms
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

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Deepfake model
class DeepfakeDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=2
        )

    def forward(self, x):
        return self.model(x)


print("⏳ Loading AI model...")
detector = DeepfakeDetector()
detector.eval()
print("✅ Model loaded!")

# ------------------------------------------------
# IMAGE PREDICTION
# ------------------------------------------------
def predict_image(image: Image.Image):

    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return {
            "label": "NO FACE DETECTED",
            "confidence": 0,
            "faces_detected": 0
        }

    fake_scores = []

    for (x, y, w, h) in faces:

        face = img_np[y:y+h, x:x+w]

        face_img = Image.fromarray(face)

        tensor = transform(face_img).unsqueeze(0)

        with torch.no_grad():

            output = detector(tensor)

            probs = torch.softmax(output, dim=1)

            fake_prob = probs[0][1].item()

            fake_scores.append(fake_prob)

    avg_fake = sum(fake_scores) / len(fake_scores)

    label = "FAKE" if avg_fake > 0.5 else "REAL"
    confidence = avg_fake if avg_fake > 0.5 else 1 - avg_fake

    return {
        "label": label,
        "confidence": round(confidence * 100, 2),
        "fake_probability": round(avg_fake * 100, 2),
        "real_probability": round((1 - avg_fake) * 100, 2),
        "faces_detected": len(faces)
    }


# ------------------------------------------------
# VIDEO PREDICTION
# ------------------------------------------------
def predict_video(video_path: str):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {"label": "ERROR", "confidence": 0}

    frame_scores = []
    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % 30 == 0 and len(frame_scores) < 10:

            try:

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                img = Image.fromarray(rgb)

                result = predict_image(img)

                if result["label"] != "NO FACE DETECTED":
                    frame_scores.append(result["fake_probability"])

            except:
                pass

        frame_count += 1

    cap.release()

    if not frame_scores:
        return {
            "label": "UNKNOWN",
            "confidence": 0,
            "frames_analyzed": 0
        }

    avg_fake = sum(frame_scores) / len(frame_scores)

    label = "FAKE" if avg_fake > 50 else "REAL"
    confidence = avg_fake if avg_fake > 50 else 100 - avg_fake

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "fake_probability": round(avg_fake, 2),
        "real_probability": round(100 - avg_fake, 2),
        "frames_analyzed": len(frame_scores)
    }


# ------------------------------------------------
# ROOT API
# ------------------------------------------------
@app.get("/")
def root():
    return {"message": "✅ Deepfake & Document Detection API Running!"}


# ------------------------------------------------
# IMAGE API
# ------------------------------------------------
@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):

    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files allowed!")

    try:

        contents = await file.read()

        image = Image.open(io.BytesIO(contents)).convert("RGB")

        result = predict_image(image)

        result["filename"] = file.filename
        result["type"] = "image"

        return result

    except Exception as e:
        raise HTTPException(500, str(e))


# ------------------------------------------------
# VIDEO API
# ------------------------------------------------
@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):

    if not file.content_type.startswith("video/"):
        raise HTTPException(400, "Only video files allowed!")

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
        raise HTTPException(500, str(e))

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ------------------------------------------------
# DOCUMENT API
# ------------------------------------------------
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
            raise HTTPException(
                400,
                "Unsupported file! Use PDF, DOCX, or Image."
            )

        result["filename"] = file.filename
        result["detection_mode"] = "document"

        return result

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(500, str(e))
