import os
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)

# ── Load model ───────────────────────────────────────────────────
model = tf.keras.models.load_model(r"model/CNN1_Joint_BestFold.keras", compile=False)
print("✓ Model loaded!")

# ── Constants ─────────────────────────────────────────────────────
IMG_SIZE          = 224
META_AGE_MEAN     = 20.227703094482422
META_AGE_STD      = 2.31186842918396
OPTIMAL_THRESHOLD = 0.3051

# ── Palm ROI constants ────────────────────────────────────────────
CENTRE_X     = 0.50
CENTRE_Y     = 0.48
ROI_FRACTION = 0.38

# ── Nail ROI constants ────────────────────────────────────────────
CX     = 0.50
CY     = 0.45
W_FRAC = 0.25
H_FRAC = 0.40

# ── WHO Severity ──────────────────────────────────────────────────
def get_severity(age, gender, hb):
    if age < 5:
        if   hb >= 11.0: return "Non-Anemic"
        elif hb >= 10.0: return "Mild Anaemia"
        elif hb >= 7.0:  return "Moderate Anaemia"
        else:            return "Severe Anaemia"
    elif 15 <= age <= 65:
        if gender == 0:
            if   hb >= 12.0: return "Non-Anemic"
            elif hb >= 11.0: return "Mild Anaemia"
            elif hb >= 8.0:  return "Moderate Anaemia"
            else:            return "Severe Anaemia"
        else:
            if   hb >= 13.0: return "Non-Anemic"
            elif hb >= 11.0: return "Mild Anaemia"
            elif hb >= 8.0:  return "Moderate Anaemia"
            else:            return "Severe Anaemia"
    else:
        return "Unknown"

# ── Palm ROI extraction ───────────────────────────────────────────
def extract_palm_roi(img_rgb):
    h, w   = img_rgb.shape[:2]
    cx     = int(w * CENTRE_X)
    cy     = int(h * CENTRE_Y)
    half   = int(min(h, w) * ROI_FRACTION / 2)
    half   = max(10, half)
    cx     = max(half, min(cx, w - half))
    cy     = max(half, min(cy, h - half))
    return img_rgb[cy - half:cy + half, cx - half:cx + half]

# ── Nail background removal ───────────────────────────────────────
def remove_nail_background(img_bgr):
    img_ycbcr  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    lower_skin = np.array([0,   133, 77],  dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask  = cv2.inRange(img_ycbcr, lower_skin, upper_skin)
    kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skin_mask  = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask  = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN,  kernel)
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest      = max(contours, key=cv2.contourArea)
        filled_mask  = np.zeros_like(skin_mask)
        cv2.drawContours(filled_mask, [largest], -1, 255, thickness=cv2.FILLED)
        skin_mask    = filled_mask
    return cv2.bitwise_and(img_bgr, img_bgr, mask=skin_mask)

# ── Nail ROI extraction ───────────────────────────────────────────
def extract_nail_roi(img_bgr):
    H, W   = img_bgr.shape[:2]
    half_w = int(W * W_FRAC / 2)
    half_h = int(H * H_FRAC / 2)
    cx     = int(W * CX)
    cy     = int(H * CY)
    x1, y1 = max(0, cx - half_w), max(0, cy - half_h)
    x2, y2 = min(W, cx + half_w), min(H, cy + half_h)
    roi        = img_bgr[y1:y2, x1:x2]
    roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return remove_nail_background(roi_resized)

# ── Preprocessing ─────────────────────────────────────────────────
def preprocess_palm(file):
    nparr   = np.frombuffer(file.read(), np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    roi     = extract_palm_roi(img_rgb)
    roi     = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi     = roi.astype(np.float32) / 255.0
    return np.expand_dims(roi, axis=0)

def preprocess_nail(file):
    nparr   = np.frombuffer(file.read(), np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    roi     = extract_nail_roi(img_bgr)
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi_rgb = cv2.resize(roi_rgb, (IMG_SIZE, IMG_SIZE))
    roi_rgb = roi_rgb.astype(np.float32) / 255.0
    return np.expand_dims(roi_rgb, axis=0)

def preprocess_meta(age, gender):
    age_norm = (age - META_AGE_MEAN) / META_AGE_STD
    return np.array([[age_norm, float(gender)]], dtype=np.float32)

# ── Routes ────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    age    = float(request.form["age"])
    gender = int(request.form["gender"])

    palm_arr = preprocess_palm(request.files["palm"])   
    nail_arr = preprocess_nail(request.files["nail"])  
    meta_arr = preprocess_meta(age, gender)

    class_out, reg_out = model.predict(
        [palm_arr, nail_arr, meta_arr], verbose=0
    )

    prob       = float(class_out.flatten()[0])
    hb_pred    = float(reg_out.flatten()[0])
    hb_rounded = round(hb_pred, 2)                          
    label      = "Anemic" if prob >= OPTIMAL_THRESHOLD else "Non-Anemic"
    confidence = prob if label == "Anemic" else 1 - prob
    severity   = get_severity(age, gender, hb_rounded)     

    return jsonify({
        "label"      : label,
        "probability": round(prob, 4),
        "confidence" : round(confidence * 100, 1),
        "hb_level"   : hb_rounded,                         
        "severity"   : severity
    })

if __name__ == "__main__":
    # For local testing
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)