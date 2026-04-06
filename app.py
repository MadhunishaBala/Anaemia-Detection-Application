import os
import base64
import io
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for Flask
import matplotlib.pyplot as plt

app = Flask(__name__)


model = tf.keras.models.load_model(r"model/CNN_Model_4.keras", compile=False)
print("✓ Model loaded!")


IMG_SIZE          = 224
META_AGE_MEAN     = 20.266304347826086
META_AGE_STD      = 2.3899143687548925
OPTIMAL_THRESHOLD = 0.6381


CENTRE_X     = 0.50
CENTRE_Y     = 0.48
ROI_FRACTION = 0.38


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


def extract_palm_roi(img_rgb):
    h, w = img_rgb.shape[:2]
    cx   = int(w * CENTRE_X)
    cy   = int(h * CENTRE_Y)
    half = int(min(h, w) * ROI_FRACTION / 2)
    half = max(10, half)
    cx   = max(half, min(cx, w - half))
    cy   = max(half, min(cy, h - half))
    return img_rgb[cy - half:cy + half, cx - half:cx + half]


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
        largest     = max(contours, key=cv2.contourArea)
        filled_mask = np.zeros_like(skin_mask)
        cv2.drawContours(filled_mask, [largest], -1, 255, thickness=cv2.FILLED)
        skin_mask   = filled_mask
    return cv2.bitwise_and(img_bgr, img_bgr, mask=skin_mask)


def extract_nail_roi(img_bgr):
    H, W   = img_bgr.shape[:2]
    half_w = int(W * W_FRAC / 2)
    half_h = int(H * H_FRAC / 2)
    cx     = int(W * CX)
    cy     = int(H * CY)
    x1, y1 = max(0, cx - half_w), max(0, cy - half_h)
    x2, y2 = min(W, cx + half_w), min(H, cy + half_h)
    roi         = img_bgr[y1:y2, x1:x2]
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
    return np.expand_dims(roi, axis=0), roi  # return both batch and raw

def preprocess_nail(file):
    nparr   = np.frombuffer(file.read(), np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    roi     = extract_nail_roi(img_bgr)
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi_rgb = cv2.resize(roi_rgb, (IMG_SIZE, IMG_SIZE))
    roi_rgb = roi_rgb.astype(np.float32) / 255.0
    return np.expand_dims(roi_rgb, axis=0), roi_rgb  # return both batch and raw

def preprocess_meta(age, gender):
    age_norm = (age - META_AGE_MEAN) / META_AGE_STD
    return np.array([[age_norm, float(gender)]], dtype=np.float32)

# ── Attention map extraction ──────────────────────────────────────
def extract_attention_maps(model, palm_batch, nail_batch, meta_arr):
    inputs = [palm_batch, nail_batch, meta_arr]

    palm_self_model  = tf.keras.Model(inputs=model.input,
                                      outputs=model.get_layer("palm_self_attn").output)
    nail_self_model  = tf.keras.Model(inputs=model.input,
                                      outputs=model.get_layer("nail_self_attn").output)
    palm_cross_model = tf.keras.Model(inputs=model.input,
                                      outputs=model.get_layer("palm_cross_attn").output)
    nail_cross_model = tf.keras.Model(inputs=model.input,
                                      outputs=model.get_layer("nail_cross_attn").output)

    def to_heatmap(attn):
        scores = np.mean(np.abs(attn.numpy()[0]), axis=-1)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return scores.reshape(7, 7)

    return {
        "palm_self":  to_heatmap(palm_self_model(inputs,  training=False)),
        "nail_self":  to_heatmap(nail_self_model(inputs,  training=False)),
        "palm_cross": to_heatmap(palm_cross_model(inputs, training=False)),
        "nail_cross": to_heatmap(nail_cross_model(inputs, training=False)),
    }

# ── Overlay heatmap on image ──────────────────────────────────────
def overlay_heatmap(img, heatmap_7x7, colormap, alpha=0.4):
    img_uint8       = (img * 255).astype(np.uint8)
    heatmap_resized = cv2.resize(heatmap_7x7, (224, 224))
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_uint8, 1 - alpha, heatmap_colored, alpha, 0)

# ── Convert matplotlib figure to base64 string ───────────────────
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded

# ── Generate attention map figure ────────────────────────────────
def generate_attention_figure(palm_img, nail_img, maps):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Spatial Attention Maps", fontsize=14, fontweight="bold")

    # Row 1 — Palm
    axes[0][0].imshow(palm_img);                                          axes[0][0].set_title("Original Palm");             axes[0][0].axis("off")
    axes[0][1].imshow(maps["palm_self"], cmap="winter");                  axes[0][1].set_title("Palm Self-Attention");        axes[0][1].axis("off")
    axes[0][2].imshow(overlay_heatmap(palm_img, maps["palm_self"],  cv2.COLORMAP_WINTER)); axes[0][2].set_title("Palm Self Overlay");  axes[0][2].axis("off")
    axes[0][3].imshow(overlay_heatmap(palm_img, maps["palm_cross"], cv2.COLORMAP_COOL));   axes[0][3].set_title("Palm Cross Overlay\n(Age/Gender guided)"); axes[0][3].axis("off")

    # Row 2 — Nail
    axes[1][0].imshow(nail_img);                                          axes[1][0].set_title("Original Nail");             axes[1][0].axis("off")
    axes[1][1].imshow(maps["nail_self"], cmap="winter");                  axes[1][1].set_title("Nail Self-Attention");        axes[1][1].axis("off")
    axes[1][2].imshow(overlay_heatmap(nail_img, maps["nail_self"],  cv2.COLORMAP_WINTER)); axes[1][2].set_title("Nail Self Overlay");  axes[1][2].axis("off")
    axes[1][3].imshow(overlay_heatmap(nail_img, maps["nail_cross"], cv2.COLORMAP_COOL));   axes[1][3].set_title("Nail Cross Overlay\n(Age/Gender guided)"); axes[1][3].axis("off")

    plt.tight_layout()
    return fig_to_base64(fig)

# ── Routes ────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    age    = float(request.form["age"])
    gender = int(request.form["gender"])

    palm_arr, palm_img = preprocess_palm(request.files["palm"])
    nail_arr, nail_img = preprocess_nail(request.files["nail"])
    meta_arr           = preprocess_meta(age, gender)

    class_out, reg_out = model.predict(
        [palm_arr, nail_arr, meta_arr], verbose=0
    )

    prob       = float(class_out.flatten()[0])
    hb_pred    = float(reg_out.flatten()[0])
    hb_rounded = round(hb_pred, 2)
    label      = "Anemic" if prob >= OPTIMAL_THRESHOLD else "Non-Anemic"
    confidence = prob if label == "Anemic" else 1 - prob
    severity   = get_severity(age, gender, hb_rounded)

    # Generate attention maps
    try:
        maps           = extract_attention_maps(model, palm_arr, nail_arr, meta_arr)
        attention_plot = generate_attention_figure(palm_img, nail_img, maps)
    except Exception as e:
        print(f"Attention map error: {e}")
        attention_plot = None

    return jsonify({
        "label"         : label,
        "probability"   : round(prob, 4),
        "confidence"    : round(confidence * 100, 1),
        "hb_level"      : hb_rounded,
        "severity"      : severity,
        "attention_map" : attention_plot  # base64 encoded PNG
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)