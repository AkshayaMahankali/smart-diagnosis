import gradio as gr
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import gdown

# ✅ Fix for Hugging Face issues
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# ------------------ MODEL DOWNLOAD ------------------
model = load_model(MODEL_PATH, compile=False)
MODEL_URL = "https://drive.google.com/uc?id=1sq-Cz_Jvtyns3bxx8_kqdt8dfZDInZMr"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)

# ------------------ LOAD MODEL ------------------
print("Loading model...")
model = load_model(MODEL_PATH)

# ------------------ CLASS LABELS ------------------
class_labels = [
    'adenocarcinoma',
    'large.cell.carcinoma',
    'normal',
    'squamous.cell.carcinoma'
]

# ------------------ GRAD-CAM ------------------
def get_gradcam(img_array):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer('block5_conv3').output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        # Fix for list output
        if isinstance(predictions, list):
            predictions = predictions[0]

        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-10)

    return heatmap.numpy()

# ------------------ STAGE CALCULATION ------------------
def calculate_stage(heatmap):
    heatmap = heatmap / (np.max(heatmap) + 1e-8)

    tumor_pixels = np.sum(heatmap > 0.5)
    total_pixels = heatmap.size

    coverage = (tumor_pixels / total_pixels) * 100

    if coverage <= 10:
        stage = "Stage I"
    elif coverage <= 25:
        stage = "Stage II"
    elif coverage <= 45:
        stage = "Stage III"
    else:
        stage = "Stage IV"

    return round(coverage, 2), stage

# ------------------ PREDICTION FUNCTION ------------------
def predict(img, patient_name, age, gender, smoking):

    if img is None:
        return None, "No image uploaded", "", "", "", "", ""

    # Resize & normalize
    img_resized = cv2.resize(img, (224, 224))
    arr = img_resized / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Prediction
    preds = model.predict(arr)[0]
    idx = np.argmax(preds)

    label = class_labels[idx]
    confidence = float(np.max(preds) * 100)

    confidences = {
        class_labels[i]: float(preds[i])
        for i in range(len(class_labels))
    }

    # Grad-CAM
    heatmap = get_gradcam(arr)

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    coverage, stage = calculate_stage(heatmap_resized)

    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    # Format confidence text
    confidence_text = f"{label} ({confidence:.2f}%)"

    return (
    superimposed,
    confidence_text,
    f"{coverage}%",
    stage,
    str(confidences)
)

# ------------------ GRADIO UI ------------------
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="numpy", label="Upload CT Scan"),
        gr.Textbox(label="Patient Name"),
        gr.Textbox(label="Age"),
        gr.Radio(["Male", "Female"], label="Gender"),
        gr.Radio(["Yes", "No"], label="Smoking")
    ],
    outputs=[
        gr.Image(label="Grad-CAM Output"),
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Tumor Coverage"),
        gr.Textbox(label="Cancer Stage"),
        gr.Textbox(label="Class Probabilities"),
    ],
    title="🧠 Lung Cancer Detection System",
    description="Upload CT scan to detect lung cancer type with Grad-CAM visualization and stage estimation"
)

# ------------------ RUN ------------------
# ------------------ RUN ------------------
if __name__ == "__main__":
    # Disable Hugging Face hot reload bug
    gr.utils.watchfn = lambda *args, **kwargs: None

    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False
    )
