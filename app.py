import cv2
import easyocr
import torch
from ultralytics import YOLO
import os
import gradio as gr
import numpy as np

device_used = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 Hardware Check: Using {device_used.upper()} for inference.")

if device_used == 'cpu':
    print("⚠️ Warning: GPU not detected. EasyOCR might be slow.")

model_path = 'general text/best.pt'
if not os.path.exists(model_path):
    print("⚠️ Custom model not found, downloading standard YOLO model...")
    model = YOLO('yolo11n.pt')
else:
    print(f"✅ Loading Custom Model: {model_path}")
    model = YOLO(model_path)

print("⏳ Loading EasyOCR model...")
reader = easyocr.Reader(['en'], gpu=(device_used == 'cuda')) 
print("✅ EasyOCR Ready!")

def detect_text(image, confidence_threshold):
    if image is None:
        return None, "Waiting for image..."

    output_img = image.copy()

    results = model.predict(image, conf=confidence_threshold, verbose=False)

    detected_list = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            h_img, w_img, _ = image.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)

            cropped_roi = image[y1:y2, x1:x2]
            if cropped_roi.size == 0: continue 

            try:
                ocr_result = reader.readtext(cropped_roi, detail=0, paragraph=True)
                text_part = " ".join(ocr_result).strip()
            except Exception as e:
                print(f"OCR Error: {e}")
                text_part = ""

            if text_part:
                detected_list.append(text_part)
                cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(output_img, text_part, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if not detected_list:
        final_text_output = "No text detected."
    else:
        final_text_output = "\n".join(detected_list)

    return output_img, final_text_output

with gr.Blocks(title="AI Text Reader") as demo:

    gr.Markdown("## 📷 AI Text Reader (YOLO + EasyOCR + GPU)")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Input")
            input_image = gr.Image(
                label="Source", 
                type="numpy", 
                sources=["upload", "clipboard", "webcam"]
            )
            conf_slider = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.25, step=0.05, 
                label="Detection Confidence"
            )

        with gr.Column():
            gr.Markdown("### Results")
            output_image = gr.Image(label="Visual Detections")

            output_text = gr.Textbox(
                label="Extracted Text", 
                lines=10,
                interactive=False  

            )

    input_image.change(
        fn=detect_text, 
        inputs=[input_image, conf_slider], 
        outputs=[output_image, output_text]
    )

    conf_slider.change(
        fn=detect_text, 
        inputs=[input_image, conf_slider], 
        outputs=[output_image, output_text]
    )

if __name__ == "__main__":

    demo.launch(theme=gr.themes.Soft())