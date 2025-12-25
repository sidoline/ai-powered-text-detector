# 🔍 AI Text Detection & Recognition (YOLO + EasyOCR)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv11](https://img.shields.io/badge/Model-YOLOv11-green)
![Gradio](https://img.shields.io/badge/GUI-Gradio-orange)

An intelligent application that uses Deep Learning to **detect** text regions in images (using a fine-tuned YOLO model) and **read** the content (using EasyOCR). The project features a user-friendly web interface powered by Gradio.

## 🚀 Features

* **Advanced Detection:** Uses a custom trained YOLOv11 model (fine-tuned on SROIE and general text datasets).
* **Robust OCR:** Integrates `EasyOCR` to read text even in challenging conditions (angled text, low contrast).
* **GPU Acceleration:** Automatically detects CUDA to speed up inference times.
* **Interactive GUI:** Upload images, paste from clipboard, or use your webcam directly in the browser.

## 🛠️ How It Works

1.  **Input:** The user provides an image (upload or webcam).
2.  **Detection (YOLO):** The model scans the image and identifies bounding boxes where text is present.
3.  **Cropping:** The script dynamically crops these regions from the original image.
4.  **Recognition (EasyOCR):** Each cropped region is passed to the OCR engine to extract string data.
5.  **Output:** The original image is annotated with bounding boxes, and the extracted text is displayed in a list.

## 📦 Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git](https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git)
    cd YOUR-REPO-NAME
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If you have an NVIDIA GPU, ensure you have the correct CUDA-enabled PyTorch version installed).*

3.  **Download/Place Model Weights**
    * Ensure your trained model file (`best.pt`) is placed in the folder `general text/`.
    * *If no model is found, the app will auto-download a standard YOLO model for testing.*

## 🎮 Usage

Run the application with a single command:

```bash
python app.py