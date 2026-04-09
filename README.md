# Deepfake Face Detection System

A high-precision deep learning pipeline to distinguish between authentic human faces and AI-generated (GAN) deepfakes using **InceptionResNetV2** and **Streamlit**.

---

## 🚀 Overview
This project addresses the growing threat of synthetic media by utilizing Transfer Learning to detect microscopic artifacts in facial images. The system achieves high accuracy by analyzing high-frequency textural inconsistencies that are invisible to the naked eye.

## 📁 Repository Structure
* **`fake_vs_real.ipynb`**: The research sandbox used for data exploration, model training, and fine-tuning experiments.
* **`model.py`**: Contains the core architecture logic, including the InceptionResNetV2 base and the custom classification head.
* **`main.py`**: The entry point for the **Streamlit** web application, handling the UI and real-time inference pipeline.

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow 2.x, Keras, InceptionResNetV2
* **Image Processing:** OpenCV, PIL
* **Frontend:** Streamlit
* **Environment:** Python 3.10+

## ⚙️ How It Works
1. **Preprocessing:** Images are resized to 299x299 and normalized to the [-1, 1] range.
2. **Feature Extraction:** The model extracts 1,536-dimensional feature vectors using a hybrid Inception-Residual architecture.
3. **Classification:** A Sigmoid-activated dense layer outputs a probability score (0 = Fake, 1 = Real).
4. **Visualization:** OpenCV helps in identifying and isolating facial regions for analysis.

## 📥 Installation & Usage

### 1. Clone the repository
```bash
git clone [https://github.com/Padmaja3457/Deep_fake_face_detection.git](https://github.com/Padmaja3457/Deep_fake_face_detection.git)
cd Deep_fake_face_detection
### 2. Install Dependencies
```bash
pip install -r requirements.txt
### 3. Run the Application
```bash
streamlit run main.py
