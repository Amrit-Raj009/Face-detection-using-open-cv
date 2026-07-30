# 😀 Face Detection using OpenCV and Streamlit

## 📌 Project Overview

This project is a **Face Detection Application** built using **Python, OpenCV, and Streamlit**.

The application allows users to upload an image, detects human faces using the **Haar Cascade Classifier**, and displays the detected faces by drawing bounding boxes around them.

---

## 🚀 Features

- Upload images in JPG, JPEG, and PNG formats
- Detect multiple faces in an image
- Draw bounding boxes around detected faces
- Simple and interactive web interface using Streamlit
- Real-time image processing using OpenCV

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming language |
| OpenCV | Image processing and face detection |
| Streamlit | Web application interface |
| Pillow | Image handling |
| NumPy | Image array operations |

---

## 📂 Project Structure
Face-Detection-OpenCV/
│
├── app.py # Streamlit application
├── requirements.txt # Required Python libraries
├── README.md # Project documentation

---

## ⚙️ How It Works

1. User uploads an image through the Streamlit interface.
2. The image is loaded using Pillow.
3. The image is converted into an OpenCV-compatible format.
4. OpenCV converts the image into grayscale.
5. Haar Cascade Classifier detects faces.
6. Bounding boxes are drawn around detected faces.
7. The output image is displayed with detected faces.

---

## 🧠 Face Detection Algorithm

### Haar Cascade Classifier

Haar Cascade is a machine learning-based object detection algorithm used for detecting objects in images.

In this project:

- Pre-trained Haar Cascade model is used.
- The classifier searches for facial features.
- Detected faces are returned as coordinates:
  - X position
  - Y position
  - Width
  - Height

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Amrit-Raj009/face-detection-opencv.git

🎯 Future Improvements
Add real-time webcam face detection
Add face recognition system
Improve accuracy using Deep Learning models
Add emotion detection
Deploy application on cloud platforms

Author

Amrit Raj

B.Tech Computer Science Engineering