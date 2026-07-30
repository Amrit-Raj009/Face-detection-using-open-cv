import streamlit as st
import cv2
import numpy as np
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Face Detection App",
    page_icon="😀",
    layout="centered"
)

st.title("😀 Face Detection using OpenCV")
st.write("Upload an image and detect faces using Haar Cascade Classifier")


# -----------------------------
# Load Haar Cascade
# -----------------------------
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    st.error("Face detector failed to load")
    st.stop()


# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Read image
    image = Image.open(uploaded_file)

    image_array = np.array(image)

    # Convert RGB to BGR for OpenCV
    img = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2BGR
    )


    # -----------------------------
    # Face Detection
    # -----------------------------
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )


    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30)
    )


    # Draw rectangles
    output = img.copy()

    for (x, y, w, h) in faces:

        cv2.rectangle(
            output,
            (x, y),
            (x+w, y+h),
            (255,0,0),
            3
        )


    # Convert back BGR → RGB
    output_rgb = cv2.cvtColor(
        output,
        cv2.COLOR_BGR2RGB
    )


    # -----------------------------
    # Display Results
    # -----------------------------
    st.subheader(
        f"Faces Detected: {len(faces)}"
    )

    col1, col2 = st.columns(2)


    with col1:
        st.write("Original Image")
        st.image(
            image,
            use_container_width=True
        )


    with col2:
        st.write("Detected Faces")
        st.image(
            output_rgb,
            use_container_width=True
        )


else:
    st.info("Please upload an image")


# -----------------------------
# Footer
# -----------------------------
st.markdown(
    "---\n"
    "Built using **Python + OpenCV + Streamlit**"
)