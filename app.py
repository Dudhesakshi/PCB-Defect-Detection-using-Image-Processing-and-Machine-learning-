import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import torch


# --- Configuration ---
MODEL_PATH = "C:\\Users\\HP\\OneDrive\\Desktop\\PCB project\\runs\\detect\\pcb_defects_yolov82\\weights\\best.pt"
CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for defect detection

def load_model():
    """Load the YOLOv8 model."""
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- Function to process and detect defects ---
def detect_defects(image):
    """Detect defects in the uploaded PCB image."""
    image_np = np.array(image)
    results = model.predict(image_np, conf=CONFIDENCE_THRESHOLD)
    detections = []
    
    for box in results[0].boxes:
        b = box.xyxy[0].int().numpy()  
        
        c = int(box.cls)
        conf = float(box.conf)
        class_name = results[0].names[c]
        detections.append({'box': [int(x) for x in b], 'class': class_name, 'confidence': conf})
    
    return detections, results

# --- Function to visualize detections ---
def visualize_detections(image, detections):
    """Draw bounding boxes around detected defects."""
    
    image_np = np.array(image)
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    for detection in detections:
        box = detection['box']
        class_name = detection['class']
        confidence = detection['confidence']
        label = f"{class_name} {confidence:.2f}"
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# --- Streamlit UI ---
st.title("PCB Defect Detection")
st.markdown("### Upload a PCB Image to Check Quality")

uploaded_file = st.file_uploader("Choose a PCB Image", type=["jpg", "jpeg", "png"])

def get_quality_label(detections):
    """Determine PCB quality based on detection results."""
    if not detections:
        return "Good"
    defect_classes = [d['class'] for d in detections]
    if "Invalid" in defect_classes:
        return "Invalid"
    return "Bad"

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    
    if st.button("Detect"):
        with st.spinner("Processing..."):
            detections, results = detect_defects(image)
            quality = get_quality_label(detections)
            
            if detections:
                st.image(visualize_detections(image, detections), caption="Detected Defects", use_column_width=True)
                
                st.subheader("Detection Results:")
                for i, detection in enumerate(detections):
                    st.write(f"**Defect {i + 1}:**")
                    st.write(f"- Class: {detection['class']}")
                    st.write(f"- Confidence: {detection['confidence']:.2f}")
                    st.write(f"- Bounding Box: {detection['box']}")
            
            st.subheader(f"PCB Quality: **{quality}**")
