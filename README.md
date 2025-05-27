# 🛠️ PCB Defect Detection using Image Processing and Machine Learning

This project automates the inspection of Printed Circuit Boards (PCBs) using a combination of traditional image processing techniques Machine learning.

---

## 👩‍💻 Authors

- **Sakshi Dudhe**
- **Snehal Ghogare**  
- **Suhani Harsule**  
Final Year Engineering Scholars  
Prof. Ram Meghe College of Engineering & Management, Amravati

---

## 📄 Research Publication

📚 *Published in:*  
**International Journal of Innovative Research in Technology (IJIRT), Volume 11, Issue 11, April 2025**  
📜 *Paper Title:*  
**"Analysis and Overview of Printed Circuit Board Defect Detection Methods"**  
https://ijirt.org/Article?manuscript=174666  
🧾 ISSN: 2349-6002

---

## 🔍 Project Overview

The system identifies:
- Broken Tracks
- Missing Holes
- Misalignments
- Shorts and other PCB anomalies

Achieved **97.1% accuracy** using YOLOv8 with a clean and responsive UI built using **Streamlit**.

---

## 📂 Project Structure

├── dataset/
│ └── [PCB images and labels]
├── yolov8s.pt
├── notebooks/
│ └── ImageProcessingPipeline.ipynb
│ └── YOLOv8_Training_and_Inference.ipynb
├── app/
│ └── streamlit_app.py
├── requirements.txt
└── README.md

## 🚀 Getting Started

### 1. Clone the Repository


git clone https://github.com/sakshidudhe/PCB-Defect-Detection-using-Image-Processing-and-Machine-learning-.git
cd PCB-Defect-Detection-using-Image-Processing-and-Machine-learning-
2. Install Required Libraries
bash
Copy
Edit
pip install -r requirements.txt
3. Download YOLOv8 Model
Due to GitHub's file size limit, the model is not included.



🧪 How to Run
🧠 Jupyter Notebooks:
notebooks/ImageProcessingPipeline.ipynb
notebooks/YOLOv8_Training_and_Inference.ipynb

🌐 Streamlit App:
streamlit run app/streamlit_app.py


📈 Evaluation Metrics
Defect Type	Precision	Recall	mAP@0.5
Broken Track	0.96	0.97	0.97
Missing Hole	0.98	0.96	0.97
Overall Accuracy			97.1%

💡 Future Scope
Implement Explainable AI (XAI) for better model interpretability
Real-time deployment on edge devices

