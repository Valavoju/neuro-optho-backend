# 🧠 Neuro-Optho AI

### 👁️ Retinal Disease Detection | 🧠 Alzheimer's Classification | 🔬 Cognitive Risk Assessment

---

## 🏷️ Badges

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?style=for-the-badge&logo=tensorflow)
![React](https://img.shields.io/badge/React-Frontend-blue?style=for-the-badge&logo=react)
![Flask](https://img.shields.io/badge/Flask-Backend-black?style=for-the-badge&logo=flask)
![EfficientNet](https://img.shields.io/badge/EfficientNet-Classification-green?style=for-the-badge)
![U--Net](https://img.shields.io/badge/U--Net-Segmentation-red?style=for-the-badge)
![DenseNet](https://img.shields.io/badge/DenseNet-MRI%20Analysis-purple?style=for-the-badge)
![GradCAM](https://img.shields.io/badge/GradCAM-Explainable%20AI-yellow?style=for-the-badge)

</p>

---

## 📌 Overview

Neuro-Optho AI is a multimodal deep learning framework developed for early cognitive risk assessment and neurodegenerative disease screening by integrating retinal fundus image analysis and brain MRI classification.

The system combines:

- 👁️ Retinal Disease Detection
- 🌐 Retinal Vessel Biomarker Extraction
- 🧠 Alzheimer's Disease Classification
- 📊 Multimodal Risk Fusion
- 🔍 Explainable AI Visualization

to generate an overall cognitive risk score for clinical decision support.

---

## 🎯 Key Highlights

✨ Multimodal AI-Based Cognitive Risk Assessment

✨ Retinal Disease Detection using EfficientNet

✨ Retinal Vessel Segmentation using U-Net

✨ Alzheimer's Stage Classification using DenseNet

✨ Eye Cognitive Risk Scoring

✨ Brain Risk Scoring

✨ Explainable AI with Grad-CAM

✨ Automated Medical Report Generation

---

## 🧠 System Architecture

```text
Fundus Images
      │
      ▼
 Image Preprocessing
      │
      ▼
 EfficientNet
 (Glaucoma + DR Detection)
      │
      ▼
 U-Net
 (Vessel Segmentation)
      │
      ▼
 Biomarker Extraction
      │
      ▼
 Eye Cognitive Risk Score
      │
      ▼

MRI Images
      │
      ▼
 DenseNet
 (Alzheimer Classification)
      │
      ▼
 Brain Risk Score
      │
      ▼

 Multimodal Risk Fusion
      │
      ▼
 Final Cognitive Risk Score
      │
      ▼
 Grad-CAM Explainability
      │
      ▼
 Dashboard & PDF Reports
```

---

## ⚙️ Core Components

### 👁️ Retinal Disease Detection — EfficientNet

- Detects Glaucoma
- Detects Diabetic Retinopathy
- Generates Disease Probability Scores
- Transfer Learning Based Classification

### 🌐 Retinal Vessel Segmentation — U-Net

- Extracts Retinal Blood Vessels
- Generates Binary Vessel Masks
- Supports Biomarker Computation

Biomarkers Extracted:

- Vessel Density
- Branch Density
- Tortuosity Index
- Vascular Structural Complexity

### 🧠 Alzheimer's Classification — DenseNet

- Processes Brain MRI Images
- Classifies Alzheimer's Disease Stages
- Detects Mild Cognitive Impairment (MCI)
- Generates Brain Risk Score

### 📊 Multimodal Risk Fusion

Combines:

- Eye Cognitive Risk Score
- Brain Risk Score

Using a weighted fusion mechanism to generate:

### 🎯 Final Cognitive Risk Score

### 🔍 Explainable AI — Grad-CAM

- Generates Visual Heatmaps
- Highlights Important Regions
- Improves Clinical Interpretability
- Supports Trustworthy AI

---

## 🚦 System Outputs

| Module | Output |
|----------|----------|
| EfficientNet | Glaucoma / DR Prediction |
| U-Net | Vessel Segmentation Mask |
| Biomarker Engine | Vessel Metrics |
| DenseNet | Alzheimer's Classification |
| Fusion Engine | Cognitive Risk Score |
| Grad-CAM | Explainability Heatmap |
| Dashboard | Medical Report |

---

## 📊 Working Pipeline

```mermaid
graph TD
A[Fundus Image] --> B[Preprocessing]
B --> C[EfficientNet]
C --> D[Glaucoma & DR Detection]

B --> E[U-Net]
E --> F[Vessel Segmentation]
F --> G[Biomarker Extraction]
G --> H[Eye Cognitive Risk Score]

I[MRI Image] --> J[Preprocessing]
J --> K[DenseNet]
K --> L[Alzheimer Classification]
L --> M[Brain Risk Score]

H --> N[Risk Fusion]
M --> N

N --> O[Final Cognitive Risk Score]
O --> P[Grad-CAM]
P --> Q[Dashboard]
Q --> R[PDF Report]
```

---

## ⚡ Features

- Real-Time AI Inference
- Automated Biomarker Extraction
- Explainable Predictions
- Multimodal Risk Assessment
- Clinical Dashboard
- PDF Report Generation
- Telemedicine Ready

---

## 🛠️ Tech Stack

### Frontend

![React](https://img.shields.io/badge/React.js-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)
![Axios](https://img.shields.io/badge/Axios-5A29E4?style=for-the-badge&logo=axios&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white)

### Backend

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Flask-CORS](https://img.shields.io/badge/Flask--CORS-4B8BBE?style=for-the-badge)
![REST API](https://img.shields.io/badge/REST_API-FF6B6B?style=for-the-badge)

### Deep Learning & AI

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)

### AI Models

![EfficientNet](https://img.shields.io/badge/EfficientNet-Retinal_Disease_Detection-success?style=for-the-badge)
![U-Net](https://img.shields.io/badge/U--Net-Vessel_Segmentation-red?style=for-the-badge)
![DenseNet](https://img.shields.io/badge/DenseNet-Alzheimer_Classification-purple?style=for-the-badge)
![Grad-CAM](https://img.shields.io/badge/GradCAM-Explainable_AI-yellow?style=for-the-badge)

---

## ▶️ How to Run

### Clone Repository

```bash
git clone https://github.com/your-username/Neuro-Optho-AI.git
cd Neuro-Optho-AI
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

### Install Backend Dependencies

```bash
pip install -r requirements.txt
```

### Run Flask Backend

```bash
python app.py
```

Backend runs at:

```text
http://127.0.0.1:5000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at:

```text
http://localhost:5173
```

---

## 📁 Project Structure

```bash
Neuro-Optho-AI/
│
├── frontend/
│   ├── src/
│   ├── components/
│   ├── pages/
│   └── assets/
│
├── backend/
│   ├── app.py
│   ├── routes/
│   ├── models/
│   │   ├── EfficientNet/
│   │   ├── U-Net/
│   │   └── DenseNet/
│   │
│   ├── reports/
│   └── utils/
│
├── datasets/
│   ├── Fundus/
│   └── MRI/
│
├── assets/
│   ├── architecture.png
│   ├── workflow.png
│   ├── dashboard.png
│   ├── gradcam.png
│   └── vessel_segmentation.png
│
├── requirements.txt
└── README.md
```

---

## 📌 Applications

- Neurodegenerative Disease Screening
- Alzheimer's Risk Assessment
- Ophthalmic Disease Diagnosis
- Telemedicine Platforms
- Clinical Decision Support Systems
- Healthcare AI Research

---

## 🔮 Future Enhancements

- Longitudinal Patient Monitoring
- Electronic Health Record Integration
- Cloud Deployment
- Multi-Disease Prediction
- Mobile Application Support
- Real-Time Hospital Integration

---

## 👨‍💻 Authors

### Faculty Mentors

- Mrs. G. Aishwarya
- Mrs. P. Pavani

### Student Team

- Avinash Valavoju
- N. Joy Darren
- E. Ramya
- N. Sarayu

---

## 📄 License

This project is developed for academic research, healthcare innovation, and educational purposes.

---

## 🌟 Show Your Support

If you like this project:

⭐ Star this repository

🍴 Fork this repository

📢 Share it with the community

---

## 🚀 Vision

> "The eye is the window to the brain. Neuro-Optho AI transforms retinal and neuroimaging data into actionable cognitive insights through multimodal artificial intelligence."
