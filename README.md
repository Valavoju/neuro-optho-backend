# 🧠 Neuro-Optho AI

### 👁️ Retinal Disease Detection | 🧠 Alzheimer's Classification | 🔬 Cognitive Risk Assessment

---

## 📌 Overview

Neuro-Optho AI is a multimodal deep learning framework for early cognitive risk assessment and neurodegenerative disease screening. The system combines retinal fundus image analysis, retinal vessel biomarker extraction, and brain MRI classification to generate an overall cognitive risk score.

The backend automatically downloads trained model weights from Hugging Face and performs inference using EfficientNet, U-Net, and DenseNet architectures.

---

## 🎯 Key Features

- 👁️ Glaucoma Detection
- 👁️ Diabetic Retinopathy Detection
- 🌐 Retinal Vessel Segmentation
- 📊 Biomarker Extraction
- 🧠 Alzheimer's Stage Classification
- 🎯 Eye Cognitive Risk Score
- 🎯 Brain Risk Score
- 📈 Multimodal Risk Fusion
- 🔍 Grad-CAM Explainability
- 📄 Automated Report Generation

---

## 🧠 AI Models Used

| Model | Purpose |
|---------|---------|
| EfficientNet-B0 | Glaucoma Detection |
| EfficientNet | Diabetic Retinopathy Detection |
| U-Net (ResNet34 Encoder) | Retinal Vessel Segmentation |
| DenseNet | Alzheimer's Classification |
| Grad-CAM | Explainable AI |

---

## 🛠️ Tech Stack

### Backend

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Flask-CORS](https://img.shields.io/badge/Flask--CORS-4B8BBE?style=for-the-badge)

### Deep Learning

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
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

## 📊 System Workflow

```text
Fundus Image
     │
     ▼
Retinal Disease Detection
(EfficientNet)
     │
     ▼
Vessel Segmentation
(U-Net)
     │
     ▼
Biomarker Extraction
     │
     ▼
Eye Cognitive Risk Score

MRI Image
     │
     ▼
Alzheimer Classification
(DenseNet)
     │
     ▼
Brain Risk Score

Eye Score + Brain Score
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
Clinical Report
```

---

## 📡 API Endpoints

### Health Check

```http
GET /
```

Response:

```json
{
  "message": "Neuro-Optho Backend Running"
}
```

---

### Prediction Endpoint

```http
POST /predict
```

Form Data:

| Field | Type |
|---------|---------|
| fundus | Image |
| brain | Image |

Response:

```json
{
  "glaucoma_prob": 0.91,
  "dr_prob": 0.84,
  "alz_stage": "Very Mild Demented",
  "eye_score": 20,
  "brain_score": 30,
  "final_score": 25,
  "risk_level": "Low"
}
```

---

## 🤗 Model Storage

Model weights are hosted on Hugging Face due to GitHub file size limitations.

Repository:

https://huggingface.co/Avi-Valavoju/neuro-optho-models

Downloaded automatically at runtime:

- 03_eye_glaucoma_classifier_efficientnet.pth
- 03_eye_vessel_unet.pth
- dr_final_messidor.keras
- alz_densenet_best.keras

No manual download is required.

---

## ▶️ Local Setup

Clone repository:

```bash
git clone https://github.com/your-username/Neuro-Optho-AI.git

cd Neuro-Optho-AI
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run application:

```bash
python app.py
```

Backend starts at:

```text
http://127.0.0.1:10000
```

---

## 📁 Repository Structure

```text
Neuro-Optho-AI/
│
├── models/
│   └── README.md
│
├── screenshots/
│   ├── architecture.png
│   ├── workflow.png
│   ├── dashboard.png
│   ├── gradcam.png
│   └── vessel_segmentation.png
│
├── app.py
├── requirements.txt
├── runtime.txt
├── .gitignore
└── README.md
```

---

## 🚀 Deployment

This backend is deployment-ready for:

- Render
- Railway
- Hugging Face Spaces
- Docker
- VPS Servers

Model weights are automatically downloaded from Hugging Face during startup.

---

## 👨‍💻 Authors

### Faculty Mentors

- Mrs. G. Aishwarya


### Student Team

- Avinash Valavoju
- N. Joy Darren
- E. Ramya
- N. Sarayu

---

## 📄 License

Developed for academic research, healthcare innovation, and educational purposes.

---

## 🚀 Vision

> "The eye is the window to the brain. Neuro-Optho AI transforms retinal and neuroimaging data into actionable cognitive insights through multimodal artificial intelligence."
