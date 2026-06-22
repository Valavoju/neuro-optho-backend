# 🧠 Neuro-Optho AI

### 👁️ Retinal Disease Detection | 🧠 Alzheimer's Classification | 🔬 Cognitive Risk Assessment

---

## 🏷️ Badges

<p align="center">

<img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python"/>
<img src="https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?style=for-the-badge&logo=tensorflow"/>
<img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch"/>
<img src="https://img.shields.io/badge/Flask-Backend-black?style=for-the-badge&logo=flask"/>
<img src="https://img.shields.io/badge/HuggingFace-Model%20Hosting-yellow?style=for-the-badge&logo=huggingface"/>

</p>

---

# 📌 Overview

Neuro-Optho AI is a multimodal deep learning framework developed for early cognitive risk assessment and neurodegenerative disease screening using retinal fundus images and brain MRI scans.

The system combines retinal disease detection, retinal vessel segmentation, retinal biomarker analytics, Alzheimer's disease classification, multimodal risk fusion, and explainable AI to generate a comprehensive cognitive risk score.

---

# 🎯 Key Features

- 👁️ Glaucoma Detection using EfficientNet
- 👁️ Diabetic Retinopathy Detection
- 🌐 Retinal Vessel Segmentation using U-Net
- 📊 Vessel Biomarker Extraction
- 🧠 Alzheimer's Disease Classification using DenseNet
- 🎯 Eye Cognitive Risk Score Generation
- 🎯 Brain Risk Score Generation
- 📈 Multimodal Risk Fusion
- 🔍 Explainable AI using Grad-CAM
- 📄 Automated Medical Report Generation
- ☁️ Cloud Deployment Ready

---

# 🧠 AI Models Used

| Model | Purpose |
|---------|---------|
| EfficientNet-B0 | Glaucoma Detection |
| EfficientNet | Diabetic Retinopathy Detection |
| U-Net (ResNet34 Encoder) | Retinal Vessel Segmentation |
| DenseNet | Alzheimer's Disease Classification |
| Grad-CAM | Explainable AI Visualization |

---

# ⚙️ System Workflow

```text
Fundus Image
     │
     ▼
Image Preprocessing
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
Image Preprocessing
     │
     ▼
Alzheimer Classification
(DenseNet)
     │
     ▼
Brain Risk Score

Eye Risk Score + Brain Risk Score
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
      Clinical Report Output
```

---

# 🚦 Output Parameters

| Output | Description |
|----------|----------|
| Glaucoma Probability | Risk of Glaucoma |
| DR Probability | Risk of Diabetic Retinopathy |
| Vessel Density | Retinal Vascular Density |
| Branch Density | Retinal Branching Pattern |
| Tortuosity | Vessel Curvature Measure |
| Eye Risk Score | Ocular Cognitive Risk |
| Alzheimer Stage | MRI Classification Output |
| Brain Risk Score | Neurological Risk |
| Final Cognitive Risk Score | Combined Risk Assessment |
| Risk Level | Low / Moderate / High |

---

# 🛠️ Technology Stack

## Backend

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Flask-CORS](https://img.shields.io/badge/Flask--CORS-4B8BBE?style=for-the-badge)

### Deep Learning & AI

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
![GradCAM](https://img.shields.io/badge/GradCAM-Explainable_AI-yellow?style=for-the-badge)

---

# 🤗 Model Hosting

Due to GitHub file size limitations, trained model weights are hosted on Hugging Face.

**Repository:**

https://huggingface.co/Avi-Valavoju/neuro-optho-models

The backend automatically downloads:

- 03_eye_glaucoma_classifier_efficientnet.pth
- 03_eye_vessel_unet.pth
- dr_final_messidor.keras
- alz_densenet_best.keras

No manual download is required.

---

# 📡 API Endpoints

## Health Check

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

## Prediction Endpoint

```http
POST /predict
```

### Form Data

| Parameter | Type |
|------------|--------|
| fundus | Image |
| brain | Image |

### Response

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

# ▶️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/Neuro-Optho-AI.git

cd Neuro-Optho-AI
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

Backend starts on:

```text
http://127.0.0.1:10000
```

---

# 📁 Repository Structure

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

# 🚀 Deployment

Supported Platforms:

- Render
- Railway
- Hugging Face Spaces
- Docker
- VPS Servers

Model weights are downloaded automatically from Hugging Face during startup.

---

# 🎯 Applications

- Neurodegenerative Disease Screening
- Alzheimer's Disease Risk Assessment
- Ophthalmic Disease Diagnosis
- Telemedicine Systems
- Clinical Decision Support Systems
- AI-Based Healthcare Research

---

# 👨‍💻 Authors

### Faculty Mentors

- Mrs. G. Aishwarya


### Student Team

- Avinash Valavoju
- N. Joy Darren
- E. Ramya
- N. Sarayu

---

# 📄 License

This project is developed for academic research, healthcare innovation, and educational purposes.

---

# 🌟 Show Your Support

If you find this project useful:

⭐ Star this repository

🍴 Fork this repository

📢 Share it with others

---

# 🚀 Vision

> "The eye is the window to the brain. Neuro-Optho AI transforms retinal and neuroimaging data into actionable cognitive insights through multimodal artificial intelligence."
