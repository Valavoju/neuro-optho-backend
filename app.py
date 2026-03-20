# ==========================
# Neuro-Optho AI Backend API
# ==========================

import os
import torch
import timm
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import scipy.ndimage as ndi
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from tensorflow.keras.applications.efficientnet import preprocess_input

# ==========================
# DEVICE
# ==========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", DEVICE)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================
# LOAD MODELS
# ==========================

print("Loading models...")

glaucoma_model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=1
).to(DEVICE)

glaucoma_model.load_state_dict(
    torch.load(os.path.join(BASE_DIR,"models/03_eye_glaucoma_classifier_efficientnet.pth"),
               map_location=DEVICE)
)
glaucoma_model.eval()

dr_model = tf.keras.models.load_model(
    os.path.join(BASE_DIR,"models/dr_final_messidor.keras")
)

alz_model = tf.keras.models.load_model(
    os.path.join(BASE_DIR,"models/alz_densenet_best.keras")
)

vessel_model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
).to(DEVICE)

vessel_model.load_state_dict(
    torch.load(os.path.join(BASE_DIR,"models/03_eye_vessel_unet.pth"),
               map_location=DEVICE)
)
vessel_model.eval()

print("All models loaded successfully.")

# ==========================
# TRANSFORMS
# ==========================

glaucoma_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

vessel_transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor()
])

# ==========================
# BIOMARKERS
# ==========================

def extract_biomarkers(image_tensor):

    with torch.no_grad():
        pred = vessel_model(image_tensor.unsqueeze(0).to(DEVICE))
        pred = torch.sigmoid(pred).cpu().squeeze().numpy()

    binary_mask = (pred > 0.5).astype(np.uint8)

    skeleton = skeletonize(binary_mask)
    vessel_length = np.sum(skeleton)

    neighbor_kernel = np.array([[1,1,1],[1,0,1],[1,1,1]])
    neighbor_count = ndi.convolve(skeleton.astype(int),
                                  neighbor_kernel,
                                  mode="constant")

    branch_points = np.logical_and(skeleton==1,
                                   neighbor_count>=3)

    num_branches = np.sum(branch_points)
    branch_density = num_branches / vessel_length if vessel_length!=0 else 0

    endpoints = np.logical_and(skeleton==1,
                               neighbor_count==1)

    num_endpoints = np.sum(endpoints)
    tortuosity = vessel_length / num_endpoints if num_endpoints!=0 else 0

    vessel_density = np.sum(binary_mask)/binary_mask.size

    return vessel_density,branch_density,tortuosity

def compute_eye_risk(density):
    if density < 0.11:
        return 40
    elif density < 0.13:
        return 20
    return 0

# ==========================
# MRI PREPROCESS
# ==========================

def preprocess_mri_backend(pil_img):

    img = np.array(pil_img)

    if len(img.shape)==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)
    _,th = cv2.threshold(img,10,255,cv2.THRESH_BINARY)
    coords = cv2.findNonZero(th)

    if coords is not None:
        x,y,w,h = cv2.boundingRect(coords)
        img = img[y:y+h,x:x+w]

    img = cv2.resize(img,(224,224))
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    img = img/255.0

    return np.expand_dims(img,0)

# ==========================
# FLASK
# ==========================

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Neuro-Optho AI Backend Running"

@app.route("/predict", methods=["POST"])
def predict():

    fundus = Image.open(request.files["fundus"]).convert("RGB")
    brain = Image.open(request.files["brain"]).convert("RGB")

    # Glaucoma
    tensor = glaucoma_transform(fundus).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        glaucoma_prob = torch.sigmoid(glaucoma_model(tensor)).item()

    # DR
    img = np.array(fundus.resize((300,300)))
    img = preprocess_input(img)
    img = np.expand_dims(img,0)
    dr_prob = float(dr_model.predict(img)[0][0])

    # Biomarkers
    vessel_tensor = vessel_transform(fundus)
    density,branch_density,tortuosity = extract_biomarkers(vessel_tensor)
    eye_score = compute_eye_risk(density)

    # Alzheimer
    brain_tensor = preprocess_mri_backend(brain)
    alz_pred = alz_model.predict(brain_tensor)[0]
    idx = int(np.argmax(alz_pred))

    classes = ["Mild Demented","Moderate Demented",
               "Non Demented","Very Mild Demented"]

    stage = classes[idx]

    brain_risk_map = {
        "Non Demented":0,
        "Very Mild Demented":30,
        "Mild Demented":60,
        "Moderate Demented":90
    }

    brain_score = brain_risk_map[stage]

    final_score = int(0.5*eye_score + 0.5*brain_score)

    level = "Low" if final_score<30 else "Moderate" if final_score<60 else "High"

    return jsonify({
        "glaucoma_prob":glaucoma_prob,
        "dr_prob":dr_prob,
        "alz_stage":stage,
        "eye_score":eye_score,
        "brain_score":brain_score,
        "final_score":final_score,
        "risk_level":level
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0", port=port)