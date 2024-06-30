import os
import torch
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import torchvision.transforms as transforms
from model import model
import logging

app = Flask(__name__)
CORS(app)

model_path = "Tumormodel2.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict_tumor(model, image_path, transform, device):
    original_image = Image.open(image_path).convert("L")
    if transform is not None:
        original_image = transform(original_image)
        original_image = original_image.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(original_image)

        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

        if predicted_class == 0:
            response = "No Tumor Detected"
        elif predicted_class == 1:
            response = "Pituitary Detected"
        elif predicted_class == 2:
            response = "Meningioma Detected"
        elif predicted_class == 3:
            response = "Glioma Detected"
    return response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if 'file1' not in request.files:
        return jsonify({'error': 'Please provide image'}), 400
    
    file1 = request.files['file1']
    file_path = 'temp1.png'
    file1.save(file_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    app.logger.info('Predicting tumor type...')
    tumor = predict_tumor(model, file_path, transform, device)
    os.remove(file_path)

    return jsonify({'tumor': tumor})
