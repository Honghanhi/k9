from flask import Flask, request, jsonify
from ClassificationModule import Classifier
from UTF8ClassificationModule import UTF8Classifier
from HandTrackingModule import HandDetector
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load models
asl_model = Classifier("model_asl/keras_model.h5", "model_asl/labels.txt")
isl_model = Classifier("model_isl/keras_model.h5", "model_isl/labels.txt")
rsl_model = UTF8Classifier("model_rsl/keras_model.h5", "model_rsl/labels.txt")
detector = HandDetector(detectionCon=0.7)

# Preprocessing settings
offset = 20
imgSize = 300

def process_image(image_np, lang='asl'):
    hands = detector.findHands(image_np, draw=False)

    if not hands:
        return None, None, "No hand detected"

    hand = hands[0]
    x, y, w, h = hand['bbox']

    # Ensure bounding box stays within image bounds
    height, width = image_np.shape[:2]
    x1, y1 = max(x - offset, 0), max(y - offset, 0)
    x2, y2 = min(x + w + offset, width), min(y + h + offset, height)
    imgCrop = image_np[y1:y2, x1:x2]

    imgWhite = np.ones((imgSize, imgSize, 3), dtype=np.uint8) * 255
    aspect_ratio = h / w

    try:
        if aspect_ratio > 1:
            k = imgSize / h
            wCal = int(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = (imgSize - wCal) // 2
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgSize / w
            hCal = int(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = (imgSize - hCal) // 2
            imgWhite[hGap:hGap + hCal, :] = imgResize
    except Exception as e:
        return None, None, f"Image preprocessing error: {str(e)}"

    # Prediction
    try:
        if lang == 'asl':
            prediction, index = asl_model.getPrediction(imgWhite, draw=False)
            label = asl_model.list_labels[index]
        elif lang == 'isl':
            prediction, index = isl_model.getPrediction(imgWhite, draw=False)
            label = isl_model.list_labels[index]
        elif lang == 'rsl':
            prediction, index = rsl_model.getPrediction(imgWhite, draw=False)
            label = rsl_model.list_labels[index]
        else:
            return None, None, "Invalid language code"
    except Exception as e:
        return None, None, f"Model prediction error: {str(e)}"

    confidence = float(prediction[index])
    return label, confidence, None

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    lang = request.form.get('lang', 'asl').lower()

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400
    except Exception as e:
        return jsonify({'error': f'Image decoding error: {str(e)}'}), 400

    label, confidence, error = process_image(img, lang)

    if error:
        return jsonify({'error': error}), 400

    return jsonify({
        'label': label,
        'confidence': round(confidence * 100, 2)
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render sẽ tự đặt PORT
    app.run(host='0.0.0.0', port=port)
