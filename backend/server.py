import cv2
import joblib
import numpy as np
import mediapipe as mp
import base64
import os
from flask import Flask, request, jsonify, send_from_directory

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BACKEND_DIR)
FRONTEND_DIR = os.path.join(ROOT_DIR, 'frontend')
model = joblib.load(os.path.join(BACKEND_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BACKEND_DIR, "scaler.pkl"))

app = Flask(__name__,
            static_folder=FRONTEND_DIR,
            static_url_path='')

# MediaPipe setup
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

hand_landmarker = HandLandmarker.create_from_options(HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=os.path.join(ROOT_DIR, "hand_landmarker.task")),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.2,
    min_hand_presence_confidence=0.2
))

face_landmarker = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=os.path.join(ROOT_DIR, "face_landmarker.task")),
    running_mode=VisionRunningMode.IMAGE,
    min_face_detection_confidence=0.2
))


def extract_features(frame):
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    hand_result = hand_landmarker.detect(mp_image)
    face_result = face_landmarker.detect(mp_image)

    if not hand_result.hand_landmarks or not face_result.face_landmarks:
        return None

    mouth_center = face_result.face_landmarks[0][13]
    nose = face_result.face_landmarks[0][1]

    tip_ids = [4, 8, 12, 16, 20]
    best_hand = None
    global_min_dist = 999

    # pick hand with a fingertip closest to mouth
    for hand in hand_result.hand_landmarks:
        for tip_id in tip_ids:
            tip = hand[tip_id]
            d = ((tip.x - mouth_center.x) ** 2 + (tip.y - mouth_center.y) ** 2) ** 0.5
            if d < global_min_dist:
                global_min_dist = d
                best_hand = hand

    if best_hand is None:
        return None

    features = []

    # features are normalized to nose
    for lm in best_hand:
        features.extend([lm.x - nose.x, lm.y - nose.y])

    for idx in [13, 14, 78, 308]:
        lm = face_result.face_landmarks[0][idx]
        features.extend([lm.x - nose.x, lm.y - nose.y])

    # dist and spread
    features.append(global_min_dist)
    wrist, middle_tip = best_hand[0], best_hand[12]
    hand_spread = ((wrist.x - middle_tip.x) ** 2 + (wrist.y - middle_tip.y) ** 2) ** 0.5
    features.append(hand_spread)

    return np.array(features).reshape(1, -1)  # 52 features


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)


@app.route("/predict", methods=["POST"])
def predict():
    # Receive base64 image frame from browser
    data = request.json.get("frame")
    if not data:
        return jsonify({"error": "No frame provided"}), 400

    # Decode base64 → numpy frame
    img_bytes = base64.b64decode(data.split(",")[1])
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    features = extract_features(frame)
    if features is None:
        return jsonify({"prediction": "no_detection"})

    import pandas as pd

    cols = [f"h{'x' if i % 2 == 0 else 'y'}{i // 2}" for i in range(42)] + \
           [f"m{'x' if i % 2 == 0 else 'y'}{i // 2}" for i in range(8)] + \
           ["dist", "spread"]

    features_df = pd.DataFrame(features, columns=cols)
    features_scaled = scaler.transform(features_df)
    prediction = int(model.predict(features_scaled)[0])

    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
