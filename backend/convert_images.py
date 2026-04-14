import cv2
import mediapipe as mp
import pandas as pd
import os
import numpy as np

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_face_landmarks=1, min_detection_confidence=0.5)


def get_landmarks(image_path, target_label):
    image = cv2.imread(image_path)
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process hand and face landmarks
    hand_results = hands.process(image_rgb)
    face_results = face_mesh.process(image_rgb)

    # Only keep the data if both hand and face are visible
    if not hand_results.multi_hand_landmarks or not face_results.multi_face_landmarks:
        return None

    row = []

    # hand
    hand_lms = hand_results.multi_hand_landmarks[0]
    for lm in hand_lms.landmark:
        row.extend([lm.x, lm.y])

    # mouth
    face_lms = face_results.multi_face_landmarks[0]
    mouth_indices = [0, 13, 61, 291]
    mouth_pts = []
    for idx in mouth_indices:
        lm = face_lms.landmark[idx]
        row.extend([lm.x, lm.y])
        mouth_pts.append([lm.x, lm.y])

    # index to center of mouth
    index_tip = np.array([hand_lms.landmark[8].x, hand_lms.landmark[8].y])
    mouth_center = np.mean(mouth_pts, axis=0)
    dist = np.linalg.norm(index_tip - mouth_center)
    row.append(dist)

    # wrist to top of mouth
    hand_y_rel = hand_lms.landmark[0].y - face_lms.landmark[0].y
    row.append(hand_y_rel)

    row.append(target_label)
    return row


def create_dataset(base_path="data"):
    data = []
    for label in ['0', '1']:
        folder_path = os.path.join(base_path, label)
        if not os.path.exists(folder_path):
            continue

        print(f"Processing {label} folder...")
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            features = get_landmarks(img_path, int(label))
            if features:
                data.append(features)

    cols = []
    for i in range(21): cols.extend([f'hx{i}', f'hy{i}'])
    for i in range(4): cols.extend([f'mx{i}', f'my{i}'])
    cols.extend(['dist', 'hand_y_rel', 'target'])

    df = pd.DataFrame(data, columns=cols)
    df.to_csv("dataset.csv", index=False)
    print(f"Dataset saved")


if __name__ == "__main__":
    create_dataset()