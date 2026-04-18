# NagMe! | AI-Powered Habit Breaker

**NagMe!** is a real-time nail-biting monitor designed to help users break this subconscious habit.
By combining Google's **MediaPipe** framework with a custom-trained **Ensemble Machine Learning** model, the app
identifies target gestures through a webcam feed and provides immediate audio feedback to interrupt the user.

## The Motivation: Why I Built This

When my partner and I were in the same city, I
could give him a physical nudge or a quick word whenever I saw him subconsciously biting his nails.However, since we
transitioned to a long-distance relationship, that physical support system disappeared.

I built NagMe! to act as a digital proxy for those nudges. To make the experience more personal and effective, the app
allows users to **customize the alert voice**. Whether it's the voice of a loved one, a recording of a pet, or a
preferred motivational tone, the app replaces a physical presence with a familiar sound that triggers mindfulness at the
exact moment it's needed most.

(But really, I built it just so I can remotely nag him 💩)

---

## Live Demo

The application is deployed and can be accessed here:  
**[nag-me.onrender.com](https://nag-me.onrender.com)**

> **⚠️ Note on Loading Time:** This app is hosted on Render's free tier. If the site has been idle for more than 15
> minutes, the server will go into a "sleep" state. It may take **30–60 seconds** to spin back up on the first load.

---

## Project Structure

```text
NagMe/
├── backend/
│   ├── data/
│   │   ├── nail_biting/      # Training images for the biting class
│   │   └── safe_action/      # Training images for the safe class
│   ├── compare_models.py     # Model shootout and hyperparameter tuning
│   ├── convert_images.py     # Image-to-landmark data pipeline
│   ├── dataset.csv           # Extracted 52-feature dataset
│   ├── model.pkl             # Serialized Soft Voting Ensemble (The Brain)
│   ├── scaler.pkl            # Serialized StandardScaler (The Dictionary)
│   ├── scraper.py            # Utility for data collection/augmentation
│   └── server.py             # Flask entry point for real-time inference
└── frontend/
    ├── index.html            # Web interface
    ├── script.js             # MediaPipe camera loop and client-side logic
    └── styles.css            # UI styling
```    

## The Tech Stack

| Category             | Tools                              |
|:---------------------|:-----------------------------------|
| **Computer Vision**  | MediaPipe (Hand & Face Mesh)       |
| **Machine Learning** | Scikit-learn, XGBoost, Joblib      |
| **Backend**          | Flask (Python)                     |
| **Data Processing**  | Pandas, NumPy                      |
| **Frontend**         | JavaScript (MediaPipe Integration) |

---

## Modeling

### 1. The Distance Baseline

To justify the need for Machine Learning, I first developed a simple baseline using standard Euclidean distance
threshold between the index fingertip $(x_1, y_1)$ and the mouth center$(x_2, y_2)$.

$$d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$

**The Problem:** While fast, this model couldn't tell the difference between a bite and someone simply resting their
chin on their hand, resulting in a lot of false nags.

### 2. Feature Engineering & Extraction

I extracted a **52-dimensional feature space** using MediaPipe:

* **42 Features:** $(x, y)$ coordinates for 21 hand joints.
* **8 Features:** $(x, y)$ coordinates for 4 key mouth landmarks.
* **2 Engineered Features:** Euclidean distance and relative vertical hand-to-face position.

### 3. Handling Class Imbalance

The training data was gathered via Google Image scraping using `Selenium` and consisted of **460 'Safe'** samples and *
*180 'Biting'** samples.

* **Cost-Sensitive Learning:** I used `scale_pos_weight` in XGBoost (set to `neg/pos` ratio) to penalize the model more
  heavily
  for missing a "Bite."
* **F1-Score Optimization:** I prioritized the **F1-Score** over accuracy to balance **Precision** and **Recall**,
  ensuring the app catches the habit without being a nuisance during normal activity. A nag is already annoying by
  nature. Can't make it worse!

### 4. The Soft Voting Ensemble

The final model app aggregates predictions from below three models:

* **Random Forest:** Stabilizes the prediction against jittery webcam noise.
* **XGBoost:** Fine-tunes the decision boundary by correcting previous errors.
* **SVM (RBF Kernel):** Isolates the "Biting" cluster from "Safe" gestures using a non-linear boundary.

---

## Pipeline Architecture

1. **Client-Side:** MediaPipe extracts 52 landmarks from the webcam feed in the browser.
2. **Privacy and Efficiency:** Only numerical coordinates (`JSON`) are sent to the Flask server; raw video data never
   leaves the user's device.
3. **Inference:** The server processes the landmarks through a `StandardScaler` and the Ensemble model to predict the
   probability of nail-biting.
4. **Feedback:** If the model reaches a high-confidence consensus, the server triggers an audio alert on the frontend.

---

## Reference Scripts

- `backend/convert_images.py`: Processes raw images into 52-feature `dataset.csv`
- `backend/compare_models.py`: Performs model comparison, hyperparameter tuning via `GridSearchCV`, and exports
  `scaler.pkl` and `model.pkl`.
- `backend/server.py`: The Flask entry point for the inference engine.
- `backend/scraper.py`: Utility used to collect images from Google used for training.
- `frontend/script.js`: The Frontend engine that runs MediaPipe in browser and sends JSON data to the Flask server.
