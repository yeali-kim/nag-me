import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


# make baseline model using simple distance calculation
class DistanceBaseline(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=-0.5):
        self.threshold = threshold
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            return (X['dist'] < self.threshold).astype(int)
        return (X[:, -2] < self.threshold).astype(int)


def prepare_data(file_path="dataset.csv"):
    df = pd.read_csv(file_path)
    X = df.drop('target', axis=1)
    y = df['target']

    # scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_pd = pd.DataFrame(X_scaled, columns=X.columns)

    joblib.dump(scaler, 'scaler.pkl')
    print(f"scaler saved.")
    return X_pd, y


def run_comparison(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True)

    # finetune xgb for f1
    xgb_params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.05, 0.1],
    }

    # grid search using f1
    neg = np.sum(y == 0)
    pos = np.sum(y == 1)

    xgb_grid = GridSearchCV(
        XGBClassifier(eval_metric='logloss', scale_pos_weight=neg / pos),
        xgb_params, cv=cv, scoring="f1", n_jobs=-1
    )
    xgb_grid.fit(X, y)

    print(f"Best XGBoost Params: {xgb_grid.best_params_}")
    best_xgb = xgb_grid.best_estimator_

    # models
    models = {
        "Distance Baseline": DistanceBaseline(threshold=-0.5),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced"),
        "SVM (RBF)": SVC(probability=True, class_weight="balanced"),
        "XGBoost (tuned)": best_xgb,
        "Ensemble": VotingClassifier(estimators=[
            ("rf", RandomForestClassifier(n_estimators=100, class_weight="balanced")),
            ("xgb", best_xgb),
            ("svm", SVC(probability=True, class_weight="balanced")),
        ], voting="soft")
    }

    results = []
    best_f1 = 0
    best_model = None

    print("Starting Model Comparison (Target: F1)\n")

    for name, model in models.items():
        # track accuracy, f1
        scores = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "f1"])

        avg_acc = scores["test_accuracy"].mean()
        avg_f1 = scores["test_f1"].mean()

        results.append({
            "Model": name,
            "Accuracy": f"{avg_acc * 100:.1f}%",
            "F1-Score": f"{avg_f1:.3f}",  # The decision metric
        })

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_model = model
            best_model_name = name

    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nModel with the highest F1 score: {best_model_name}")
    return best_model


def save_best_model(X, y, best_model):
    # re-train with full dataset and save
    best_model.fit(X, y)
    joblib.dump(best_model, 'model.pkl')
    print("Model saved to model.pkl")


if __name__ == "__main__":
    X, y = prepare_data()
    save_best_model(X, y, best_model=run_comparison(X, y))
