
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as sk_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import sys
import pathlib

# Add src to path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0]))
from src.data_loader import load_and_clean_data
from src.label_encoder import encode_labels
from src.data_utils import train_test_split as my_split

DATA_PATH = "data/data.csv"
RANDOM_STATE = 42

def run_comparison():
    X_raw, y_raw = load_and_clean_data(DATA_PATH)
    y = encode_labels(y_raw)
    X = X_raw.to_numpy()

    # Use our split logic to match the validation script
    X_train_raw, X_test_raw, y_train, y_test = my_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Sklearn SVC
    svc = SVC(kernel='linear', C=1.0)
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)

    print("--- Sklearn Results ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    run_comparison()
