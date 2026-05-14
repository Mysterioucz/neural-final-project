
import sys
import pathlib
import numpy as np

# Allow running directly from the repo root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.data_loader import load_and_clean_data
from src.label_encoder import encode_labels
from src.preprocessing import ManualStandardScaler
from src.data_utils import train_test_split
from src.model import SVM
from src.metrics import accuracy_score

DATA_PATH = "data/data.csv"

def multi_split_validation(n_splits=10):
    X_raw, y_raw = load_and_clean_data(DATA_PATH)
    y = encode_labels(y_raw)
    X_np = X_raw.to_numpy()

    accuracies = []
    print(f"Running {n_splits} random splits...")
    for i in range(n_splits):
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_np, y, test_size=0.2, random_state=i
        )
        
        scaler = ManualStandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        
        svm = SVM(C=1.0, kernel="linear")
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        print(f"Split {i}: {acc * 100:.2f}%")

    print(f"\nMean Accuracy: {np.mean(accuracies) * 100:.2f}%")
    print(f"Std Dev: {np.std(accuracies) * 100:.2f}%")
    print(f"Min: {np.min(accuracies) * 100:.2f}%")
    print(f"Max: {np.max(accuracies) * 100:.2f}%")

if __name__ == "__main__":
    multi_split_validation()
