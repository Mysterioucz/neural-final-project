
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.data_loader import load_and_clean_data
from src.label_encoder import encode_labels
from src.model import SVM
from src.preprocessing import ManualStandardScaler
from src.data_utils import train_test_split

DATA_PATH = "data/data.csv"
RANDOM_STATE = 42

def visualize_svm():
    # 1. Load and prepare data
    X_raw, y_raw = load_and_clean_data(DATA_PATH)
    y = encode_labels(y_raw)
    
    # Select two features for 2D visualization
    # radius_mean and texture_mean are usually good for separation
    features = ['radius_mean', 'texture_mean']
    X = X_raw[features].to_numpy()
    
    # Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # Standardize
    scaler = ManualStandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    # 2. Train Custom SVM
    model = SVM(kernel='linear', C=1.0)
    model.fit(X_train, y_train)
    
    # 3. Visualization
    plt.figure(figsize=(12, 8))
    
    # Create meshgrid for decision boundary
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    
    # Predict over meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and margins
    # The decision function is f(x) = sign(w*x + b)
    # We can also plot the raw decision values to show the margins
    decision_values = (np.c_[xx.ravel(), yy.ravel()] @ model.w_ + model.b_)
    decision_values = decision_values.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.2, cmap='coolwarm')
    plt.contour(xx, yy, decision_values, colors='k', levels=[-1, 0, 1], 
                alpha=0.5, linestyles=['--', '-', '--'])
    
    # Plot training points
    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], 
                c='red', label='Malignant (Train)', edgecolors='k', alpha=0.6)
    plt.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1], 
                c='blue', label='Benign (Train)', edgecolors='k', alpha=0.6)
    
    # Plot test points
    plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], 
                c='red', marker='x', label='Malignant (Test)', alpha=0.8)
    plt.scatter(X_test[y_test == -1, 0], X_test[y_test == -1, 1], 
                c='blue', marker='x', label='Benign (Test)', alpha=0.8)
    
    # Highlight support vectors
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], 
                s=100, facecolors='none', edgecolors='green', 
                linewidths=2, label='Support Vectors')
    
    plt.xlabel('Standardized radius_mean')
    plt.ylabel('Standardized texture_mean')
    plt.title('SVM Classification Boundary (Linear Kernel)')
    plt.legend(loc='best')
    
    # Save result
    pathlib.Path("results").mkdir(exist_ok=True)
    save_path = "results/svm_2d_visualization.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    
    # Also show accuracy for context
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy on these 2 features: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    visualize_svm()
