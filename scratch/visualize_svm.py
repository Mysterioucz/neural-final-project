
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

def visualize_kernels():
    # 1. Load and prepare data
    X_raw, y_raw = load_and_clean_data(DATA_PATH)
    y = encode_labels(y_raw)
    
    # Select two features for 2D visualization
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
    
    # Define kernels to compare
    kernel_configs = [
        {"name": "Linear", "kernel": "linear", "C": 1.0},
        {"name": "RBF", "kernel": "rbf", "C": 1.0, "gamma": "scale"},
        {"name": "Polynomial (d=3)", "kernel": "poly", "C": 1.0, "degree": 3}
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Create meshgrid once for all plots
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    for i, config in enumerate(kernel_configs):
        ax = axes[i]
        print(f"Training SVM with {config['name']} kernel...")
        
        # Train SVM
        model_params = {k: v for k, v in config.items() if k != "name"}
        model = SVM(**model_params)
        model.fit(X_train, y_train)
        
        # Predict over meshgrid
        Z = model.predict(grid_points)
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        ax.contourf(xx, yy, Z, alpha=0.2, cmap='coolwarm')
        
        # Plot training points
        ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], 
                   c='red', s=20, edgecolors='k', alpha=0.5, label='Malignant (Train)')
        ax.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1], 
                   c='blue', s=20, edgecolors='k', alpha=0.5, label='Benign (Train)')
        
        # Highlight support vectors
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], 
                   s=60, facecolors='none', edgecolors='green', 
                   linewidths=1.5, label='Support Vectors')
        
        # Metrics
        y_pred = model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        
        ax.set_title(f"{config['name']} Kernel\nAccuracy: {accuracy * 100:.2f}%")
        ax.set_xlabel('Std radius_mean')
        if i == 0:
            ax.set_ylabel('Std texture_mean')
            ax.legend(loc='lower right', fontsize='small')

    plt.tight_layout()
    
    # Save result
    pathlib.Path("results").mkdir(exist_ok=True)
    save_path = "results/svm_kernels_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison saved to {save_path}")

if __name__ == "__main__":
    visualize_kernels()
