# Research: Features

## Table Stakes (Must Have)
- **Dual Lagrangian Formulation**: Maximize the dual objective function to find Lagrange multipliers ($\alpha$).
- **Kernel Trick**: Implementation of Linear, RBF (Radial Basis Function), and Polynomial kernels.
- **Support Vector Selection**: Identify vectors where $\alpha > \epsilon$ for prediction.
- **Data Standardization**: Scaling features to zero mean and unit variance (critical for SVM performance).
- **Label Transformation**: Mapping input labels to $\{-1, 1\}$.

## Differentiators
- **Kernel Comparison**: Side-by-side performance analysis of Linear vs RBF vs Polynomial.
- **Decision Boundary Visualization**: Plotting (in 2D reduced space) how each kernel separates the data.
- **Hyperparameter Sensitivity**: Analysis of how $C$ (regularization) and $\gamma$ (RBF spread) affect results.

## Anti-Features
- **Black-box libraries**: Avoiding `sklearn.svm.SVC` for core training logic.
- **Multi-class Support**: Sticking to binary classification for the Breast Cancer dataset.
