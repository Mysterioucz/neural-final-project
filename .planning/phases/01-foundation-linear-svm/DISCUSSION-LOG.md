# Phase 1: Foundation & Linear SVM - Discussion Log

## Q&A Session (2026-05-14)

### 1. SVM Class Interface
- **Question**: How should the `SVM` class be structured?
- **Options**:
    - Mirror scikit-learn (`fit`/`predict`) [Recommended]
    - Functional / Mathematical
    - You decide
- **Selected**: **Mirror scikit-learn**
- **Rationale**: Industry standard, intuitive, and compatible with evaluation tools.

### 2. Data Pipeline Implementation
- **Question**: For feature scaling (Standardization), how should we implement it?
- **Options**:
    - Manual NumPy implementation [Recommended]
    - Use `scikit-learn`'s `StandardScaler`
- **Selected**: **Manual NumPy implementation**
- **Rationale**: Aligns with the project's goal of a "custom-built pipeline" and manual math implementation.

### 3. Weight & Bias Handling
- **Question**: For the Linear SVM, how should we compute predictions?
- **Options**:
    - Compute $w$ and $b$ once [Recommended for Linear]
    - Predict using Support Vectors
    - Hybrid
- **Selected**: **Hybrid**
- **Rationale**: Efficient for Linear SVM while preparing the architecture for non-linear kernels in Phase 2.

### 4. Solver Constraints & Defaults
- **Question**: What should be the default value for $C$ (regularization), and how should we handle the QP solver's output?
- **Options**:
    - Standard Soft Margin ($C=1.0$) [Recommended]
    - Hard Margin ($C=$ very large)
    - Solver Verbosity
- **Selected**: **Standard Soft Margin ($C=1.0$)**
- **Rationale**: Provides a robust default for datasets that aren't perfectly linearly separable.
