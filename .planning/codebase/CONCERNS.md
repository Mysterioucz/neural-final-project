# Concerns

## Data Quality
- **Preprocessing Required:** The raw CSV data in `data/data.csv` contains un-normalized features. SVM is sensitive to feature scaling.
- **Categorical Encoding:** The `diagnosis` column is 'M'/'B', which needs binary encoding (e.g., 1/-1).

## Implementation Risk
- **Mathematical Complexity:** Implementing SVM from the Lagrangian dual formulation requires an efficient Quadratic Programming (QP) solver.
- **Optimization Stability:** Custom solvers must handle large datasets without convergence issues.

## Technical Debt
- **Minimal Boilerplate:** Currently just a `main.py` with a print statement. Needs proper project structure (src/ layout recommended as code grows).
