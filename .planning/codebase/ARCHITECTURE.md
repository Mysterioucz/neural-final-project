# Architecture

## Overview
Currently, the project is a simple Python script-based application. It follows a monolithic structure where logic resides in the root directory.

## Core Components
- **Entry Point:** `main.py`
- **Data Directory:** `data/` containing raw datasets.

## Data Flow
1. Load data from `data/data.csv`.
2. (Proposed) Preprocess data.
3. (Proposed) Train SVM model using Lagrangian formulation.
4. (Proposed) Evaluate and output results.
