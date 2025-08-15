# COMPSCI 284 – Homework Assignments

This repository contains solutions for multiple homework assignments completed as part of **COMPSCI 284: AI in Biology and Medicine**.

---

## HW1 – COMPSCI 284
[![Open In Colab](https://colab.research.google.com/github/Eeshababar/AI-in-Biology-and-Medicine/blob/main/HW1_COMPSCI284.ipynb)](https://colab.research.google.com/github/Eeshababar/AI-in-Biology-and-Medicine/blob/main/HW1_COMPSCI284.ipynb)

**Highlights:**
- Dataset: **148 data points** and **4 features**.
- Performed exploratory data analysis and feature visualization:
  - Plots between Feature 1 & Feature 2, Feature 1 & Feature 3, Feature 1 & Feature 4.
- Applied **KNN model selection**:
  - Found optimal **K = 50** (minimum validation error before increasing again).

**File:** `HW1_COMPSCI284.ipynb`

---

## HW2 – COMPSCI 284
**Highlights:**
- Polynomial regression with **training, validation, and test splits**.
- Model complexity analysis:
  - Found **degree d = 10** to minimize test error before overfitting.
- Cross-validation:
  - MSE trends for both cross-validation and test matched:
    - Underfitting before degree 1.
    - Overfitting after degree 10.

**File:** `HW2_COMPSCI284.ipynb`

---

## HW3 – COMPSCI 284  
**Topic:** Predicting Transcription Factor (TF) Binding Sites  

**Highlights:**
- Objective: Predict binding sites for the **JUND transcription factor** in human chromosome 22 sequences.
- **Data:**
  - 101-length DNA segments (A, C, G, T) encoded as one-hot vectors.
  - Binary labels (bind / no bind) with strong class imbalance (only 0.42% binding sites).
  - Weights for loss adjustment and **chromosome accessibility** scores.
- **Approach:**
  - Built an **MLP model** with at least one hidden layer.
  - Applied **weighted loss** to handle imbalance.
  - Incorporated accessibility features into model input.
  - Used training, validation, and test splits for proper evaluation.
- **Dataset Handling:**
  - Loaded `.joblib` files containing sequences, labels, weights, and accessibility scores.
  - Created PyTorch `DataLoader` objects for mini-batch gradient descent.

**File:** `HW3_COMPSCI284.ipynb`

