# DecodeLabs Internship — Practical Task Submissions

> **Industrial Training Kit | Batch 2026**  
> **Candidate**: Raja Narasimha  
> **GitHub**: [@rajanarasimha018-art](https://github.com/rajanarasimha018-art)  
> **Repository**: [decodelabs_tasks](https://github.com/rajanarasimha018-art/decodelabs_tasks)

---

## ?? Repository Overview

This repository aggregates all practical tasks completed for the **DecodeLabs Industrial Training & Internship (Batch 2026)**. Each task is self-contained within its own directory, including full source code, test suites, assets, documentation, and dependency specifications.

| Task | Title | Domain | Tech Stack | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Task-1** | [The Rule-Based AI Chatbot (The Logic Engine)](./Task-1) | Deterministic AI & Logic Foundations | Python 3.10+, Pytest | Complete (5/5 Tests Passing) |
| **Task-2** | [Iris Flower Classification Using K-Nearest Neighbors](./Task-2) | Supervised Machine Learning | Scikit-Learn, Streamlit, Pandas, NumPy, Seaborn, Pytest | Complete (9/9 Tests Passing) |

---

## ??? Directory Architecture

```text
decodelabs_tasks/
¦
+-- .gitignore                      # Global gitignore (ignores caches, venvs, secrets)
+-- README.md                       # Master portfolio submission documentation (this file)
¦
+-- Task-1/                         # Task 1: The Rule-Based AI Chatbot
¦   +-- chatbot.py                  # Core conversational runtime loop with O(1) hash map lookup
¦   +-- test_chatbot.py             # Pytest automated test suite (input sanitization, fallback, loop)
¦   +-- requirements.txt            # Task 1 dependencies (pytest)
¦   +-- README.md                   # Detailed Task 1 technical documentation
¦   +-- Artificial intelligence P1.pdf # DecodeLabs Project 1 syllabus and reference guide
¦
+-- Task-2/                         # Task 2: Data Classification Using AI
    +-- app.py                      # Interactive diagnostic Streamlit dashboard
    +-- train.py                    # Scikit-Learn training pipeline, StandardScaler & KNN classifier
    +-- pytest.ini                  # Pytest configuration
    +-- requirements.txt            # Task 2 dependencies (scikit-learn, streamlit, pandas, etc.)
    +-- README.md                   # Detailed Task 2 technical documentation & metrics
    +-- models/
    ¦   +-- iris_knn_pipeline.joblib# Serialized, trained model pipeline artifact
    +-- assets/                     # High-resolution architectural and diagnostic charts
    ¦   +-- confusion_matrix.png    # Test evaluation confusion matrix
    ¦   +-- feature_space.png       # 2D feature separability visualization
    ¦   +-- k_elbow_curve.png       # Hyperparameter elbow optimization plot
    +-- tests/
        +-- test_model.py           # Pytest test suite (9 comprehensive pipeline & workflow tests)
```

---

## ?? Getting Started

### 1. Clone the Submission Repository

```bash
git clone https://github.com/rajanarasimha018-art/decodelabs_tasks.git
cd decodelabs_tasks
```

### 2. Python Environment Setup

It is recommended to use Python 3.10 or higher. You can set up a unified virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
.\venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

---

## ?? Task Details & Execution Instructions

### Task 1: The Rule-Based AI Chatbot (The Logic Engine)

#### Overview
Demonstrates deterministic AI foundations and input guardrails using pure Python without probabilistic hallucinations. Implements the Input–Process–Output (IPO) model with $O(1)$ constant-time dictionary hash map retrieval and input sanitization (`.lower().strip()`).

#### Running the Chatbot
```bash
python Task-1/chatbot.py
```

#### Running Task 1 Tests
```bash
pytest Task-1/test_chatbot.py -v
```

---

### Task 2: Iris Flower Classification Using K-Nearest Neighbors

#### Overview
Demonstrates supervised machine learning on the canonical Fisher Iris dataset. Features an end-to-end reproducible pipeline combining `StandardScaler` with a `KNeighborsClassifier` ($k=5$), hyperparameter elbow curve optimization, zero-leakage test validation, and an interactive diagnostic Streamlit web interface.

#### Install Task 2 Dependencies
```bash
pip install -r Task-2/requirements.txt
```

#### Retraining the Model (Generates Artifacts & Visualizations)
```bash
cd Task-2
python train.py
cd ..
```

#### Launching the Streamlit Web Application
```bash
cd Task-2
streamlit run app.py
```

#### Running Task 2 Tests
```bash
cd Task-2
pytest tests/ -v
cd ..
```

---

## ?? Comprehensive Test Summary

All automated test suites across all included tasks have been verified locally:

```text
Task-1 Test Results (test_chatbot.py):
============================== 5 passed in 0.10s ==============================

Task-2 Test Results (tests/test_model.py):
============================== 9 passed in 5.63s ==============================
```

---

## ?? Security & Code Hygiene

- **Zero Secrets / Tokens**: No API keys, credentials, or `.env` files are tracked in version control.
- **Cache Cleaned**: `__pycache__`, `.pytest_cache`, and virtual environment directories are strictly excluded via `.gitignore`.
- **Existing Repositories Preserved**: Project 2 original standalone repository remains intact at [Task-2-RajaNarasimha](https://github.com/rajanarasimha018-art/Task-2-RajaNarasimha).

---

## ?? Contact & Submission Info

- **Candidate**: Raja Narasimha
- **Email**: rajanarasimha.018@gmail.com
- **GitHub**: [rajanarasimha018-art](https://github.com/rajanarasimha018-art)
- **Submission Repository**: [https://github.com/rajanarasimha018-art/decodelabs_tasks](https://github.com/rajanarasimha018-art/decodelabs_tasks)
