# truthtell_Tech4Stack
# Truth Tell: Fake News Classification

Truth Tell is a machine learning project designed to classify fake news using the LIAR2 dataset. The project employs a combination of traditional machine learning models and advanced natural language processing techniques using BERT. The primary goal is to identify fake news with high accuracy and robustness.

---

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Datasets](#datasets)
5. [Models](#models)
6. [Backend and Frontend](#backend-and-frontend)
7. [Outputs](#outputs)
8. [How to Use](#how-to-use)
9. [Acknowledgments](#acknowledgments)

---

## Overview

Fake news poses a significant challenge to society. Truth Tell leverages the LIAR2 dataset to build predictive models capable of identifying fake news. The project supports both traditional machine learning models (e.g., Logistic Regression, Naive Bayes, and SVM) and deep learning approaches like BERT.

---

## Project Structure

```
truthtell_Tech4Stack-main
├── .gitignore                 # Ignored files for version control
├── README.md                  # Detailed project documentation
├── requirements.txt           # Python dependencies
├── logistic_model.pkl         # Trained Logistic Regression model
├── naive_bayes_model.pkl      # Trained Naive Bayes model
├── svm_model.pkl              # Trained SVM model
├── Datasets/                  # Folder containing data files
│   ├── train.csv              # Training dataset
│   ├── test.csv               # Testing dataset
│   └── valid.csv              # Validation dataset
├── backend/                   # Backend scripts
│   ├── backend-bert.py        # Backend for BERT-based models
│   └── backend.py             # Backend for traditional pipelines
├── frontend/                  # Frontend application
│   └── app.py                 # Frontend script
├── models/                    # Model scripts
│   ├── bert-pipe.py           # BERT pipeline implementation
│   └── traditional-pipeline.py # Traditional ML pipeline
├── output/                    # Output folder
    ├── Logistic-regression.png # Logistic Regression performance
    ├── bert.png                # BERT performance
    ├── naive-bayes.png         # Naive Bayes performance
    └── svm.png                 # SVM performance
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/truthtell.git
   ```

2. Navigate to the project directory:
   ```bash
   cd truthtell_Tech4Stack-main
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Datasets

The project uses the LIAR2 dataset, preprocessed into the following files:

- **train.csv**: Data for training models.
- **test.csv**: Data for testing models.
- **valid.csv**: Data for validation and hyperparameter tuning.

Ensure these files are located in the `Datasets` directory.

---

## Models

### Traditional Machine Learning Models

1. **Logistic Regression**:
   - Model: `logistic_model.pkl`
   - Performance plot: `output/Logistic-regression.png`

2. **Naive Bayes**:
   - Model: `naive_bayes_model.pkl`
   - Performance plot: `output/naive-bayes.png`

3. **Support Vector Machine (SVM)**:
   - Model: `svm_model.pkl`
   - Performance plot: `output/svm.png`

### BERT-Based Model

- **BERT**:
  - Script: `models/bert-pipe.py`
  - Backend: `backend/backend-bert.py`
  - Performance plot: `output/bert.png`

---

## Backend and Frontend

### Backend

- **Framework**: FastAPI
- `backend.py`: Handles preprocessing and prediction for traditional ML models.
- `backend-bert.py`: Handles BERT-based operations including tokenization and prediction.

### Frontend

- **Framework**: Streamlit
- `app.py`: Provides a user-friendly interface to interact with the models, visualize outputs, and evaluate news articles.

---

## Outputs

Performance plots for each model:

1. `Logistic-regression.png`: Performance of Logistic Regression.
2. `naive-bayes.png`: Performance of Naive Bayes.
3. `svm.png`: Performance of SVM.
4. `bert.png`: Performance of BERT.

These plots are located in the `output/` directory.

---

## How to Use

1. Start the application:
   ```bash
   python frontend/app.py
   ```

2. Access the frontend interface in your web browser (default: `http://localhost:5000`).

3. Upload a news article or enter text for classification.

4. View the prediction results and performance metrics.

---
## screenshots:
  main screen:
  
   ![alt text](image.png)

 model selection:

 ![alt text](image-1.png)

 dashboard:

1. label predicted:

![alt text](image-2.png)

2. result visualized:

![alt text](image-3.png)

---
## Acknowledgments

- **Dataset**: LIAR2 Dataset
- **Libraries**: Scikit-learn, TensorFlow, Streamlit, FastAPI, Pandas, NumPy
- **Tools**: Matplotlib for visualization

---

Feel free to contribute by submitting issues or pull requests!

