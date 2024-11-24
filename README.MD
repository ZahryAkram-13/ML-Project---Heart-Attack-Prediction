# Heart Attack Prediction

## Problem Description
Heart disease is one of the leading causes of death worldwide. Early prediction of heart attacks can help save lives by enabling timely medical intervention. This project leverages machine learning techniques to predict the likelihood of a heart attack based on various patient health metrics.

## Objective
The primary goal of this project is to build a predictive model that can analyze patient data and estimate the risk of a heart attack. The model aims to assist healthcare professionals in identifying high-risk individuals and making informed decisions.

## Data
The dataset used for this project contains patient health records with attributes such as:
- **Age**
- **Gender**
- **Cholesterol levels**
- **Blood pressure**
- **Heart rate**

Each record is labeled with whether the patient experienced a heart attack (`1`) or not (`0`).  
### Data Organization:
- **`raw`**: Contains the original dataset (`dataset.csv`).
- **`processed`**: Contains the cleaned and preprocessed dataset (`cleaned_heart_attack.csv`).

The data is preprocessed to handle missing values, normalize features, and prepare it for modeling.

---

## Tools and Techniques
- **Programming Language**: Python
- **Libraries**:
  - Data manipulation: `Pandas`
  - Data visualization: `Matplotlib`, `Seaborn`
  - Machine learning: `Scikit-learn`
- **Modeling**: 
  - Logistic Regression
  - Random Forest
  - Support Vector Machines (SVM)
- **Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-Score

---

## Project Architecture
This project is organized into the following structure:

```plaintext
├── data/
│   ├── raw/                     # Raw data files (original dataset)
│   ├── processed/               # Preprocessed and cleaned datasets
│   └── heart_attack_prediction_dataset.csv.zip  # Compressed dataset
├── lib/                         # Utility functions and reusable code (to be expanded)
├── notebooks/
│   └── dev.ipynb                # Jupyter notebook for data exploration and visualization
├── src/
│   └── prepare_data.py          # Script for preprocessing and cleaning the data
├── tests/                       # Unit tests for validating functionality (to be expanded)
├── README.md                    # Project documentation
├── requirements.txt             # List of dependencies for the project
