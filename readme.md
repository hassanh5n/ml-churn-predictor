# Telco Customer Churn Prediction — End-to-End ML Pipeline

**DevelopersHub Corporation — AI/ML Engineering Internship**

---

## Objective

Build a production-ready, reusable machine learning pipeline to predict whether a telecom customer will churn, using the Telco Customer Churn dataset. The pipeline covers everything from raw data preprocessing to a deployed Streamlit web app, with experiment tracking via MLflow and containerization via Docker.

---

## Project Structure

```
churn-pipeline/
├── notebook/
│   └── exploration.ipynb       # EDA, experimentation, and model comparison
├── src/
│   ├── preprocess.py           # Preprocessing logic (scaling, encoding)
│   ├── train.py                # Model training and GridSearchCV
│   └── predict.py              # Inference script using saved pipeline
├── app/
│   └── app.py                  # Streamlit web application
├── models/
│   └── churn_pipeline.pkl      # Serialized final pipeline (joblib)
├── Dockerfile                  # Container configuration
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Methodology

### 1. Data Preprocessing
- Handled missing values and irrelevant columns (e.g., `customerID`)
- Used `ColumnTransformer` to apply `StandardScaler` on numerical features and `OneHotEncoder` on categorical features
- All preprocessing steps wrapped inside a scikit-learn `Pipeline` to prevent data leakage

### 2. Model Training
- Trained two models inside the pipeline: Logistic Regression and Random Forest Classifier
- Used `GridSearchCV` for hyperparameter tuning on both models
- Final model selected based on F1-score, the most appropriate metric for imbalanced churn data

### 3. Experiment Tracking
- Tracked all runs, parameters, and metrics using MLflow
- Experiment name: `telco-churn-pipeline`

### 4. Model Export
- Saved the full pipeline (preprocessor + model) using `joblib`
- Stored at `models/churn_pipeline.pkl` — can be loaded directly on raw input without any additional preprocessing

### 5. Deployment
- Built an interactive Streamlit web app for live churn predictions
- Containerized the app with Docker for portability and reproducibility

---

## Results

| Model               | Accuracy | F1-Score |
|---------------------|----------|----------|
| Logistic Regression | 0.8048   | 0.6032   |
| Random Forest       | 0.8041   | 0.5905   |

Logistic Regression was selected as the final model (`lr_grid.best_estimator_`). F1-score was prioritized over accuracy due to class imbalance in the dataset.

---

## Tech Stack

| Tool        | Purpose                          |
|-------------|----------------------------------|
| scikit-learn | Pipeline, models, GridSearchCV  |
| pandas / numpy | Data manipulation             |
| MLflow      | Experiment tracking              |
| joblib      | Model serialization              |
| Streamlit   | Web app deployment               |
| Docker      | Containerization                 |

---

## How to Run

### Option 1 — Run Locally

```bash
pip install -r requirements.txt
python src/train.py
streamlit run app/app.py
```

### Option 2 — Run with Docker

```bash
docker build -t churn-pipeline .
docker run -p 8501:8501 churn-pipeline
```

Open `http://localhost:8501` in your browser.

### Option 3 — MLflow UI

```bash
mlflow ui
```

Open `http://localhost:5000` to view experiment runs and metrics.

---

## Key Observations

- Logistic Regression outperformed Random Forest on F1-score despite being a simpler model, likely due to the dataset size and linear separability of churn patterns
- Wrapping preprocessing in a Pipeline eliminated the risk of data leakage and made the codebase significantly cleaner and production-safe
- MLflow made it straightforward to compare runs and select the best hyperparameters objectively

---

## Dataset

Telco Customer Churn Dataset — IBM Sample Dataset
Available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

*Part of the DevelopersHub AI/ML Engineering Internship — Task 2 of 5*