import pandas as pd
import joblib
import os
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

#Constants
DATA_PATH = "data/telco_churn.csv"
MODEL_PATH = "models/churn_pipeline.pkl"

NUMERICAL_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']

CATEGORICAL_COLS = [
    'gender', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

def load_data(path):
    df = pd.read_csv(path)
    df = df.drop(columns=['customerID'])
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    return df

def build_pipeline():
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, NUMERICAL_COLS),
        ('cat', categorical_pipeline, CATEGORICAL_COLS)
    ])
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])

def train():
    # Load
    df = load_data(DATA_PATH)
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build and tune
    pipeline = build_pipeline()
    param_grid = {
        'model__C': [0.01, 0.1, 1, 10],
        'model__solver': ['lbfgs', 'liblinear']
    }
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)

    # MLflow tracking
    mlflow.set_experiment("telco-churn-pipeline")
    with mlflow.start_run(run_name="lr_production_run"):
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(best_model, name="churn_model")

        # Save locally
        os.makedirs("models", exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)

        print(f"Done. Accuracy: {acc:.4f} | F1: {f1:.4f}")
        print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()