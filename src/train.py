from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import build_features


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)

CUSTOMERS_PATH = DATA_DIR / "customers.csv"
ORDERS_PATH = DATA_DIR / "orders.csv"
EVENTS_PATH = DATA_DIR / "website_events.csv"

MODEL_PATH = MODEL_DIR / "churn_model.joblib"


# ============================================================
# Load raw data
# ============================================================

print("Loading data...")

customers = pd.read_csv(CUSTOMERS_PATH)
orders = pd.read_csv(ORDERS_PATH)
website_events = pd.read_csv(EVENTS_PATH)


# ============================================================
# Feature engineering
# ============================================================

print("Building features...")

features = build_features(
    customers=customers,
    orders=orders,
    website_events=website_events,
)


print(f"Eligible customers: {len(features)}")
print(
    f"Churn rate: {features['churn'].mean():.3f}"
)


# ============================================================
# Create X and y
# ============================================================

X = features.drop(
    columns=[
        "customer_id",
        "signup_date",
        "last_order_date",
        "future_orders",
        "churn",
    ]
)

y = features["churn"]


# ============================================================
# Train / validation / test split
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42,
)


print()
print("Dataset split:")
print(f"Train:      {X_train.shape}")
print(f"Validation: {X_val.shape}")
print(f"Test:       {X_test.shape}")


# ============================================================
# Feature definitions
# ============================================================

numeric_features = [
    "age",
    "total_orders",
    "total_spent",
    "days_since_last_order",
    "has_previous_order",
    "total_events",
    "add_to_cart_count",
    "checkout_count",
    "login_count",
    "product_view_count",
    "tenure_days",
    "events_last_30_days",
]

categorical_features = [
    "country",
]


# ============================================================
# Preprocessing
# ============================================================

numeric_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median"),
    ),
    (
        "scaler",
        StandardScaler(),
    ),
])


categorical_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent"),
    ),
    (
        "onehot",
        OneHotEncoder(handle_unknown="ignore"),
    ),
])


preprocessor = ColumnTransformer([
    (
        "num",
        numeric_transformer,
        numeric_features,
    ),
    (
        "cat",
        categorical_transformer,
        categorical_features,
    ),
])


# ============================================================
# Gradient Boosting pipeline
# ============================================================

pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor,
    ),
    (
        "model",
        GradientBoostingClassifier(
            random_state=42,
        ),
    ),
])


# ============================================================
# Hyperparameter tuning
# ============================================================

print()
print("Running hyperparameter search...")

param_grid = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__max_depth": [2, 3],
    "model__min_samples_leaf": [5, 10],
}


grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
    verbose=1,
)


grid_search.fit(
    X_train,
    y_train,
)


print()
print(
    "Best CV ROC-AUC:",
    grid_search.best_score_,
)

print(
    "Best parameters:",
    grid_search.best_params_,
)


# ============================================================
# Validation evaluation
# ============================================================

y_val_pred = grid_search.predict(X_val)
y_val_prob = grid_search.predict_proba(X_val)[:, 1]


print()
print("VALIDATION RESULTS")

print(
    "Accuracy:",
    accuracy_score(y_val, y_val_pred),
)

print(
    "Precision:",
    precision_score(y_val, y_val_pred),
)

print(
    "Recall:",
    recall_score(y_val, y_val_pred),
)

print(
    "F1:",
    f1_score(y_val, y_val_pred),
)

print(
    "ROC-AUC:",
    roc_auc_score(y_val, y_val_prob),
)


# ============================================================
# Final test evaluation
# ============================================================

y_test_pred = grid_search.predict(X_test)
y_test_prob = grid_search.predict_proba(X_test)[:, 1]


print()
print("TEST RESULTS")

print(
    "Accuracy:",
    accuracy_score(y_test, y_test_pred),
)

print(
    "Precision:",
    precision_score(y_test, y_test_pred),
)

print(
    "Recall:",
    recall_score(y_test, y_test_pred),
)

print(
    "F1:",
    f1_score(y_test, y_test_pred),
)

print(
    "ROC-AUC:",
    roc_auc_score(y_test, y_test_prob),
)


# ============================================================
# Save model
# ============================================================

joblib.dump(
    grid_search.best_estimator_,
    MODEL_PATH,
)


print()
print(f"Model saved to: {MODEL_PATH}")