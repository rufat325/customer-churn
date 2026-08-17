## Step 1 — Open `README.md`

From your project root:

```powershell
code README.md
```

If it doesn't exist, VS Code will create it.

Paste this:

````markdown
# Customer Churn Prediction API

An end-to-end machine learning project for predicting customer churn from customer demographics, historical orders, and website activity.

The project covers the complete ML lifecycle:

- Data generation and profiling
- Feature engineering
- Churn target construction
- Model training and hyperparameter tuning
- Validation and test evaluation
- Automated testing
- Model serialization
- REST API development with FastAPI
- Docker containerization
- AWS EC2 deployment

---

## 1. Project Objective

The objective is to predict whether an eligible customer will churn based on information available up to a fixed prediction date.

The project is designed to simulate a realistic customer analytics problem where future customer behavior must not be used as an input feature.

The model produces:

- Churn probability
- Binary churn prediction
- Risk level

---

## 2. Project Architecture

```text
Customer Data
     |
     v
Feature Engineering
     |
     v
Train / Validation / Test Split
     |
     v
Gradient Boosting Model
     |
     v
Hyperparameter Tuning
     |
     v
Saved Model
     |
     +-------------------+
     |                   |
     v                   v
predict.py           FastAPI
                         |
                         v
                      Docker
                         |
                         v
                      AWS EC2
                         |
                         v
                    Public REST API
````

---

## 3. Data

The project uses three main datasets:

### Customers

| Column      | Description                |
| ----------- | -------------------------- |
| customer_id | Unique customer identifier |
| age         | Customer age               |
| country     | Customer country           |
| signup_date | Customer registration date |

### Orders

| Column      | Description             |
| ----------- | ----------------------- |
| order_id    | Unique order identifier |
| customer_id | Customer identifier     |
| order_date  | Order date              |
| amount      | Order amount            |

### Website Events

| Column      | Description           |
| ----------- | --------------------- |
| event_id    | Event identifier      |
| customer_id | Customer identifier   |
| event_type  | Type of website event |
| event_date  | Event date            |

The final development dataset contains approximately:

* 5,000 customers
* 20,250 orders
* 50,000 website events

---

## 4. Prediction Methodology

A fixed prediction date of:

```text
2025-10-31
```

was used.

Only customers eligible on or before the prediction date were included.

Historical information was calculated using data available before the prediction point.

Future orders were used only to construct the target variable and were excluded from model features.

This prevents direct target leakage.

---

## 5. Feature Engineering

The final feature set includes:

### Customer features

* age
* country
* tenure_days

### Historical order features

* total_orders
* total_spent
* days_since_last_order
* has_previous_order

### Website activity features

* total_events
* add_to_cart_count
* checkout_count
* login_count
* product_view_count
* events_last_30_days

The target is:

```text
churn
```

where:

```text
1 = customer churned
0 = customer did not churn
```

---

## 6. Train / Validation / Test

The dataset was split into:

```text
Training:   70%
Validation: 15%
Test:       15%
```

with stratification by the churn target.

The final reproducible training script uses:

```text
5-fold cross-validation
```

for hyperparameter selection.

---

## 7. Model

Several models were investigated during experimentation, including:

* Logistic Regression
* Gradient Boosting
* Random Forest

Gradient Boosting produced the strongest results in the final experiments.

The selected hyperparameters in the reproducible training pipeline were:

```text
learning_rate = 0.03
max_depth = 2
min_samples_leaf = 5
n_estimators = 100
```

The preprocessing and model are stored together in a scikit-learn Pipeline.

---

## 8. Model Performance

### Reproducible training pipeline

The final `src/train.py` pipeline produced:

| Metric    | Validation |      Test |
| --------- | ---------: | --------: |
| Accuracy  |      0.729 |     0.732 |
| Precision |      0.751 |     0.769 |
| Recall    |      0.907 |     0.874 |
| F1        |      0.822 |     0.818 |
| ROC-AUC   |      0.735 | **0.764** |

5-fold cross-validation ROC-AUC:

```text
0.7501
```

The test ROC-AUC of approximately **0.764** indicates useful ranking ability, although the model is not perfect.

---

## 9. Reproducibility Note

During development, notebook experiments and the final reproducible training script produced different results.

The notebook's final experiment produced:

```text
Test ROC-AUC: 0.7233
```

while the reproducible `src/train.py` pipeline produced:

```text
Test ROC-AUC: 0.7641
```

This difference was retained rather than hidden.

The final training script is treated as the authoritative reproducible training pipeline because it rebuilds the feature dataset and training process from the project source code.

This illustrates an important ML engineering principle:

> Experimental notebooks and production training pipelines must be validated for reproducibility before deployment.

---

## 10. Project Structure

```text
customer-churn/
|
├── data/
|   └── generated datasets
|
├── models/
|   └── churn_model.joblib
|
├── notebooks/
|   └── 01_data_profiling.ipynb
|
├── src/
|   ├── generate_data.py
|   ├── generate_data_v2.py
|   ├── generate_data_v3.py
|   ├── features.py
|   ├── train.py
|   ├── predict.py
|   └── api.py
|
├── tests/
|   ├── test_features.py
|   ├── test_model.py
|   └── test_api.py
|
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 11. Automated Testing

The project contains automated tests using pytest.

The test suite covers:

### Feature engineering

* Expected feature columns
* Target construction
* Binary churn labels
* Unique customer rows

### Model

* Model artifact existence
* Model loading
* Valid probability output
* Binary predictions

### API

* Health endpoint
* Prediction endpoint
* Invalid input validation

Current test result:

```text
11 passed
```

---

## 12. Running the Tests

Activate the virtual environment and run:

```powershell
python -m pytest tests -v
```

Expected result:

```text
11 passed
```

---

## 13. Training the Model

Run:

```powershell
python src/train.py
```

This:

1. Loads the datasets
2. Builds features
3. Splits the data
4. Performs cross-validation
5. Tunes Gradient Boosting
6. Evaluates validation and test performance
7. Saves the trained model

The model is saved to:

```text
models/churn_model.joblib
```

---

## 14. Local Prediction

Run:

```powershell
python src/predict.py
```

Example output:

```text
CUSTOMER CHURN PREDICTION
--------------------------
Churn probability: 85.86%
Predicted churn: 1
Risk level: HIGH
```

---

## 15. FastAPI

The prediction service is implemented using FastAPI.

Start the API locally:

```powershell
python -m uvicorn src.api:app --reload
```

The interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### Health endpoint

```text
GET /health
```

Response:

```json
{
  "status": "healthy"
}
```

### Prediction endpoint

```text
POST /predict
```

Example request:

```json
{
  "age": 35,
  "country": "DE",
  "total_orders": 2,
  "total_spent": 150,
  "days_since_last_order": 75,
  "has_previous_order": 1,
  "total_events": 5,
  "add_to_cart_count": 1,
  "checkout_count": 0,
  "login_count": 2,
  "product_view_count": 2,
  "tenure_days": 180,
  "events_last_30_days": 0
}
```

Example response:

```json
{
  "churn_probability": 0.8586,
  "predicted_churn": 1,
  "risk_level": "HIGH"
}
```

---

## 16. Docker

The application is containerized using Docker.

Build the image:

```powershell
docker build -t customer-churn:latest .
```

Run the API:

```powershell
docker run -d \
  --name customer-churn-api \
  -p 8000:8000 \
  customer-churn:latest
```

Check the container:

```powershell
docker ps
```

The Docker image includes a health check against:

```text
/health
```

---

## 17. AWS Deployment

The Dockerized API was deployed to an Amazon EC2 instance.

Deployment architecture:

```text
Internet
    |
    | Port 8000
    v
AWS EC2
    |
    v
Docker
    |
    v
FastAPI
    |
    v
Customer Churn Model
```

The EC2 instance runs the same Dockerized application that was tested locally.

The API was verified using:

```text
GET /health
POST /predict
```

The public Swagger documentation is available through the EC2 public IP on port 8000 while the instance is running.

---

## 18. Security Considerations

The current deployment is intended for demonstration and portfolio purposes.

Current setup:

* SSH restricted to the administrator's IP
* API exposed on port 8000
* Private SSH key stored locally
* `.pem` files excluded from Git
* Model artifact excluded from Git

For a production deployment, the API should additionally use:

* HTTPS
* A reverse proxy such as Nginx
* Authentication/authorization
* Restricted network access
* Secrets management
* Model artifact storage
* Monitoring and logging
* Rate limiting
* CI/CD

---

## 19. Limitations

The model has several limitations.

### Synthetic data

The development dataset is generated for educational and portfolio purposes rather than collected from a real business.

### Model performance

The final test ROC-AUC is approximately:

```text
0.764
```

This represents useful predictive ability but leaves substantial room for improvement.

### Probability calibration

The churn probability should be interpreted primarily as a risk score unless formal probability calibration is performed.

### Deployment

The current EC2 deployment is a demonstration environment rather than a hardened production service.

---

## 20. Future Improvements

Potential improvements include:

* Probability calibration
* More sophisticated churn definitions
* Additional behavioral features
* Time-based validation
* Model monitoring
* Data drift detection
* Feature drift detection
* Automated retraining
* CI/CD
* HTTPS
* Authentication
* Cloud-based model storage
* Production database integration
* Customer retention recommendation system

---

## 21. Technologies

* Python
* pandas
* NumPy
* scikit-learn
* FastAPI
* Uvicorn
* pytest
* Docker
* AWS EC2
* Git / GitHub

````

