# Customer Churn Prediction API

An end-to-end machine learning project that predicts customer churn from
demographics, order history and website activity, and serves the model behind a
containerized REST API.

The project covers the full lifecycle: data generation, profiling, feature
engineering, target construction, model selection and tuning, probability
calibration, cost-based threshold selection, evaluation against a baseline *and
against a measured theoretical ceiling*, automated tests, CI, a FastAPI service,
Docker packaging and AWS EC2 deployment.

**Headline result:** test ROC-AUC **0.7231** against a 68.1% majority-class base
rate, with a measured ceiling of **0.7575**. The model captures roughly 87% of
the signal that is theoretically extractable from this data — see
[section 10](#10-how-good-can-this-model-possibly-get), which is the most
distinctive part of the project.

---

## 1. Quickstart

```bash
git clone https://github.com/rufat325/customer-churn.git
cd customer-churn

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements-dev.txt
```

Neither the generated data nor the trained model is tracked in git, so build
them before anything else:

```bash
python src/generate_data.py     # writes data/*.csv
python src/train.py             # writes models/churn_model.joblib + model_card.json
python -m pytest                # 48 passed
```

Then score a customer, or start the API:

```bash
python src/predict.py
python -m uvicorn src.api:app --reload
```

Tests that need the model artifact skip with an explanatory message rather than
failing if you have not run `train.py` yet.

---

## 2. Project Objective

Predict whether an eligible customer will churn, using only information
available on or before a fixed prediction date. Future behaviour must never
leak into the model's inputs.

The service returns a calibrated churn probability, a binary prediction at a
cost-optimal threshold, and a risk band.

### What "churn" means here

Churn is defined as **placing no order in the 30 days following the prediction
date**. This is a purchase-frequency definition rather than a
subscription-cancellation definition, and it is why the base rate is so high
(68.1%): most customers simply do not buy something in any given month.

That is a legitimate target, but read every metric with it in mind. It is also
the single biggest lever on model quality — see
[section 10](#10-how-good-can-this-model-possibly-get).

---

## 3. Project Architecture

```text
generate_data.py
      |
      v
Feature Engineering  (features.py)
      |
      v
Train / Validation / Test Split
      |
      v
Gradient Boosting + Grid Search
      |
      v
Probability Calibration (isotonic)
      |
      v
Cost-Based Threshold Selection
      |
      +---------------------------+
      |                           |
      v                           v
churn_model.joblib         model_card.json
      |                           |
      +------------+--------------+
                   |
      +------------+------------+
      |                         |
      v                         v
  predict.py                FastAPI
                                |
                                v
                             Docker
                                |
                                v
                             AWS EC2
```

---

## 4. Data

Three synthetic datasets, produced by `src/generate_data.py`: 5,000 customers,
20,298 orders (18,040 history + 2,208 target + 50 deliberate duplicates) and
50,018 website events.

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

### Learnable without being leaky

Each customer is assigned a hidden `activity_level` drawn from a Beta
distribution. It drives their order rate and their browsing rate, and is
**dropped before the CSVs are written**. So there is genuine signal connecting
past behaviour to future behaviour, but a model has to infer it from observed
history rather than reading it off a column.

The generator also injects realistic defects on purpose: 100 missing ages and
50 duplicate order rows.

### Behaviour is a rate process

For each customer, the expected number of orders in a window is

```text
rate(activity_level) x exposure_days
```

where exposure is the overlap between the window and that customer's lifetime.
The realised count is Poisson, and dates are drawn uniformly inside the
customer's own eligible range.

**This is a correction.** An earlier version drew a fixed global number of
orders, picked *who* placed them from the activity weights, then assigned each
one a date drawn uniformly across the whole calendar year — independently of
signup. The consequences were severe:

```text
orders placed before their customer signed up: 9,744 / 20,200  (48.2%)
events logged before their customer signed up: 24,842 / 50,000 (49.7%)
correlation(tenure_days, total_orders):        0.006
```

One customer signed up on 2025-12-31 and had an order dated 2025-01-03. Under
the current generator no event can precede its customer's signup, and that
correlation is **0.498**, as it should be. `src/generate_data.py` asserts both
properties at build time, and `tests/test_generate_data.py` pins them.

This was not a cosmetic fix. It reversed a modelling conclusion — see
[section 6](#6-feature-engineering).

---

## 5. Prediction Methodology

The prediction date is `2025-11-30`, the **last day of observed history,
inclusive**. The timeline is partitioned into two adjacent windows:

```text
        history window                    target window
  <------------------------->   <--------------------------->
  2025-01-01 ... 2025-11-30     2025-12-01 ... 2025-12-30
       (features)                       (label only)
                            ^
                     prediction date
```

- **History:** `order_date <= 2025-11-30`. All features are built from this.
- **Target:** `2025-11-30 < order_date <= 2025-12-30`. Label only, never a
  feature.

Both boundaries are derived from the same anchor in `src/features.py`, so they
are adjacent by construction: no order can fall in both (which would leak the
target) and none can fall in neither (which would silently discard data). A test
sweeps every offset from −2 to +32 days to enforce it.

Customers must have signed up at least 30 days before the prediction date to be
eligible: **4,156** of 5,000 customers, with a churn rate of **0.6809**.

---

## 6. Feature Engineering

### Supplied by the caller (13 columns)

| Group | Features |
| --- | --- |
| Customer | `age`, `country`, `tenure_days` |
| Orders | `total_orders`, `total_spent`, `days_since_last_order`, `has_previous_order` |
| Website | `total_events`, `add_to_cart_count`, `checkout_count`, `login_count`, `product_view_count`, `events_last_30_days` |

### Derived inside the pipeline (6 more)

`orders_per_day`, `events_per_day`, `spend_per_order`, `checkout_rate`,
`cart_rate`, `recency_ratio`.

These are computed by a `FunctionTransformer` as the first pipeline step, **not
requested from the caller**. They are deterministic functions of the columns
above, so asking a client to supply them would only create an opportunity to
disagree with training. The API contract stays at 13 fields, and train/serve
skew is structurally impossible.

### Why rates matter — and a reversed conclusion

The hidden driver is a *rate*. A raw count is that rate multiplied by exposure,
so `total_orders` conflates "how keen is this customer" with "how long have they
had the chance". Dividing by tenure separates the two:

| Feature | Spearman vs. hidden driver |
| --- | ---: |
| `total_orders` | 0.622 |
| **`orders_per_day`** | **0.729** |
| `total_events` | 0.609 |
| **`events_per_day`** | **0.806** |
| `tenure_days` | −0.009 |

Worth recording honestly: on the *old, temporally incoherent* data these rate
features made the model slightly worse, and `orders_per_day` was a weaker signal
than `total_orders` (0.633 vs 0.757). That was not a fact about rate features —
it was a symptom of the data bug. With signup dates unrelated to order dates,
dividing by tenure only added noise. Fixing the generator reversed the result.

`tenure_days` sitting near zero on its own is expected and correct: signup dates
are drawn independently of activity, so tenure predicts nothing by itself. It
matters as a *denominator*.

### Missing values

Count features are filled with `0`: no matching rows genuinely means no
activity.

`days_since_last_order` is different. Customers who have never ordered have an
undefined value, not a missing-at-random one. Filling it with the column median
told the model that a customer who has never bought anything last bought at a
typical time. It is now filled with `tenure_days` — it has been their entire
lifetime since they last ordered, which is true and monotonic with risk.

`age` retains genuine missing values and is median-imputed inside the pipeline,
which is appropriate — those are missing at random by construction.

---

## 7. Train / Validation / Test

```text
Training:   70%   (2,909)
Validation: 15%   (623)
Test:       15%   (624)
```

Stratified on churn. Hyperparameters are selected by 5-fold cross-validation on
the training split only. Calibration is fitted with its own internal CV on the
training split, which leaves validation clean for threshold selection. The test
split is touched once, at the end.

---

## 8. Model

Model families investigated in `notebooks/01_data_profiling.ipynb`: Logistic
Regression, Random Forest, Gradient Boosting. Gradient Boosting won.

```text
learning_rate    = 0.03
max_depth        = 2
min_samples_leaf = 10
n_estimators     = 100
```

5-fold CV ROC-AUC on the training split: **0.7378**

### Calibration

Gradient boosting optimises a ranking-friendly loss, so its raw scores are not
probabilities. Both isotonic and sigmoid calibration are fitted and the better
is selected on validation Brier score:

| | Validation Brier |
| --- | ---: |
| uncalibrated | 0.1850 |
| sigmoid | 0.1847 |
| **isotonic (selected)** | **0.1834** |

Reliability on validation after calibration:

| Bin | n | Mean predicted | Observed |
| --- | ---: | ---: | ---: |
| 0.2–0.4 | 78 | 0.342 | 0.385 |
| 0.4–0.6 | 127 | 0.519 | 0.551 |
| 0.6–0.8 | 170 | 0.706 | 0.641 |
| 0.8–1.0 | 246 | 0.874 | 0.874 |

Close to the diagonal, so the probabilities can be read as probabilities. The
improvement over uncalibrated is small — gradient boosting on this data was
already reasonably calibrated — but the check is what makes that claim, rather
than an assumption.

The whole thing (rate derivation, preprocessing, estimator, calibration) is one
serialized object, so the exact transformations used at training time are
applied at inference time. Feature lists live in `src/scoring.py` and are
imported by both the training script and the API.

---

## 9. Model Performance

| | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline / test | 0.6811 | 0.6811 | 1.0000 | 0.8103 | n/a | n/a |
| **Model / test** @ 0.83 | 0.5545 | 0.8731 | 0.4047 | 0.5531 | **0.7231** | 0.1900 |

"Baseline" is the majority-class classifier: predict churn for everyone.

### Reading these numbers honestly

At the operating threshold the model has **lower accuracy and lower F1 than
predicting churn for everyone**. That is not a failure, and it is not hidden
here — it is the expected consequence of two things:

1. **Accuracy is the wrong metric at a 68% base rate.** A constant classifier
   scores 0.681 while being useless.
2. **The threshold is not tuned for accuracy.** It is tuned for expected value
   under a cost model, which deliberately trades recall for precision.

The two numbers that matter are **ROC-AUC 0.7231** (the model ranks a random
churner above a random non-churner 72% of the time — something the baseline
cannot do at all) and **precision 0.8731** at the operating point (of the
customers it flags, 87% really do churn, versus 68% for blanket contact).

### Permutation importance (validation, drop in ROC-AUC)

| Feature | Δ ROC-AUC |
| --- | ---: |
| `total_events` | 0.1264 ± 0.0170 |
| `tenure_days` | 0.0906 ± 0.0125 |
| `total_orders` | 0.0575 ± 0.0160 |
| `total_spent` | 0.0039 ± 0.0015 |
| `days_since_last_order` | 0.0017 ± 0.0018 |

Browsing volume dominates, which makes sense: with 50,018 events against 18,040
orders, events are simply a larger sample from which to estimate the same
underlying rate. `tenure_days` ranks second precisely because of its role as a
denominator — permuting it corrupts every derived rate at once.

### Choosing the threshold from costs, not convention

An operating threshold should come from what a decision is worth. The cost model
is: contacting a customer costs `offer_cost`; if they were going to churn and
they accept, it earns `margin` with probability `accept_rate`.

The optimal threshold depends entirely on those numbers, so the project sweeps
them rather than quoting one figure. Threshold chosen on validation, valued on
the 624-customer test split:

| Scenario | TP value | FP value | Threshold | Contact all | Targeted | Uplift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cheap offer | +10 | −5 | 0.26 | 3,255 | 3,275 | +20 |
| **standard offer** (shipped) | +3 | −12 | **0.83** | −1,113 | **216** | **+1,329** |
| premium offer | −5 | −20 | 0.98 | −6,105 | 0 | +6,105 |

Three genuinely different regimes:

- **Cheap offer:** contacting the entire base is already profitable at a 68%
  churn rate, and the model adds almost nothing (+20, or 0.6%). Worth stating
  plainly — a model is not always the answer.
- **Standard offer:** blanket contact loses 1,113; targeting turns it into +216.
  The model converts a loss-making campaign into a profitable one.
- **Premium offer:** a true positive is worth −5, so no contact is ever
  worthwhile. The optimiser correctly selects a threshold that contacts nobody.
  The right answer is "do not run this campaign".

The shipped threshold of **0.83** comes from the standard scenario and is
persisted in `models/model_card.json`, which the API reads at startup and serves
at `GET /model`. Changing the economics changes the threshold without touching
the model.

Risk bands are anchored to that threshold rather than to fixed cut points:
`LOW` below it, and the region above it split into `MEDIUM` and `HIGH`. Fixed
bands at 0.50/0.75 left `MEDIUM` unreachable once the tuned threshold rose above
0.75; a test now asserts all three bands are reachable at any threshold.

---

## 10. How Good Can This Model Possibly Get?

Full analysis: [`notebooks/02_headroom_analysis.ipynb`](notebooks/02_headroom_analysis.ipynb)

A test ROC-AUC of 0.72 invites an obvious question — is that good, or is there
another 0.15 on the table? Normally you cannot answer it. Here you can, because
the generating process is known.

Since every customer's hidden `activity_level` is recoverable from the seeded
generator, we can build an **oracle** that sees the driver directly. Nothing
built from observed behaviour can beat a model that already knows the quantity
that behaviour is a noisy measurement of. Its score is the ceiling.

| Feature set | Test ROC-AUC |
| --- | ---: |
| Raw counts only | 0.7139 |
| Shipped (raw + derived rates) | 0.7195 |
| **ORACLE: hidden `activity_level` alone** | **0.7575** |
| Oracle + observed features | 0.7611 |

*(Fixed hyperparameters across all four, so the only thing varying is the
feature set. The shipped model's tuned and calibrated score of 0.7231 is
slightly higher than the 0.7195 shown here.)*

**The ceiling is 0.7575, not 1.0 and not 0.85.** Knowing every customer's true
propensity *perfectly* still only scores about 0.76, because the target asks
something inherently noisy: not "is this customer disengaging" but "will this
specific customer happen to order in these specific 30 days".

Two consequences:

1. **Feature engineering has little room left.** The shipped model captures
   ~87% of the extractable signal. The derived rate features closed 13% of the
   gap between raw counts and the ceiling; the remainder is thin.
2. **Raising the ceiling requires changing the target, not the model.** A longer
   window, or a definition based on sustained disengagement rather than a single
   30-day gap, would carry more signal per label. That is a problem-framing
   change, and it dominates anything available on the modelling side.

This is also why the project reports ROC-AUC rather than accuracy as its
headline.

---

## 11. The Target Window Bug

An earlier version reported a test ROC-AUC of **0.7641** from `src/train.py`
while the notebook reported **0.7233**, and the README described this as an
unexplained reproducibility issue. It was not unexplained: `src/features.py` had
an off-by-one error in the target window.

The generator lays out orders in two blocks — history through 2025-11-30, target
from 2025-12-01. The old feature code derived its windows independently:

```python
# before
historical_orders = orders[orders["order_date"] <  prediction_date]
future_orders     = orders[(orders["order_date"] >= prediction_date)
                         & (orders["order_date"] <  prediction_date + 30d)]
```

That window, `[2025-11-30, 2025-12-30)`, was shifted one day early and did two
wrong things at once:

1. **Swallowed 2025-11-30** — 42 orders from the historical block were treated
   as future behaviour and used to build the label.
2. **Dropped 2025-12-30** — 82 orders from the future block fell outside the
   window entirely.

61 customers received the wrong label.

```python
# after
target_start = prediction_date + pd.Timedelta(days=1)
target_end   = prediction_date + pd.Timedelta(days=target_window_days)

historical_orders = orders[orders["order_date"] <= prediction_date]
future_orders     = orders[(orders["order_date"] >= target_start)
                         & (orders["order_date"] <= target_end)]
```

Correcting it reproduced the notebook's result to five decimal places,
confirming the diagnosis. **The inflated 0.7641 was the bug, not the
achievement** — a mislabelled target made the problem look easier than it is.

The lesson worth keeping: when a notebook and a pipeline disagree, that is a
defect to diagnose, not a curiosity to document. Window boundaries should come
from a single anchor and be pinned by tests, because off-by-one errors in date
filters are silent.

---

## 12. Project Structure

```text
customer-churn/
|
+-- .github/workflows/ci.yml   test + docker smoke test on every push
|
+-- data/                      generated CSVs (not tracked)
+-- models/
|   +-- churn_model.joblib     calibrated pipeline (not tracked)
|   +-- model_card.json        params, metrics, threshold, importances
|
+-- notebooks/
|   +-- 01_data_profiling.ipynb    exploration and model selection
|   +-- 02_headroom_analysis.ipynb how good can this model get
|
+-- src/
|   +-- generate_data.py       synthetic dataset generator
|   +-- features.py            feature engineering + target construction
|   +-- train.py               training, calibration, threshold selection
|   +-- scoring.py             shared contract: paths, features, thresholds
|   +-- predict.py             CLI demo
|   +-- api.py                 FastAPI service
|
+-- tests/
|   +-- test_generate_data.py
|   +-- test_features.py
|   +-- test_model.py
|   +-- test_api.py
|
+-- Dockerfile
+-- pytest.ini
+-- requirements.txt           runtime dependencies
+-- requirements-dev.txt       runtime + test dependencies
+-- README.md
```

`src/scoring.py` exists so the model path, feature lists, derived-feature logic,
threshold and risk bands are defined once. Before it, `predict.py` reported two
risk levels and `api.py` reported three.

---

## 13. Automated Testing

```bash
python -m pytest
```

Expected: `48 passed`.

**Generator** — no order or event precedes its customer's signup; tenure and
order volume are positively related (the regression test for the incoherence
bug); history and target blocks stay separated; generation is deterministic
under a seed; deliberate defects survive.

**Feature engineering** — expected columns, target construction, binary labels,
unique rows, no missing values in count features, sentinel imputation, four
boundary tests plus a sweep asserting the history and target windows never
overlap and never leave a gap.

**Model** — artifact loads, probabilities in range, the pipeline accepts only
documented input columns, batch and single scoring agree, risk bands are
reachable and never contradict the prediction at any threshold, derived features
survive zero denominators and do not mutate their input, a missing artifact
raises an actionable error.

**API** — health, single predict, batch predict (including empty and oversized
rejection), `/model`, request-ID headers, internal consistency, and rejection of
malformed, missing, negative and injected-derived-feature input.

---

## 14. Training

```bash
python src/train.py
```

Loads data, builds features, splits, runs a 5-fold cross-validated grid search,
calibrates, sweeps the cost scenarios to pick an operating threshold, evaluates
against the baseline, computes permutation importances, and writes both
`churn_model.joblib` and `model_card.json`.

The script asserts no target-derived column reaches the feature matrix, and
exits with an actionable message if the input CSVs are missing.

---

## 15. Local Prediction

```bash
python src/predict.py
```

---

## 16. FastAPI

```bash
python -m uvicorn src.api:app --reload
```

Interactive documentation: `http://127.0.0.1:8000/docs`

Every response carries an `X-Request-ID` header, and each request is logged with
its method, path, status and duration.

### `GET /health`

```json
{ "status": "healthy", "model_loaded": true }
```

If the artifact is missing or unreadable the service still starts but returns
`503` with `"status": "unhealthy"` and a message explaining how to build it. The
container becomes visibly unhealthy for the correct reason instead of
crash-looping on an import-time traceback.

### `GET /model`

Serves `model_card.json`: hyperparameters, calibration method, operating
threshold, cost model, metrics, reliability table and permutation importances.

### `POST /predict`

```json
{
  "age": 35, "country": "DE",
  "total_orders": 2, "total_spent": 150,
  "days_since_last_order": 75, "has_previous_order": 1,
  "total_events": 5, "add_to_cart_count": 1, "checkout_count": 0,
  "login_count": 2, "product_view_count": 2,
  "tenure_days": 180, "events_last_30_days": 0
}
```

Response:

```json
{ "churn_probability": 0.6773, "predicted_churn": 0, "risk_level": "LOW" }
```

### `POST /predict/batch`

Same schema wrapped in `{"customers": [...]}`, 1 to 1000 per call. Returns
`{"predictions": [...], "count": n}`.

**Note on `days_since_last_order`:** for a customer who has never ordered, pass
`tenure_days`. That is how the training data encodes it; passing `0` would
describe a customer who ordered today.

---

## 17. Docker

Train the model first — the image copies `models/` at build time.

```bash
python src/train.py
docker build -t customer-churn:latest .
docker run -d --name customer-churn-api -p 8000:8000 customer-churn:latest
```

The image runs as a non-root user (`appuser`, uid 10001) and includes a health
check against `/health`. Test dependencies are not installed in the runtime
image.

---

## 18. Continuous Integration

`.github/workflows/ci.yml` runs on every push and pull request:

1. **test** — install, generate data, train, run the full suite, verify the
   model card is valid JSON, upload it as a build artifact.
2. **docker** — build the image, start the container, poll `/health` until
   ready, smoke-test `/health` and `/predict`, dump container logs.

The training step is what makes the run meaningful: without a model artifact the
model-dependent tests would skip rather than run.

---

## 19. AWS Deployment

The containerized API was deployed to an Amazon EC2 instance running the same
image tested locally, verified through `GET /health` and `POST /predict`.

```text
Internet --> :8000 --> EC2 --> Docker --> FastAPI --> Model
```

---

## 20. Security Considerations

A portfolio demonstration, not a hardened service.

- SSH restricted to the administrator's IP; `.pem` files excluded from git
- Container runs as a non-root user
- Model artifact and generated data excluded from git
- Request logging with correlation IDs

Production would additionally need HTTPS, a reverse proxy, authentication and
authorization, restricted network access, secrets management, versioned model
storage, monitoring, and rate limiting.

---

## 21. Limitations

**Synthetic data.** The relationship between history and future behaviour is one
the generator was written to contain. The measured ceiling of 0.7575 is a
property of that design, not evidence about real customers. The *method* of
measuring a ceiling transfers; the number does not.

**Model performance.** ROC-AUC 0.7231 against a 0.7575 ceiling. Useful ranking,
and close to the limit of this target definition.

**Threshold economics are illustrative.** The margin, acceptance rate and offer
costs are invented. The sensitivity table shows how much the answer depends on
them, which is the point.

**Random split, not temporal.** Train/validation/test are split randomly at a
single prediction date. Production would validate across multiple prediction
dates in time order.

**Single prediction date.** The model has never been tested for stability across
different times of year.

**Calibration is measured on 623 validation rows.** The reliability table's
lowest bin holds 2 customers, so the fit at the extremes is not well determined.

---

## 22. Future Improvements

Ordered by expected value, informed by
[section 10](#10-how-good-can-this-model-possibly-get):

1. **A better-framed target** — longer window, or sustained disengagement rather
   than a single 30-day gap. This raises the ceiling; nothing else does.
2. **Time-based cross-validation** across multiple prediction dates.
3. **Drift monitoring** on inputs and predictions, with automated retraining.
4. **Uplift modelling** — target customers whose behaviour the intervention
   would *change*, rather than those most likely to churn.
5. Cloud model registry with versioning and rollback.
6. HTTPS, authentication, rate limiting.
7. Production database integration in place of CSVs.

---

## 23. Technologies

Python, pandas, NumPy, scikit-learn, FastAPI, Uvicorn, pytest, Docker, GitHub
Actions, AWS EC2, Git / GitHub.
