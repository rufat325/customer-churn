# Customer Churn Prediction API

An end-to-end machine learning project that predicts customer churn from
demographics, order history and website activity, serves it behind a
containerized REST API — and then asks the harder questions: how good could
this model possibly be, does it survive contact with time, and is predicting
churn even the right problem?

**Headline result:** test ROC-AUC **0.669 [0.626, 0.711]** against a 64.6%
base rate and a measured ceiling of **0.727**.

Three findings matter more than that number:

| Finding | Where |
| --- | --- |
| Reframing the target from a 30-day to a 90-day window raises the achievable ceiling by **+0.10 AUC** — six times the best feature-engineering gain | [§10](#10-the-ceiling-and-how-to-raise-it) |
| A regime change in the data leaves ROC-AUC almost untouched while calibration error hits **11 points** — monitoring discrimination alone would miss it entirely | [§11](#11-walk-forward-validation) |
| Targeting a retention campaign by churn probability performs **no better than random**; an uplift model captures 87% of the achievable gain | [§12](#12-uplift-who-should-we-actually-contact) |

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

Data and model artifacts are not tracked in git, so build them first:

```bash
python src/generate_data.py     # writes data/*.csv
python src/train.py             # writes models/churn_model.joblib + model_card.json
python -m pytest                # 73 passed
```

Then score a customer, or start the API:

```bash
python src/predict.py
python -m uvicorn src.api:app --reload
```

Tests that need the model artifact skip with an explanatory message rather than
failing if you have not run `train.py` yet.

---

## 2. What "churn" means here

Churn is **placing no order in the 30 days following the prediction date**.

This is a purchase-frequency definition, not a subscription cancellation, which
is why the base rate is 64.6% — most customers simply do not buy in a given
month. Read every metric with that in mind.

It is also, as [§10](#10-the-ceiling-and-how-to-raise-it) shows with numbers,
the single largest constraint on how good this model can be.

---

## 3. Data

`src/generate_data.py` produces four datasets spanning **2024-01-01 to
2026-03-31**: 5,000 customers, ~34,900 orders, ~89,900 website events, and a
randomized retention campaign.

| File | Contents |
| --- | --- |
| `customers.csv` | customer_id, age, country, signup_date |
| `orders.csv` | order_id, customer_id, order_date, amount |
| `website_events.csv` | event_id, customer_id, event_type, event_date |
| `campaign.csv` | customer_id, treated, ordered_in_campaign |

### Hidden traits

Every customer carries three traits that are dropped before the CSVs are
written, so a model must infer them from behaviour:

- **`activity_level`** — how often they order
- **`drift`** — how that propensity changes over time
- **`browse_bias`** — how much they browse relative to how much they buy

Keeping them out of the data is what makes the problem honest; being able to
recover them from the seeded generator is what makes the ceiling measurable.

### Behaviour is a rate process

Orders and events are an inhomogeneous Poisson process. For each customer, each
day of their lifetime:

```text
intensity = base_rate(activity_level) * exp(drift * years_since_signup)
```

Counts are Poisson, dates fall only inside a customer's own lifetime, and
exposure drives volume.

### Two corrections worth recording

**Dates used to precede signup.** An earlier generator drew a fixed global
number of orders, chose *who* placed them from the activity weights, then
assigned each a date drawn uniformly across the calendar — independently of
signup:

```text
orders placed before their customer signed up: 9,744 / 20,200 (48.2%)
events logged before their customer signed up: 24,842 / 50,000 (49.7%)
correlation(tenure_days, total_orders):        0.006
```

One customer signed up on 2025-12-31 with an order dated 2025-01-03. That
correlation is now **0.498**. The fix reversed a modelling conclusion: on the
incoherent data, exposure-normalised rate features made the model *worse*,
which looked like a fact about rate features and was actually a symptom of the
bug.

**History and the future used to obey different laws.** History was generated
with `activity ** 2` and the target window with `activity ** 3` — a regime
change at the prediction date. It made walk-forward validation meaningless,
since each prediction date would be predicting a different process. One
stationary process per customer, modulated by personal drift, replaces it.

Both properties are asserted at generation time and pinned by tests.

---

## 4. Prediction methodology

The prediction date is `2025-12-31`, the last day of observed history,
inclusive. The timeline partitions into two adjacent windows:

```text
        history window                    target window
  <------------------------->   <--------------------------->
  2024-01-01 ... 2025-12-31     2026-01-01 ... 2026-01-30
       (features)                       (label only)
                            ^
                     prediction date
```

Both boundaries derive from the same anchor, so no order can fall in both
(leaking the target) or in neither (silently discarding data). A test sweeps
every offset from −2 to +32 days to enforce it.

Customers must have signed up at least 30 days before the prediction date:
**4,418** of 5,000 eligible, churn rate **0.6464**.

---

## 5. Feature engineering

**Supplied by the caller (13):** `age`, `country`, `tenure_days`,
`total_orders`, `total_spent`, `days_since_last_order`, `has_previous_order`,
`total_events`, `add_to_cart_count`, `checkout_count`, `login_count`,
`product_view_count`, `events_last_30_days`.

**Derived inside the pipeline (6):** `orders_per_day`, `events_per_day`,
`spend_per_order`, `checkout_rate`, `cart_rate`, `recency_ratio`.

Derived features are computed by a `FunctionTransformer` as the first pipeline
step, not requested from the caller. They are deterministic functions of the
raw columns, so asking a client to supply them would only create an opportunity
to disagree with training. The API contract stays at 13 fields and train/serve
skew is structurally impossible.

Raw counts conflate propensity with exposure; dividing by tenure separates
them. `tenure_days` predicts almost nothing on its own — signup dates are
independent of activity — but it is the denominator that makes the rates work,
which is why it ranks second in permutation importance.

**Missing values.** Counts fill with `0`. `days_since_last_order` is undefined
rather than missing-at-random for customers who never ordered, so it fills with
`tenure_days`: it has been their entire lifetime since they last ordered.

---

## 6. Model

Gradient Boosting, selected over Logistic Regression and Random Forest in
`notebooks/01_data_profiling.ipynb`.

```text
learning_rate = 0.05    max_depth = 2
n_estimators  = 100     min_samples_leaf = 10
```

5-fold CV ROC-AUC on the training split: **0.6923**

### Calibration

Gradient boosting optimises a ranking loss, so raw scores are not
probabilities. Isotonic and sigmoid are both fitted with internal CV on the
training split — leaving validation clean for threshold selection — and the
better is chosen on validation Brier score. Isotonic wins.

| Bin | n | Predicted | Observed |
| --- | ---: | ---: | ---: |
| 0.2–0.4 | 66 | 0.332 | 0.348 |
| 0.4–0.6 | 155 | 0.494 | 0.516 |
| 0.6–0.8 | 288 | 0.702 | 0.674 |
| 0.8–1.0 | 150 | 0.859 | 0.867 |

---

## 7. Results, with error bars

| | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline (predict churn for all) | 0.6456 | 0.6456 | 1.0000 | 0.7846 | n/a | n/a |
| **Model** @ threshold 0.80 | 0.4992 | 0.8333 | 0.2804 | 0.4196 | **0.6689** | 0.2116 |

A point estimate from 663 test rows is not a four-significant-figure number:

| Score | ROC-AUC | 95% CI |
| --- | ---: | --- |
| Model | 0.6689 | [0.6255, 0.7109] |
| Recency heuristic (`days_since_last_order`) | 0.5939 | [0.5494, 0.6375] |
| **Difference (paired)** | **+0.0750** | **[0.0366, 0.1146]** |

The model beats the obvious business rule in **100% of bootstrap resamples**.
That comparison is paired — both scores are computed on identical rows — which
matters, because checking whether two independent intervals overlap is a
different and much weaker question. Two models can have heavily overlapping
intervals while one wins essentially always, since their errors are correlated.
Overlap is not a significance test. `src/evaluation.py` implements both, and
`tests/test_evaluation.py` includes a case where the naive comparison fails and
the paired test resolves it.

### Reading the thresholded numbers

At the operating threshold the model has **lower accuracy and F1 than the
majority-class baseline**. That is expected, not hidden: accuracy is a poor
metric at a 65% base rate, and the threshold is tuned for expected value, not
accuracy. What it buys is precision — **0.833 against 0.646** — and the ranking
underneath.

### Permutation importance (validation, drop in ROC-AUC)

| Feature | Δ ROC-AUC |
| --- | ---: |
| `total_orders` | 0.1193 ± 0.0137 |
| `tenure_days` | 0.0564 ± 0.0071 |
| `total_events` | 0.0237 ± 0.0073 |
| `days_since_last_order` | 0.0038 ± 0.0051 |

---

## 8. Choosing the threshold from costs

Contacting a customer costs `offer_cost`; if they were churning and accept, it
earns `margin` with probability `accept_rate`. The optimal threshold depends
entirely on those numbers, so the project sweeps them rather than quoting one.
Chosen on validation, valued on the 663-customer test split:

| Scenario | TP | FP | Threshold | Contact all | Targeted | Uplift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cheap offer | +10 | −5 | 0.25 | 3,105 | 3,120 | +15 |
| **standard offer** (shipped) | +3 | −12 | **0.80** | −1,536 | **72** | **+1,608** |
| premium offer | −5 | −20 | 0.96 | −6,840 | 0 | +6,840 |

Three regimes. With a cheap offer, blanket contact is already profitable and
the model adds almost nothing — **a model is not always the answer**. With a
standard offer it turns a loss-making campaign profitable. With a premium offer
no contact is ever worthwhile and the optimiser correctly selects nobody.

The shipped threshold is persisted in `models/model_card.json`, which the API
reads at startup and serves at `GET /model`. Changing the economics changes the
threshold without retraining.

Risk bands anchor to that threshold rather than fixed cut points: `LOW` below
it, the region above split into `MEDIUM` and `HIGH`. Fixed bands at 0.50/0.75
left `MEDIUM` unreachable once the tuned threshold rose above 0.75.

---

## 9. Notebooks

| Notebook | Question |
| --- | --- |
| [`01_data_profiling`](notebooks/01_data_profiling.ipynb) | Original exploration and model selection *(predates the generator rebuild; kept as the record)* |
| [`02_headroom_and_target`](notebooks/02_headroom_and_target.ipynb) | How good can this get, and what raises the ceiling? |
| [`03_temporal_validation`](notebooks/03_temporal_validation.ipynb) | Does it survive contact with time? |
| [`04_uplift_modelling`](notebooks/04_uplift_modelling.ipynb) | Is predicting churn even the right problem? |

---

## 10. The ceiling, and how to raise it

Full analysis: [`notebooks/02_headroom_and_target.ipynb`](notebooks/02_headroom_and_target.ipynb)

Churn is "no order in the window", so if the true expected order count is `L`,
the true churn probability is `exp(-L)`. That is monotone in `L`, so ranking by
`-L` is Bayes-optimal and its ROC-AUC is the **ceiling** for any model of this
target. `expected_orders()` computes `L` exactly from the hidden traits.

At the shipped 30-day window:

| Feature set | Test ROC-AUC |
| --- | ---: |
| Raw counts only | 0.6556 [0.6132, 0.6992] |
| Shipped (raw + derived rates) | 0.6725 [0.6290, 0.7146] |
| **ORACLE (true expected orders)** | **0.7274 [0.6869, 0.7669]** |

Those intervals overlap heavily, yet the paired comparison is unambiguous: the
oracle beats the shipped model by +0.0550 [0.0254, 0.0850], winning **100% of
resamples**. Overlap is not a significance test.

### The target is the constraint

Mean expected orders per customer in a 30-day window is well under one. The
label is mostly Poisson noise: even knowing a customer's rate perfectly,
whether they *happen* to order in those 30 days is close to a coin flip.

Widening the window changes that:

| Window | Churn rate | Model AUC | **Ceiling** | Headroom |
| ---: | ---: | ---: | ---: | ---: |
| 30 days | 0.646 | 0.6725 | **0.7274** | +0.055 |
| 60 days | 0.456 | 0.7006 | **0.7690** | +0.068 |
| 90 days | 0.335 | 0.7306 | **0.8293** | +0.099 |

**Reframing the target is worth several times what feature engineering was
worth.** 30 → 90 days raises the ceiling by **+0.102 AUC**. The best feature
change in this project — adding the derived rate features — was worth **+0.017**
(0.6556 → 0.6725), so the target reframing is roughly six times larger, and it
is available without touching the model at all.

Note the headroom *widens* with the window. A longer target does not merely
make the problem easier — it creates signal the current features do not yet
exploit, so feature work becomes worthwhile again *after* the target is fixed.
Order matters.

The catch: a 90-day window answers a slower business question, and you wait 90
days to learn whether a prediction was right. Whether that trade is worth making
is a product decision. This analysis supplies the price.

---

## 11. Walk-forward validation

Full analysis: [`notebooks/03_temporal_validation.ipynb`](notebooks/03_temporal_validation.ipynb)

Headline metrics come from a random split at one prediction date. That does not
answer whether a model trained in January still works in October. Scoring at
five dates a quarter apart, retrained each time, the problem is **stable** —
estimates wander a few points and every interval overlaps every other.

Trained once and left alone, a **twelve-month-old model loses about 0.004
AUC**. That is a real property here, not a bug: each customer's drift is a
fixed personal trait, so the mapping from history to future behaviour never
changes.

Which leaves the harness untested. So the generator can inject a
population-wide shock — every customer's order rate halved from day 550, a
macro event or a competitor launch:

| Metric | Before shock | After shock |
| --- | ---: | ---: |
| ROC-AUC | 0.673 | 0.654 |
| Brier | 0.206 | **0.167** (improved!) |
| Predicted churn rate | 0.668 | 0.712 |
| Actual churn rate | 0.662 | 0.785 |
| **Calibration error** | +0.006 | **−0.112** |

**ROC-AUC barely notices.** It measures *ordering*, and halving everyone's rate
leaves the ranking nearly intact while making every probability wrong in
absolute terms. A monitor watching discrimination alone would have raised
nothing while the model was mis-stating churn by eleven points — and any
decision built on a probability, including this project's cost-based threshold,
is now mis-set.

Worse, **Brier improved**. It is a proper scoring rule but sensitive to the base
rate, and a base rate moving toward an extreme makes it easier to score well.
Watched alone it would have suggested the model got *better*.

The cheap signal that caught this: predicted versus actual base rate. Monitor
calibration, not just discrimination.

---

## 12. Uplift: who should we actually contact?

Full analysis: [`notebooks/04_uplift_modelling.ipynb`](notebooks/04_uplift_modelling.ipynb)

A churn model ranks customers by how likely they are to leave. A retention
campaign then contacts the top of that list. That step is close to worthless.

The campaign should reach customers whose behaviour the contact *changes*:

| | Contacted | Not contacted |
| --- | --- | --- |
| **Sure thing** | stays | stays |
| **Persuadable** | stays | leaves |
| **Lost cause** | leaves | leaves |
| **Sleeping dog** | leaves | stays |

Only persuadables repay the spend; sleeping dogs are actively harmed. A churn
model ranks by outcome, not responsiveness, and cannot tell them apart.
Separating them needs randomised data, because responsiveness is causal —
hence `campaign.csv`, a simulated RCT run after the observational data ends
(2,503 treated / 2,497 control, ATE **+0.0455**, and 15% of customers have
negative true uplift).

A T-learner — one outcome model per arm, uplift as their difference:

| Targeting strategy | Qini score |
| --- | ---: |
| **Uplift model (T-learner)** | **12.73** |
| Churn probability | 1.44 |
| Random | 1.98 |
| ORACLE (true uplift) | 14.62 |

**Targeting by churn probability scores no better than random.** The model this
project spent most of its effort on is, as a targeting rule, worthless — while
the uplift model captures 87% of what the oracle could achieve.

Observed uplift by quintile shows why:

| Quintile | Uplift model | Churn probability |
| ---: | ---: | ---: |
| 1 (top) | **+0.125** | +0.010 |
| 2 | +0.097 | +0.092 |
| 3 | +0.045 | +0.100 |
| 4 | −0.024 | −0.018 |
| 5 | −0.000 | +0.080 |

The uplift model's quintiles decline cleanly and go negative — it found the
sleeping dogs. The churn model's are not ordered at all. Correlation between
the two scores is **−0.065**: in this data persuadability is driven by browsing
relative to buying, while churn risk is driven by order frequency.
Interested-but-hesitant customers respond to a nudge; customers who simply do
not buy much are not persuadable, just quiet.

Three consequences: **run the experiment** (uplift cannot be estimated from
observational data at all); **a churn model is still useful** for forecasting
revenue at risk and for triage, just not for deciding who to contact; and
**measure campaigns against a holdout**, since a campaign targeting sure things
shows excellent retention and achieves nothing.

---

## 13. Project structure

```text
customer-churn/
+-- .github/workflows/ci.yml   test + docker smoke test on every push
+-- data/                      generated CSVs (not tracked)
+-- models/
|   +-- churn_model.joblib     calibrated pipeline (not tracked)
|   +-- model_card.json        params, metrics, CIs, threshold, importances
+-- notebooks/                 01 profiling, 02 ceiling, 03 time, 04 uplift
+-- src/
|   +-- generate_data.py       synthetic data, hidden traits, RCT, shocks
|   +-- features.py            feature engineering + target construction
|   +-- train.py               training, calibration, thresholds, uncertainty
|   +-- scoring.py             shared contract: paths, features, thresholds
|   +-- evaluation.py          bootstrap and paired-bootstrap intervals
|   +-- uplift.py              T-learner, Qini curves, decile diagnostics
|   +-- predict.py             CLI demo
|   +-- api.py                 FastAPI service
+-- tests/                     73 tests
+-- Dockerfile
+-- requirements.txt / requirements-dev.txt
```

---

## 14. Testing

```bash
python -m pytest
```

Expected: `73 passed`.

**Generator** — nothing precedes signup; tenure and volume are positively
related (regression test for the incoherence bug); the oracle intensity
reproduces realised volume within 15%; the campaign is balanced, shows a
positive ATE, contains sleeping dogs, and its uplift is uncorrelated with churn
risk; shocks affect only days after them.

**Features** — expected columns, binary labels, unique rows, sentinel
imputation, four boundary cases plus a −2…+32 day sweep asserting the windows
never overlap and never gap.

**Evaluation** — intervals bracket point estimates and narrow with more data;
paired tests are directional, detect real differences, and resolve a case where
comparing overlapping intervals fails.

**Uplift** — the T-learner recovers the uplift ordering; Qini ranks strategies
correctly; random targeting centres on zero across 40 draws while a genuine
ranking beats every draw.

**Model / API** — artifact loading, batch/single agreement, risk bands
reachable and never contradicting predictions at any threshold, and rejection
of malformed, missing, negative and injected input.

---

## 15. API

```bash
python -m uvicorn src.api:app --reload
```

Docs at `http://127.0.0.1:8000/docs`. Every response carries an `X-Request-ID`;
each request is logged with method, path, status and duration.

- **`GET /health`** — `{"status": "healthy", "model_loaded": true}`. If the
  artifact is missing the service still starts but returns `503`, so the
  container is visibly unhealthy for the right reason rather than
  crash-looping.
- **`GET /model`** — serves the model card.
- **`POST /predict`** — 13 fields in, `{churn_probability, predicted_churn,
  risk_level}` out.
- **`POST /predict/batch`** — same schema in `{"customers": [...]}`, 1–1000 per
  call.

For a customer who has never ordered, pass `tenure_days` as
`days_since_last_order` — that is how the training data encodes it.

---

## 16. Docker and CI

```bash
python src/train.py
docker build -t customer-churn:latest .
docker run -d --name customer-churn-api -p 8000:8000 customer-churn:latest
```

Runs as non-root (`appuser`, uid 10001), with a health check against `/health`.
Test dependencies are not installed in the runtime image.

`.github/workflows/ci.yml` runs on every push: install → generate → train → run
the suite → validate the model card, then a second job builds the image, starts
the container, polls `/health` and smoke-tests `/predict`.

---

## 17. Limitations

**Synthetic data.** The relationship between history and future is one the
generator was written to contain. The *method* of measuring a ceiling
transfers; the number does not.

**Small test splits.** 663 rows gives a ±0.04 interval on AUC. Every headline
number carries one for that reason.

**Threshold economics are invented.** The sensitivity table shows how much the
answer depends on them, which is the point.

**Uplift is estimated from one simulated campaign** with a deterministic link
from `browse_bias` to true uplift. Real uplift is noisier and the achievable
Spearman correlation would be lower than the 0.28 seen here.

**Stationary process.** Per-customer drift is a fixed trait, so nothing decays
without an injected shock. Real populations shift in ways this does not model.

**No seasonality**, and a single geography-agnostic model.

---

## 18. Future work

1. **Adopt a longer target window** — the only change that raises the ceiling,
   worth ~+0.10 AUC.
2. **Ship the uplift model as the targeting rule**, keeping the churn model for
   forecasting.
3. **Monitor calibration in production**, not just discrimination — §11 shows
   why.
4. Sequence/recency models (e.g. BG/NBD) that model the Poisson process
   directly rather than approximating it with tabular aggregates.
5. Model registry with versioning and rollback; HTTPS, authentication, rate
   limiting; production database in place of CSVs.

---

## 19. Technologies

Python, pandas, NumPy, scikit-learn, FastAPI, Uvicorn, pytest, Docker, GitHub
Actions, AWS EC2, Git / GitHub.
