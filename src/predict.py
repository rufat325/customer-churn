"""Command-line demo: score one example customer.

Run from the project root:

    python src/predict.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scoring import ModelNotAvailable, load_model, score


EXAMPLE_CUSTOMER = {
    "age": 35,
    "country": "DE",
    "total_orders": 2,
    "total_spent": 150.00,
    "days_since_last_order": 75,
    "has_previous_order": 1,
    "total_events": 5,
    "add_to_cart_count": 1,
    "checkout_count": 0,
    "login_count": 2,
    "product_view_count": 2,
    "tenure_days": 180,
    "events_last_30_days": 0,
}


def main() -> int:
    try:
        model = load_model()
    except ModelNotAvailable as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    result = score(model, EXAMPLE_CUSTOMER)

    print("CUSTOMER CHURN PREDICTION")
    print("--------------------------")
    print(f"Churn probability: {result['churn_probability']:.2%}")
    print(f"Predicted churn:   {result['predicted_churn']}")
    print(f"Risk level:        {result['risk_level']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
