import numpy as np
import joblib

from sklearn.ensemble import IsolationForest

# Reproducible synthetic training data
rng = np.random.default_rng(42)

# Typical spending assumptions in INR.
# These are illustrative assumptions, NOT real customer statistics.
CATEGORY_PROFILES = {
    "Groceries": (1400, 0.45),
    "Shopping": (1500, 0.65),
    "Food & Dining": (550, 0.50),
    "Transport": (700, 0.90),
    "Utilities": (1200, 0.25),
    "Entertainment": (350, 0.45),
    "Housing": (10000, 0.20),
    "Healthcare": (500, 0.65),
    "Other": (700, 0.60)
}

models = {}

# Train one anomaly detector per category
for category, (median_amount, variation) in CATEGORY_PROFILES.items():

    # Generate illustrative normal spending examples
    amounts = rng.lognormal(
        mean=np.log(median_amount),
        sigma=variation,
        size=1000
    )

    # Isolation Forest expects a 2D input
    X = np.log1p(amounts).reshape(-1, 1)

    model = IsolationForest(
        n_estimators=100,
        contamination=0.03,
        random_state=42
    )

    model.fit(X)

    models[category] = model

# Save all trained models
joblib.dump(
    {
        "models": models,
        "category_profiles": CATEGORY_PROFILES,
        "training_note": "Trained on illustrative synthetic spending data"
    },
    "anomaly_model.joblib"
)

print("ML training completed!")
print("Categories trained:", len(models))
print("Saved model: anomaly_model.joblib")


# Function to detect unusual spending
def detect_anomaly(category, amount):

    amount = abs(float(amount))

    if not np.isfinite(amount) or amount <= 0:
        raise ValueError("Transaction amount must be positive.")

    if category not in models:
        category = "Other"

    model = models[category]

    X = np.array([[np.log1p(amount)]])

    prediction = model.predict(X)[0]
    anomaly_score = -model.decision_function(X)[0]

    return {
        "category": category,
        "amount": amount,
        "prediction": "Potential Anomaly" if prediction == -1 else "Normal",
        "anomaly_score": round(float(anomaly_score), 4)
    }


# Test the trained model
print("\n===== MODEL TESTING =====")

test_transactions = [
    ("Groceries", 1500),
    ("Groceries", 20000),
    ("Food & Dining", 600),
    ("Food & Dining", 10000),
    ("Housing", 8000)
]

for category, amount in test_transactions:
    result = detect_anomaly(category, amount)
    print(result)