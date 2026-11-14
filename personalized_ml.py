import numpy as np
from sklearn.ensemble import IsolationForest


def train_personalized_models(df):

    models = {}

    # Use expense transactions only
    expenses = df[df["amount"] < 0].copy()

    # Convert expenses to positive values
    expenses["amount"] = expenses["amount"].abs()

    for category in expenses["category"].unique():

        category_data = expenses[
            expenses["category"] == category
        ]

        amounts = category_data["amount"].values

        # Require at least 10 transactions
        if len(amounts) < 10:
            continue

        X = np.log1p(amounts).reshape(-1, 1)

        model = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42
        )

        model.fit(X)

        models[category] = model

    return models