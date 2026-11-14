
import pandas as pd

# Load transaction history
df = pd.read_csv("transactions.csv")

# Convert expenses into positive values for comparison
df["spending"] = df["amount"].abs()

# Calculate normal spending for each description
normal_spending = (
    df[df["amount"] < 0]
    .groupby("description")["spending"]
    .median()
)

# Simulate an unusual transaction for demonstration
test_transaction = {
    "description": "Restaurant",
    "amount": -4500
}

description = test_transaction["description"]
amount = abs(test_transaction["amount"])

print("===== CASHFLOW AI: SPENDING ANALYSIS =====")

if description in normal_spending.index:

    average = normal_spending[description]

    print("\nTransaction:", description)
    print("Transaction amount: ₹", amount)
    print("Usual spending (median): ₹", average)

    if amount > average * 2.5:

        print("\nWARNING: Unusual spending detected!")

        print(
            "Explanation: This transaction is more than "
            "2.5 times your usual spending."
        )

    else:
        print("\nSpending looks normal.")

else:
    print("Not enough history for this transaction category.")
