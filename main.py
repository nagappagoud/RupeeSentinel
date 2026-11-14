
import pandas as pd

# Read transaction data
df = pd.read_csv("transactions.csv")

# Convert date column into date format
df["date"] = pd.to_datetime(df["date"])

# Calculate total income
income = df[df["amount"] > 0]["amount"].sum()

# Calculate total expenses
expenses = abs(df[df["amount"] < 0]["amount"].sum())

# Display results
print("===== CASHFLOW AI =====")

print("Total Income:", income)
print("Total Expenses:", expenses)

print("Net Cash Flow:", income - expenses)

print("\nTransaction History:")
print(df)
