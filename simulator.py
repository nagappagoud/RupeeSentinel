import pandas as pd
print("===== CASHFLOW AI: PURCHASE SIMULATOR =====")

# Get the user's current bank balance
balance = float(input("Enter your current bank balance: ₹"))

# Get the purchase amount
purchase = float(input("Enter the price of the item: ₹"))

# Calculate the remaining balance
remaining_balance = balance - purchase

print("\n===== SIMULATION RESULT =====")
print("Current Balance: ₹", balance)
print("Purchase Amount: ₹", purchase)
print("Balance After Purchase: ₹", remaining_balance)

# Set minimum safety balance
minimum_balance = 5000

print("\n===== FINANCIAL RISK ANALYSIS =====")

if remaining_balance < 0:
    print("CRITICAL WARNING: You cannot afford this purchase!")
    
elif remaining_balance < minimum_balance:
    print("WARNING: Your balance will fall below ₹5,000.")
    
else:
    print("Purchase looks affordable.")
    print("Your balance will remain above the safety limit.")

# ===== 30-DAY FUTURE RISK ANALYSIS =====

df = pd.read_csv("transactions.csv")
df["date"] = pd.to_datetime(df["date"])

# Detect recurring monthly transactions
recurring = []

for description, group in df.groupby("description"):

    group = group.sort_values("date")

    if len(group) < 2:
        continue

    dates = group["date"].diff().dropna().dt.days
    amounts = group["amount"].abs()

    if (
        dates.between(25, 35).all()
        and amounts.max() - amounts.min() <= amounts.mean() * 0.10
    ):
        recurring.append(description)

# Calculate average daily variable spending
recurring_df = df[df["description"].isin(recurring)]
variable_df = df[~df["description"].isin(recurring)]

history_days = (
    df["date"].max() - df["date"].min()
).days + 1

variable_expenses = abs(
    variable_df[variable_df["amount"] < 0]["amount"].sum()
)

daily_spending = variable_expenses / history_days

# Start future prediction after the purchase
future_balance = remaining_balance

future_dates = pd.date_range(
    start=df["date"].max() + pd.Timedelta(days=1),
    periods=30
)

lowest_balance = future_balance

for day in future_dates:

    future_balance -= daily_spending

    # Apply expected recurring transactions
    for description in recurring:

        group = recurring_df[
            recurring_df["description"] == description
        ]

        last_transaction = group.sort_values("date").iloc[-1]

        next_date = (
            last_transaction["date"] + pd.DateOffset(months=1)
        )

        if day.date() == next_date.date():
            future_balance += last_transaction["amount"]

    lowest_balance = min(lowest_balance, future_balance)

print("\n===== 30-DAY FUTURE RISK ANALYSIS =====")
print("Lowest predicted balance: ₹", round(lowest_balance, 2))

if lowest_balance < 0:
    print("CRITICAL: Your balance may become negative!")
elif lowest_balance < 5000:
    print("WARNING: Your balance may fall below ₹5,000!")
else:
    print("Your projected balance stays above ₹5,000.")
