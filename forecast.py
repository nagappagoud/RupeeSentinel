
import pandas as pd

df = pd.read_csv("transactions.csv")
df["date"] = pd.to_datetime(df["date"])

balance = float(input("Enter current bank balance: ₹"))

# STEP 1: Detect recurring transactions
recurring = []

for description, group in df.groupby("description"):

    group = group.sort_values("date")

    if len(group) < 2:
        continue

    dates = group["date"].diff().dropna().dt.days

    amounts = group["amount"].abs()

    amount_change = amounts.max() - amounts.min()
    amount_limit = amounts.mean() * 0.10

    # Require monthly intervals and stable amounts
    if (
        dates.between(25, 35).all()
        and amount_change <= amount_limit
    ):
        recurring.append(description)

print("\nDetected recurring transactions:")
print(recurring)

# STEP 2: Separate recurring and variable transactions
recurring_df = df[df["description"].isin(recurring)]

variable_df = df[~df["description"].isin(recurring)]

# STEP 3: Calculate average daily variable spending
start_date = df["date"].min()
last_date = df["date"].max()

history_days = (last_date - start_date).days + 1

variable_expenses = abs(
    variable_df[variable_df["amount"] < 0]["amount"].sum()
)

daily_spending = variable_expenses / history_days

print("\nAverage daily variable spending:", round(daily_spending, 2))

# STEP 4: Forecast next 30 days
forecast_start = last_date + pd.Timedelta(days=1)

future_dates = pd.date_range(
    start=forecast_start,
    periods=30
)

results = []

for day in future_dates:

    # Subtract estimated daily spending
    balance -= daily_spending

    # Apply expected recurring payments/income
    for description in recurring:

        group = recurring_df[
            recurring_df["description"] == description
        ]

        last_transaction = group.sort_values("date").iloc[-1]

        next_date = (
            last_transaction["date"] + pd.DateOffset(months=1)
        )

        if day.date() == next_date.date():

            balance += last_transaction["amount"]

    results.append({
        "Date": day.date(),
        "Predicted Balance": round(balance, 2)
    })

# STEP 5: Display results
forecast = pd.DataFrame(results)

print("\n===== CASHFLOW AI PREDICTION =====")
print(forecast.to_string(index=False))

# STEP 6: Identify financial risk
minimum_balance = forecast["Predicted Balance"].min()

print("\nLowest predicted balance:", round(minimum_balance, 2))

if minimum_balance < 5000:
    print("WARNING: Your balance may fall below ₹5,000!")
else:
    print("Your projected balance stays above ₹5,000.")
