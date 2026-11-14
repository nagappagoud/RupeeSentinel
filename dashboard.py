
from unicodedata import category

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import joblib
from pathlib import Path
from personalized_ml import train_personalized_models   
MODEL_PATH = Path(__file__).resolve().parent / "anomaly_model.joblib"

@st.cache_resource
def load_ml_model():
    return joblib.load(MODEL_PATH)

# Page configuration
st.set_page_config(
    page_title="RupeeSentinel",
    page_icon="💰",
    layout="wide"
)
# Dashboard heading
st.title("💰 RupeeSentinel")
st.subheader("Your Personal Financial Operating System")

st.divider()
# ===== TRANSACTION DATA UPLOAD =====

st.subheader("📂 Upload Your Transactions")

uploaded_file = st.file_uploader(
    "Upload your bank transaction CSV file",
    type=["csv"]
)
# ===== LOAD TRANSACTION DATA =====

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # Rename common column variations
    df = df.rename(columns={
        "transaction": "description",
        "narration": "description",
        "particulars": "description",
        "transaction date": "date",
        "transaction amount": "amount"
    })

    # Handle Debit and Credit columns
    if "amount" not in df.columns:

        if "debit" in df.columns and "credit" in df.columns:

            df["debit"] = pd.to_numeric(
                df["debit"].astype(str).str.replace(",", "", regex=False),
                errors="coerce"
            ).fillna(0)

            df["credit"] = pd.to_numeric(
                df["credit"].astype(str).str.replace(",", "", regex=False),
                errors="coerce"
            ).fillna(0)

            df["amount"] = df["credit"] - df["debit"]

        else:
            st.error("CSV must contain Amount or both Debit and Credit.")
            st.write("Columns found:", df.columns.tolist())
            st.stop()

    # Check required columns
    required_columns = ["date", "description", "amount"]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        st.error(f"Missing columns: {missing_columns}")
        st.write("Columns found:", df.columns.tolist())
        st.stop()

    df = df[required_columns].copy()

    # Clean data
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["amount"] = pd.to_numeric(
        df["amount"].astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )

    df["description"] = df["description"].fillna("").astype(str).str.strip()

    # Remove invalid rows
    df = df.dropna(subset=["date", "amount"])
    df = df[df["description"] != ""]

    # Remove exact duplicates
    original_count = len(df)
    df = df.drop_duplicates()

    duplicates_removed = original_count - len(df)

    if duplicates_removed > 0:
        st.warning(f"{duplicates_removed} duplicate transactions removed.")

    if df.empty:
        st.error("No valid transactions found in the uploaded CSV.")
        st.stop()

    st.success(f"Successfully uploaded {len(df)} transactions!")

else:

    df = pd.read_csv("transactions.csv")
    df["date"] = pd.to_datetime(df["date"])

    st.info("Currently displaying sample transaction data.")
# ===== AUTOMATIC TRANSACTION CATEGORIZATION =====

def categorize_transaction(description, amount):

    description = str(description).lower()

    if amount > 0:
        return "Income"

    elif any(word in description for word in [
        "grocery", "supermarket", "vegetables"
    ]):
        return "Groceries"

    elif any(word in description for word in [
        "amazon", "flipkart", "shopping"
    ]):
        return "Shopping"

    elif any(word in description for word in [
        "restaurant", "swiggy", "zomato", "food", "cafe"
    ]):
        return "Food & Dining"

    elif any(word in description for word in [
        "uber", "ola", "petrol", "fuel", "transport"
    ]):
        return "Transport"

    elif any(word in description for word in [
        "electricity", "water bill", "gas bill", "internet"
    ]):
        return "Utilities"

    elif any(word in description for word in [
        "netflix", "spotify", "prime", "movie"
    ]):
        return "Entertainment"

    elif any(word in description for word in [
        "rent", "mortgage"
    ]):
        return "Housing"

    elif any(word in description for word in [
        "hospital", "medical", "pharmacy", "medicine"
    ]):
        return "Healthcare"

    else:
        return "Other"


df["category"] = df.apply(
    lambda row: categorize_transaction(
        row["description"], row["amount"]
    ),
    axis=1
)
# ===== PERSONALIZED ML TRAINING =====

personalized_models = train_personalized_models(df)

if personalized_models:
    st.success(
        f"Personalized ML models trained for "
        f"{len(personalized_models)} categories!"
    )
else:
    st.info(
        "Not enough transaction history for personalized training. "
        "Using the existing synthetic models."
    )
personalized_models = train_personalized_models(df)
# ===== AUTOMATIC ML ANALYSIS =====

model_bundle = load_ml_model()
ml_models = model_bundle["models"]

def analyze_transaction(row):

    if row["amount"] >= 0:
        return pd.Series({
            "ml_prediction": "Income",
            "ml_anomaly_score": np.nan
        })

    amount = abs(float(row["amount"]))
    category = row["category"]

    if category not in ml_models:
        category = "Other"

    # Prefer personalized model when available
    if category in personalized_models:
        model = personalized_models[category]
    elif category in ml_models:
        model = ml_models[category]
    else:
        model = ml_models["Other"]

    X = np.array([[np.log1p(amount)]])

    prediction = model.predict(X)[0]
    score = -model.decision_function(X)[0]

    return pd.Series({
        "ml_prediction": (
            "Potential Anomaly" if prediction == -1 else "Normal"
        ),
        "ml_anomaly_score": round(float(score), 4)
    })


ml_results = df.apply(analyze_transaction, axis=1)

df[["ml_prediction", "ml_anomaly_score"]] = ml_results
# Display categorized transactions
st.subheader("Automatic Transaction Categories")
st.dataframe(df[["description", "amount", "category"]])

# Apply categorization to every transaction


# Calculate financial information
income = df[df["amount"] > 0]["amount"].sum()

expenses = abs(
    df[df["amount"] < 0]["amount"].sum()
)

net_cashflow = income - expenses



# Current balance input
balance = st.number_input(
    "Enter your current bank balance (₹)",
    min_value=0.0,
    value=18000.0,
    step=500.0
)

# Financial summary
st.subheader("Financial Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Current Balance", f"₹{balance:,.2f}")
col2.metric("Total Income", f"₹{income:,.2f}")
col3.metric("Total Expenses", f"₹{expenses:,.2f}")
col4.metric("Net Cash Flow", f"₹{net_cashflow:,.2f}")

st.divider()

# Cash flow chart
st.subheader("Transaction Cash Flow")

daily_flow = (
    df.groupby("date")["amount"]
    .sum()
    .reset_index()
)

daily_flow["Cumulative Cash Flow"] = (
    daily_flow["amount"].cumsum()
)

fig = px.line(
    daily_flow,
    x="date",
    y="Cumulative Cash Flow",
    markers=True,
    title="Cumulative Cash Flow Over Time"
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Amount (₹)",
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

# ===== INTERACTIVE TRANSACTION HISTORY =====

st.subheader("📋 Transaction History")

# Category filter
category_options = ["All Categories"] + sorted(
    df["category"].unique().tolist()
)

selected_filter = st.selectbox(
    "Filter transactions by category",
    category_options
)

# Search filter
search_text = st.text_input(
    "Search transaction description"
)

# Apply filters
filtered_df = df.copy()

if selected_filter != "All Categories":
    filtered_df = filtered_df[
        filtered_df["category"] == selected_filter
    ]

if search_text:
    filtered_df = filtered_df[
        filtered_df["description"].str.contains(
            search_text,
            case=False,
            na=False
        )
    ]

# Display filtered transactions
# Format date for display
display_df = filtered_df.copy()

display_df["date"] = display_df["date"].dt.strftime("%d-%m-%Y")

# Display filtered transactions
st.dataframe(
    display_df.sort_values("date", ascending=False),
    use_container_width=True,
    hide_index=True
)

st.caption(
    f"Showing {len(filtered_df)} of {len(df)} transactions"
)

# ===== 30-DAY CASH FLOW FORECAST =====

st.divider()
st.header("🔮 30-Day Cash Flow Forecast")

# Detect recurring transactions
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

recurring_df = df[df["description"].isin(recurring)]
variable_df = df[~df["description"].isin(recurring)]

# Calculate average daily variable spending
history_days = (
    df["date"].max() - df["date"].min()
).days + 1

variable_expenses = abs(
    variable_df[variable_df["amount"] < 0]["amount"].sum()
)

daily_spending = variable_expenses / history_days
# ===== FORECAST DEBUGGING =====

st.subheader("Forecast Debugging")

st.write("Starting balance:", balance)
st.write("Daily variable spending:", daily_spending)

st.write("Recurring transactions:")
st.dataframe(recurring_df)

st.write("Variable transactions:")
st.dataframe(variable_df)

# Generate 30-day prediction
future_balance = balance
forecast_results = []

future_dates = pd.date_range(
    start=df["date"].max() + pd.Timedelta(days=1),
    periods=30
)

for day in future_dates:

    future_balance -= daily_spending

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

    forecast_results.append({
        "Date": day,
        "Predicted Balance": future_balance
    })

forecast_df = pd.DataFrame(forecast_results)

# Display forecast chart
fig_forecast = px.line(
    forecast_df,
    x="Date",
    y="Predicted Balance",
    markers=True,
    title="Predicted Bank Balance - Next 30 Days"
)

fig_forecast.add_hline(
    y=5000,
    line_dash="dash",
    annotation_text="Safety Balance ₹5,000"
)

fig_forecast.update_layout(
    template="plotly_dark",
    xaxis_title="Date",
    yaxis_title="Predicted Balance (₹)"
)

st.plotly_chart(fig_forecast, use_container_width=True)

# Display financial risk
lowest_balance = forecast_df["Predicted Balance"].min()

st.metric(
    "Lowest Predicted Balance",
    f"₹{lowest_balance:,.2f}"
)

if lowest_balance < 0:
    st.error("Critical: Your predicted balance may become negative!")

elif lowest_balance < 5000:
    st.warning("Warning: Your predicted balance may fall below ₹5,000!")

else:
    st.success("Your predicted balance stays above ₹5,000.")


# ===== AUTOMATIC ML ANALYSIS =====

model_bundle = load_ml_model()
ml_models = model_bundle["models"]

def analyze_transaction(row):

    if row["amount"] >= 0:
        return pd.Series({
            "ml_prediction": "Income",
            "ml_anomaly_score": np.nan
        })

    amount = abs(float(row["amount"]))
    category = row["category"]

    if category not in ml_models:
        category = "Other"

    model = ml_models[category]

    X = np.array([[np.log1p(amount)]])

    prediction = model.predict(X)[0]
    score = -model.decision_function(X)[0]

    return pd.Series({
        "ml_prediction": (
            "Potential Anomaly" if prediction == -1 else "Normal"
        ),
        "ml_anomaly_score": round(float(score), 4)
    })


ml_results = df.apply(analyze_transaction, axis=1)

df[["ml_prediction", "ml_anomaly_score"]] = ml_results
# ===== AUTOMATIC ML ALERTS =====

st.divider()
st.subheader("🤖 Automatic ML Transaction Analysis")

anomalies = df[
    df["ml_prediction"] == "Potential Anomaly"
]

col1, col2 = st.columns(2)

col1.metric("Transactions Analyzed", len(df))
col2.metric("Potential Anomalies", len(anomalies))

if not anomalies.empty:

    st.warning("🚨 Potentially unusual transactions detected!")

    st.dataframe(
        anomalies[
            [
                "date",
                "description",
                "amount",
                "category",
                "ml_anomaly_score"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

else:
    st.success("✅ No potential anomalies detected.")
# ===== PERSONALIZED SPENDING ANALYSIS =====

df = df.sort_values("date").reset_index(drop=True)

spending_history = {}
personalized_results = []

for _, row in df.iterrows():

    amount = abs(float(row["amount"]))
    category = row["category"]

    if row["amount"] >= 0:
        personalized_results.append({
            "personalized_status": "Income",
            "personalized_baseline": np.nan,
            "personalized_history_count": 0
        })
        continue

    previous_spending = spending_history.get(category, [])

    if len(previous_spending) > 0:

        baseline = float(np.median(previous_spending))

        if amount > baseline * 2.5:
            status = "Potentially Unusual"
        else:
            status = "Within Pattern"

    else:
        baseline = np.nan
        status = "Insufficient History"

    personalized_results.append({
        "personalized_status": status,
        "personalized_baseline": baseline,
        "personalized_history_count": len(previous_spending)
    })

    spending_history.setdefault(category, []).append(amount)

personalized_df = pd.DataFrame(personalized_results)

df[
    [
        "personalized_status",
        "personalized_baseline",
        "personalized_history_count"
    ]
] = personalized_df
# ===== PERSONALIZED SPENDING ALERTS =====

st.divider()
st.subheader("🧠 Personalized Spending Analysis")

unusual_spending = df[
    df["personalized_status"] == "Potentially Unusual"
]

col1, col2, col3 = st.columns(3)

col1.metric("Transactions Checked", len(df[df["amount"] < 0]))

col2.metric(
    "Unusual Spending",
    len(unusual_spending)
)

col3.metric(
    "Insufficient History",
    len(df[df["personalized_status"] == "Insufficient History"])
)

if not unusual_spending.empty:
    for _, row in unusual_spending.iterrows():

        baseline = row["personalized_baseline"]
        amount = abs(float(row["amount"]))

        ratio = amount / baseline if baseline > 0 else 0

        st.error(
            f"🚨 {row['description']}: ₹{amount:,.2f}"
        )

        st.write(
            f"**Why flagged?** This transaction is "
            f"{ratio:.2f} times your historical median "
            f"of ₹{baseline:,.2f}."
        )

        

    st.warning("Potentially unusual spending detected!")

    st.dataframe(
        unusual_spending[
            [
                "date",
                "description",
                "amount",
                "category",
                "personalized_baseline",
                "personalized_history_count"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

else:
    st.success("No potentially unusual spending detected.")

st.caption(
    "Comparisons use previous transactions in the same category. "
    "The threshold is 2.5 times the historical median. "
    "This is a spending alert, not a fraud prediction."
)
# ===== COMBINED ANOMALY SUMMARY =====

st.divider()
st.subheader("📊 Combined Anomaly Summary")

# Consider expense transactions only
expense_df = df[df["amount"] < 0].copy()

personalized_flag = (
    expense_df["personalized_status"] == "Potentially Unusual"
)

ml_flag = (
    expense_df["ml_prediction"] == "Potential Anomaly"
)

both_flagged = personalized_flag & ml_flag
personalized_only = personalized_flag & ~ml_flag
ml_only = ~personalized_flag & ml_flag

normal_both = (
    (expense_df["personalized_status"] == "Within Pattern")
    & (expense_df["ml_prediction"] == "Normal")
)

insufficient = ~(
    both_flagged | personalized_only |
    ml_only | normal_both
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Both Detected", int(both_flagged.sum()))
col2.metric("Personalized Only", int(personalized_only.sum()))
col3.metric("ML Only", int(ml_only.sum()))
col4.metric("Normal in Both", int(normal_both.sum()))

st.metric("Insufficient / Unclassified", int(insufficient.sum()))

# Create combined results
expense_df["combined_status"] = np.select(
    [
        both_flagged,
        personalized_only,
        ml_only,
        normal_both
    ],
    [
        "Detected by Both",
        "Personalized Only",
        "ML Only",
        "Normal in Both"
    ],
    default="Insufficient History"
)
# ===== ANOMALY COMPARISON CHART =====

chart_data = pd.DataFrame({
    "Detection Result": [
        "Detected by Both",
        "Personalized Only",
        "ML Only",
        "Normal in Both",
        "Insufficient History"
    ],
    "Transactions": [
        int(both_flagged.sum()),
        int(personalized_only.sum()),
        int(ml_only.sum()),
        int(normal_both.sum()),
        int(insufficient.sum())
    ]
})

fig = px.bar(
    chart_data,
    x="Detection Result",
    y="Transactions",
    color="Detection Result",
    title="Anomaly Detection Comparison",
    text="Transactions"
)

fig.update_layout(
    showlegend=False,
    xaxis_title="Detection Category",
    yaxis_title="Number of Transactions"
)

st.plotly_chart(fig, use_container_width=True)

st.dataframe(
    expense_df[
        [
            "date",
            "description",
            "amount",
            "category",
            "personalized_status",
            "ml_prediction",
            "combined_status"
        ]
    ],
    use_container_width=True,
    hide_index=True
)
st.dataframe(
    expense_df[
        [
            "date",
            "description",
            "amount",
            "category",
            "personalized_status",
            "ml_prediction",
            "combined_status"
        ]
    ],
    use_container_width=True,
    hide_index=True
)
# ===== PURCHASE SIMULATOR =====

st.divider()
st.header("🛒 What-If Purchase Simulator")

st.write(
    "Check how a planned purchase will affect "
    "your current and future financial position."
)

purchase_amount = st.number_input(
    "Enter planned purchase amount (₹)",
    min_value=0.0,
    value=14000.0,
    step=500.0,
    key="purchase_amount"
)

minimum_balance = 5000

# Calculate balance after purchase
remaining_balance = balance - purchase_amount

# Calculate future balance after purchase
future_balances = (
    forecast_df["Predicted Balance"] - purchase_amount
)

lowest_future_balance = future_balances.min()

col1, col2, col3 = st.columns(3)

col1.metric(
    "Current Balance",
    f"₹{balance:,.2f}"
)

col2.metric(
    "After Purchase",
    f"₹{remaining_balance:,.2f}",
    delta=f"-₹{purchase_amount:,.2f}"
)

col3.metric(
    "Lowest Future Balance",
    f"₹{lowest_future_balance:,.2f}"
)

st.subheader("Purchase Risk Assessment")

if remaining_balance < 0:

    st.error(
        "🚨 CRITICAL: You cannot afford this purchase "
        "with your current balance!"
    )

elif lowest_future_balance < 0:

    st.error(
        "🚨 CRITICAL: Your future balance may become negative!"
    )

elif lowest_future_balance < minimum_balance:

    st.warning(
        "⚠️ WARNING: This purchase may push your future "
        "balance below ₹5,000."
    )

else:

    st.success(
        "✅ This purchase appears affordable based "
        "on the current 30-day forecast."
    )

st.caption(
    "Simulation only: the purchase is not added to "
    "your transaction history."
)

# ===== SPENDING BREAKDOWN =====

st.divider()
st.header("📊 Spending Breakdown")

st.write(
    "Understand how your expenses are distributed "
    "across different transaction types."
)

# Select only expense transactions
spending_df = df[df["amount"] < 0].copy()

# Convert negative amounts into positive values
spending_df["spending"] = spending_df["amount"].abs()

# Calculate spending for each description
category_spending = (
    spending_df.groupby("category")["spending"]
    .sum()
    .reset_index()
    .sort_values("spending", ascending=False)
)


# Create pie chart
fig_pie = px.pie(
    category_spending,
    names="category",
    values="spending",
    title="Where Is Your Money Going?",
    hole=0.45
)

fig_pie.update_layout(
    template="plotly_dark"
)

st.plotly_chart(
    fig_pie,
    use_container_width=True
)

# Display category totals
st.subheader("Expense Details")

st.dataframe(
    category_spending.rename(columns={
        "description": "Transaction Type",
        "spending": "Total Spending (₹)"
    }),
    use_container_width=True,
    hide_index=True
)

# Identify highest spending type
highest_category = category_spending.iloc[0]

st.info(
    f"Your highest spending is on "
    f"{highest_category['category']}: "
    f"₹{highest_category['spending']:,.2f}"
)
# ===== MONTHLY SPENDING TRENDS =====

st.divider()

st.subheader("📅 Monthly Spending Trends")

# Keep only expense transactions
monthly_df = df[df["amount"] < 0].copy()

# Convert negative amounts into positive expenses
monthly_df["spending"] = monthly_df["amount"].abs()

# Extract month and year
monthly_df["month"] = monthly_df["date"].dt.to_period("M").astype(str)

# Calculate total spending per month
monthly_spending = (
    monthly_df.groupby("month")["spending"]
    .sum()
    .reset_index()
)

# Display monthly spending chart
fig_monthly = px.bar(
    monthly_spending,
    x="month",
    y="spending",
    title="Monthly Expense Comparison",
    labels={
        "month": "Month",
        "spending": "Total Spending (₹)"
    },
    text_auto=".2s",
    template="plotly_dark"
)

st.plotly_chart(fig_monthly, use_container_width=True)

# Display monthly spending table
st.dataframe(
    monthly_spending,
    use_container_width=True,
    hide_index=True
)
# Calculate recurring expenses (exclude recurring income like Salary)
recurring_expenses = abs(
    recurring_df[recurring_df["amount"] < 0]["amount"].sum()
)
# ===== FINANCIAL HEALTH SCORE =====
# ===== FINANCIAL HEALTH SCORE =====

st.divider()
st.header("💚 Financial Health Score")

st.write(
    "An indicative score based on savings, recurring expenses, "
    "emergency funds and future cash flow."
)

# Calculate the number of months of transaction history
history_months = df["date"].dt.to_period("M").nunique()

if history_months == 0:
    history_months = 1

# Calculate monthly averages
monthly_income = income / history_months
monthly_expenses = expenses / history_months

monthly_recurring_expenses = recurring_expenses / history_months

# 1. Savings Rate (25 points)
if income > 0:
    savings_rate = (net_cashflow / income) * 100
else:
    savings_rate = 0

if savings_rate >= 20:
    savings_score = 25
elif savings_rate >= 10:
    savings_score = 20
elif savings_rate > 0:
    savings_score = 10
else:
    savings_score = 0

# 2. Recurring Expense Burden (25 points)
if monthly_income > 0:
    recurring_ratio = (
        monthly_recurring_expenses / monthly_income
    ) * 100
else:
    recurring_ratio = 100

if recurring_ratio <= 30:
    recurring_score = 25
elif recurring_ratio <= 40:
    recurring_score = 20
elif recurring_ratio <= 50:
    recurring_score = 10
else:
    recurring_score = 0

# 3. Emergency Fund Coverage (25 points)
if monthly_expenses > 0:
    emergency_months = balance / monthly_expenses
else:
    emergency_months = 0

if emergency_months >= 3:
    emergency_score = 25
elif emergency_months >= 2:
    emergency_score = 20
elif emergency_months >= 1:
    emergency_score = 15
elif emergency_months >= 0.5:
    emergency_score = 10
else:
    emergency_score = 0

# 4. Future Cash Flow Safety (25 points)
if lowest_balance >= monthly_expenses:
    forecast_score = 25
elif lowest_balance >= monthly_expenses * 0.5:
    forecast_score = 18
elif lowest_balance >= 0:
    forecast_score = 10
else:
    forecast_score = 0

# Final score
health_score = (
    savings_score
    + recurring_score
    + emergency_score
    + forecast_score
)

# Display score
col1, col2 = st.columns(2)

with col1:
    st.metric("Financial Health Score", f"{health_score}/100")
    st.progress(health_score / 100)

with col2:
    if health_score >= 80:
        st.success("Strong financial indicators")
    elif health_score >= 60:
        st.warning("Moderate financial indicators")
    else:
        st.error("Financial indicators need attention")

# Breakdown
st.subheader("Score Breakdown")

st.write(f"Savings Rate: {savings_score}/25 ({savings_rate:.1f}%)")
st.write(f"Recurring Expenses: {recurring_score}/25 ({recurring_ratio:.1f}%)")
st.write(f"Emergency Fund: {emergency_score}/25 ({emergency_months:.2f} months)")
st.write(f"Future Cash Flow: {forecast_score}/25")

# Confidence based on available history
if history_months < 3:
    confidence = "Low"
elif history_months < 6:
    confidence = "Medium"
else:
    confidence = "Higher"

st.subheader("Prediction Confidence")
st.write(f"Available transaction history: {history_months} months")
st.write(f"Confidence level: **{confidence}**")

if confidence == "Low":
    st.info("Add at least 3–6 months of transaction history for a more reliable assessment.")

st.caption(
    "Demo scoring model. Scores and thresholds are illustrative, "
    "not professional financial advice or a validated financial rating."
)
# ===== FINANCIAL ADVISOR =====

st.divider()
st.header("🧠 Financial Advisor")

st.caption(
    "Personalized financial insights generated from your transaction data."
)

insights = []

# 1. Savings rate analysis
if savings_rate < 10:
    insights.append((
        "Low Savings Rate",
        f"Your savings rate is only {savings_rate:.1f}%. "
        "Try reducing non-essential spending and building your savings.",
        "warning"
    ))
elif savings_rate >= 20:
    insights.append((
        "Good Savings Rate",
        f"Your savings rate is {savings_rate:.1f}%. "
        "You're saving a healthy portion of your recorded income.",
        "success"
    ))
else:
    insights.append((
        "Savings Opportunity",
        f"Your savings rate is {savings_rate:.1f}%. "
        "Try gradually increasing the amount you save each month.",
        "info"
    ))

# 2. Recurring expense analysis
if recurring_ratio > 40:
    insights.append((
        "High Recurring Commitments",
        f"Recurring expenses use {recurring_ratio:.1f}% of your monthly income. "
        "Review subscriptions and other recurring payments.",
        "warning"
    ))
else:
    insights.append((
        "Recurring Expenses Under Control",
        f"Recurring expenses use {recurring_ratio:.1f}% of your monthly income.",
        "success"
    ))

# 3. Emergency fund analysis
if emergency_months < 1:
    insights.append((
        "Emergency Fund Needs Attention",
        f"Your current balance covers only {emergency_months:.2f} "
        "months of average expenses. Work towards building an emergency fund.",
        "warning"
    ))
elif emergency_months < 3:
    insights.append((
        "Build Your Emergency Fund",
        f"You have approximately {emergency_months:.2f} months "
        "of expense coverage. Aim for at least 3 months.",
        "info"
    ))
else:
    insights.append((
        "Good Emergency Fund Coverage",
        f"Your current balance covers {emergency_months:.2f} "
        "months of average expenses.",
        "success"
    ))

# 4. Future cash flow analysis
if lowest_balance < 0:
    insights.append((
        "Future Cash Shortage Risk",
        f"Your projected balance may fall to ₹{lowest_balance:,.2f}. "
        "Review upcoming expenses and recurring payments.",
        "warning"
    ))
elif lowest_balance < 5000:
    insights.append((
        "Low Future Balance",
        f"Your projected balance may fall to ₹{lowest_balance:,.2f}. "
        "Consider keeping additional money available.",
        "warning"
    ))
else:
    insights.append((
        "Future Cash Flow Looks Stable",
        f"Your lowest projected balance is ₹{lowest_balance:,.2f}.",
        "success"
    ))

# 5. Purchase simulation analysis
if purchase_amount > 0 and lowest_future_balance < 5000:
    insights.append((
        "Purchase Recommendation",
        f"After a purchase of ₹{purchase_amount:,.2f}, "
        f"your projected lowest balance becomes ₹{lowest_future_balance:,.2f}. "
        "Consider reducing or postponing this purchase.",
        "warning"
    ))

# Display recommendations
st.subheader("Your Financial Insights")

for title, message, level in insights:
    if level == "warning":
        st.warning(f"**{title}**\n\n{message}")
    elif level == "success":
        st.success(f"**{title}**\n\n{message}")
    else:
        st.info(f"**{title}**\n\n{message}")

st.caption(
    "These recommendations are generated using transparent "
    "rule-based calculations and are not professional financial advice."
)
