"""
Ingest raw SocialChat event logs and convert them into the feature CSV
expected by the MLP training pipeline.
"""

import argparse
from datetime import datetime, timedelta

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Build SocialChat MLP features from raw events")
    parser.add_argument("--users", required=True, help="Path CSV users (customer_id, created_at, current_plan)")
    parser.add_argument("--events", required=True, help="Path CSV events (customer_id, timestamp, event_type, event_value, meta/feature/channel)")
    parser.add_argument("--payments", default=None, help="Optional CSV payments (customer_id, timestamp, amount)")
    parser.add_argument("--reference-date", default=None, help="Cut-off date YYYY-MM-DD. Default today")
    parser.add_argument("--window-days", type=int, default=30, help="Aggregation window in days")
    parser.add_argument("--output", default="data/customers_real.csv", help="Output CSV path")
    parser.add_argument("--labels-from-payments", action="store_true", help="Generate labels from revenue growth")
    return parser.parse_args()


def load_csv(path, parse_dates=None):
    return pd.read_csv(path, parse_dates=parse_dates)


def build_features(users_df, events_df, reference_date, window_days=30, payments_df=None):
    reference_date = pd.to_datetime(reference_date).normalize()
    window_start = reference_date - pd.Timedelta(days=window_days)

    recent_events = events_df[
        (events_df["timestamp"] >= window_start) & (events_df["timestamp"] < reference_date)
    ].copy()

    users_df = users_df.copy()
    users_df["created_at"] = pd.to_datetime(users_df["created_at"])
    users_df["account_age_days"] = (reference_date - users_df["created_at"]).dt.days

    # Helper: extract optional channels/features from columns if present
    channel_col = "channel_id" if "channel_id" in recent_events.columns else None
    feature_col = "feature" if "feature" in recent_events.columns else None

    agg = recent_events.groupby("customer_id").agg(
        monthly_active_days=("timestamp", lambda s: s.dt.date.nunique()),
        messages_sent_per_month=("event_type", lambda x: (x == "message_sent").sum()),
        support_tickets_last_30d=("event_type", lambda x: (x == "support_ticket_created").sum()),
        csat_score=("event_value", lambda x: x[recent_events.loc[x.index, "event_type"] == "csat_rating"].mean()),
        referral_count=("event_type", lambda x: (x == "referral_completed").sum()),
        features_used=("event_value", lambda x: x[recent_events.loc[x.index, "event_type"] == "feature_used"].nunique()),
        avg_session_minutes=("event_value", lambda x: x[recent_events.loc[x.index, "event_type"] == "session_end"].mean()),
        last_event=("timestamp", "max"),
    )

    if channel_col:
        agg["channels_joined"] = recent_events.groupby("customer_id")[channel_col].nunique()
    else:
        agg["channels_joined"] = recent_events.groupby("customer_id").size()

    if feature_col:
        agg["features_used"] = recent_events.groupby("customer_id")[feature_col].nunique()
    else:
        # Keep count-based fallback
        agg["features_used"] = recent_events.groupby("customer_id").agg(
            features_used=("event_type", lambda x: x[x == "feature_used"].nunique())
        )["features_used"]

    agg["days_since_last_login"] = (reference_date - agg["last_event"]).dt.days
    agg = agg.drop(columns=["last_event"])

    # Lifetime value from payments if provided, else 0
    if payments_df is not None:
        payments_df = payments_df.copy()
        payments_df["timestamp"] = pd.to_datetime(payments_df["timestamp"])
        payments_before = payments_df[payments_df["timestamp"] < reference_date]
        ltv = payments_before.groupby("customer_id")["amount"].sum().rename("lifetime_value").to_frame()
    else:
        ltv = pd.DataFrame(columns=["lifetime_value"])
        ltv.index.name = "customer_id"

    # Placeholder churn risk if not present
    if "churn_risk_score" not in users_df.columns:
        users_df["churn_risk_score"] = 0.0

    df = users_df.set_index("customer_id")[[
        "account_age_days", "current_plan", "lifetime_value", "churn_risk_score"
    ]].join(agg, how="left").join(ltv, how="left")

    # Fill missing values
    df["csat_score"] = df["csat_score"].fillna(3.0)
    df["avg_session_minutes"] = df["avg_session_minutes"].fillna(0.0)
    df["lifetime_value"] = df["lifetime_value"].fillna(0).astype(int)
    for col in ["monthly_active_days", "messages_sent_per_month", "channels_joined",
                "support_tickets_last_30d", "referral_count", "features_used",
                "days_since_last_login"]:
        df[col] =df[col].fillna(0).astype(int)

    # Drop helper churn if zero; change placeholder based on days_since_last_login
    df["churn_risk_score"] = (
        0.3 + 0.004 * df["days_since_last_login"].clip(upper=90)
        - 0.03 * df["csat_score"]
        + 0.04 * df["support_tickets_last_30d"]
        - 0.001 * df["avg_session_minutes"].clip(upper=120)
    ).clip(0, 1).round(3)

    return df.reset_index()


def generate_labels_from_payments(users_df, payments_df, reference_date, window_days=30):
    reference_date = pd.to_datetime(reference_date)
    payments_df = payments_df.copy()
    payments_df["timestamp"] = pd.to_datetime(payments_df["timestamp"])

    before = payments_df[
        (payments_df["timestamp"] >= reference_date - pd.Timedelta(days=window_days*2)) &
        (payments_df["timestamp"] < reference_date - pd.Timedelta(days=window_days))
    ].groupby("customer_id")["amount"].sum().rename("revenue_before")

    after = payments_df[
        (payments_df["timestamp"] >= reference_date) &
        (payments_df["timestamp"] < reference_date + pd.Timedelta(days=window_days))
    ].groupby("customer_id")["amount"].sum().rename("revenue_after")

    growth = pd.concat([before, after], axis=1).fillna(0)
    growth["growth_rate"] = growth.apply(
        lambda r: (r["revenue_after"] - r["revenue_before"]) / r["revenue_before"]
        if r["revenue_before"] > 0 else -1.0,
        axis=1
    )

    def label(r):
        if r["growth_rate"] >= 0.3:
            return "High"
        if r["growth_rate"] >= 0.1:
            return "Medium"
        return "Low"

    growth["uplift_potential"] = growth.apply(label, axis=1)
    return growth[["uplift_potential"]].reset_index()


def main():
    args = parse_args()

    reference_date = args.reference_date or datetime.now().strftime("%Y-%m-%d")
    print(f"Reference date: {reference_date}, window: {args.window_days} days")

    users_df = load_csv(args.users, parse_dates=["created_at"])
    events_df = load_csv(args.events, parse_dates=["timestamp"])
    payments_df = load_csv(args.payments, parse_dates=["timestamp"]) if args.payments else None

    if "event_value" not in events_df.columns:
        events_df["event_value"] = 1.0

    df = build_features(users_df, events_df, reference_date, args.window_days, payments_df)

    if args.labels_from_payments and payments_df is not None:
        labels = generate_labels_from_payments(users_df, payments_df, reference_date, args.window_days)
        df = df.merge(labels, on="customer_id", how="left")
        df["uplift_potential"] = df["uplift_potential"].fillna("Low")
    else:
        # No labels provided; assign placeholder for inference-only scenarios
        df["uplift_potential"] = "Unknown"

    # Ensure expected column order
    expected_cols = [
        "customer_id", "account_age_days", "monthly_active_days",
        "messages_sent_per_month", "channels_joined", "support_tickets_last_30d",
        "csat_score", "referral_count", "features_used", "avg_session_minutes",
        "days_since_last_login", "current_plan", "lifetime_value",
        "churn_risk_score", "uplift_potential"
    ]

    # Add any missing columns with defaults
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_cols]
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} records to {args.output}")
    print(df.head())


if __name__ == "__main__":
    main()
