"""
Generate synthetic SocialChat customer dataset for TA demonstration.
"""

import argparse
import os
import random

import numpy as np
import pandas as pd


def generate_customers(n=2000, seed=42):
    rng = np.random.default_rng(seed)

    customer_id = [f"SC_{i+1:05d}" for i in range(n)]

    account_age_days = rng.integers(7, 730, size=n)
    monthly_active_days = rng.integers(1, 31, size=n)
    messages_sent_per_month = rng.integers(0, 5000, size=n)
    channels_joined = rng.integers(0, 50, size=n)
    support_tickets_last_30d = rng.integers(0, 12, size=n)
    csat_score = np.clip(rng.normal(3.5, 0.8, size=n), 1, 5).round(1)
    referral_count = rng.poisson(1, size=n)
    features_used = rng.integers(1, 15, size=n)
    avg_session_minutes = np.clip(rng.normal(20, 12, size=n), 1, 120).round(1)
    days_since_last_login = rng.integers(0, 90, size=n)

    plans = ["Free", "Basic", "Pro"]
    current_plan = rng.choice(plans, size=n, p=[0.6, 0.3, 0.1])

    lifetime_value = (account_age_days / 30.0 * 5 +
                      monthly_active_days * 8 +
                      messages_sent_per_month * 0.05 +
                      referral_count * 50 +
                      features_used * 30 +
                      np.where(current_plan == "Pro", 500,
                               np.where(current_plan == "Basic", 150, 0)) +
                      rng.normal(0, 200, size=n)).round(0).astype(int)
    lifetime_value = np.clip(lifetime_value, 0, None)

    churn_risk_score = np.clip(
        0.4
        + 0.005 * days_since_last_login
        - 0.03 * csat_score
        + 0.05 * support_tickets_last_30d
        - 0.001 * avg_session_minutes
        - features_used * 0.01
        + rng.normal(0, 0.1, size=n),
        0, 1
    ).round(3)

    # Hidden uplift potential logic
    plan_score = np.where(current_plan == "Free", 0.7,
                          np.where(current_plan == "Basic", 0.4, 0.15))
    uplift_score_raw = (
        0.3 * plan_score
        + 0.25 * (monthly_active_days / 30.0)
        + 0.25 * (referral_count / 5.0)
        + 0.1 * (features_used / 15.0)
        - 0.1 * churn_risk_score
        + rng.normal(0, 0.08, size=n)
    )

    uplift_potential = pd.cut(
        uplift_score_raw,
        bins=[-np.inf, 0.35, 0.65, np.inf],
        labels=["Low", "Medium", "High"]
    ).astype(str)

    df = pd.DataFrame({
        "customer_id": customer_id,
        "account_age_days": account_age_days,
        "monthly_active_days": monthly_active_days,
        "messages_sent_per_month": messages_sent_per_month,
        "channels_joined": channels_joined,
        "support_tickets_last_30d": support_tickets_last_30d,
        "csat_score": csat_score,
        "referral_count": referral_count,
        "features_used": features_used,
        "avg_session_minutes": avg_session_minutes,
        "days_since_last_login": days_since_last_login,
        "current_plan": current_plan,
        "lifetime_value": lifetime_value,
        "churn_risk_score": churn_risk_score,
        "uplift_potential": uplift_potential,
    })

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--samples", type=int, default=2000)
    parser.add_argument("-o", "--output", default="data/customers.csv")
    parser.add_argument("-s", "--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate_customers(n=args.samples, seed=args.seed)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Generated {len(df)} samples -> {args.output}")
    print(df["uplift_potential"].value_counts())


if __name__ == "__main__":
    main()
