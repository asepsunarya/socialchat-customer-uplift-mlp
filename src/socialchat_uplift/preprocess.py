"""
Preprocessing pipeline untuk dataset SocialChat.
"""

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

NUMERIC_FEATURES = [
    "account_age_days",
    "monthly_active_days",
    "messages_sent_per_month",
    "channels_joined",
    "support_tickets_last_30d",
    "csat_score",
    "referral_count",
    "features_used",
    "avg_session_minutes",
    "days_since_last_login",
    "lifetime_value",
    "churn_risk_score",
]

CATEGORICAL_FEATURES = [
    "current_plan",
]

TARGET = "uplift_potential"


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


def prepare_data(csv_path, test_size=0.2, random_state=42, save_dir="models_output"):
    df = pd.read_csv(csv_path)

    X = df.drop(columns=[TARGET, "customer_id"])
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    joblib.dump(preprocessor, f"{save_dir}/preprocessor.joblib")
    joblib.dump(label_encoder, f"{save_dir}/label_encoder.joblib")

    return X_train_processed, X_test_processed, y_train_encoded, y_test_encoded, preprocessor, label_encoder


def prepare_new_data(df, preprocessor):
    """Transform DataFrame baru (tanpa kolom target dan customer_id)."""
    return preprocessor.transform(df)
