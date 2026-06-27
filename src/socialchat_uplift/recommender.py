"""
Rekomendasi aksi peningkatan nilai pelanggan berdasarkan prediksi MLP.
"""

import joblib
import pandas as pd

from .preprocess import prepare_new_data


RECOMMENDATION_RULES = {
    "High": {
        "Free": "Upsell ke paket Premium / Pro — potensi nilai sangat tinggi.",
        "Basic": "Upgrade ke paket Pro dan tawarkan fitur eksklusif.",
        "Pro": "Jalankan program loyalitas & referral untuk mempertahankan nilai.",
    },
    "Medium": {
        "Free": "Kampanye edukasi fitur & aktivasi agar beralih ke Basic.",
        "Basic": "Aktivasi fitur yang jarang digunakan untuk meningkatkan engagement.",
        "Pro": "Berikan rekomendasi penggunaan fitur lanjutan.",
    },
    "Low": {
        "Free": "Retensi dasar — survei kebutuhan dan tawarkan trial Basic.",
        "Basic": "Program retensi & customer success — kurangi churn risk.",
        "Pro": "Periksa keluhan & perpanjang manfaat premium.",
    },
}

PRIORITY_ORDER = {"High": 1, "Medium": 2, "Low": 3}


def get_recommendation(uplift_label, current_plan):
    plan = current_plan if current_plan in RECOMMENDATION_RULES[uplift_label] else "Free"
    action = RECOMMENDATION_RULES[uplift_label][plan]
    priority = PRIORITY_ORDER[uplift_label]
    return {
        "uplift_potential": uplift_label,
        "priority": priority,
        "recommended_action": action,
    }


def recommend_batch(df, model_path="models_output/mlp_model.joblib",
                    preprocessor_path="models_output/preprocessor.joblib",
                    label_encoder_path="models_output/label_encoder.joblib"):
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    label_encoder = joblib.load(label_encoder_path)

    ids = df.get("customer_id").values if "customer_id" in df.columns else None
    plans = df["current_plan"].values if "current_plan" in df.columns else None

    X = prepare_new_data(df.drop(columns=["customer_id"], errors="ignore"), preprocessor)
    preds = model.predict(X)
    pred_labels = label_encoder.inverse_transform(preds)
    proba = model.predict_proba(X)

    results = []
    for i, label in enumerate(pred_labels):
        rec = get_recommendation(label, plans[i] if plans is not None else "Free")
        result = {
            "customer_id": ids[i] if ids is not None else i,
            "current_plan": plans[i] if plans is not None else None,
            **rec,
        }
        # confidence score = probabilitas kelas prediksi
        result["confidence"] = round(proba[i][preds[i]], 3)
        results.append(result)

    return pd.DataFrame(results)
