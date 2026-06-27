"""
Evaluate the trained MLP model.
"""

import json

import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report

from .preprocess import prepare_data


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/customers.csv")
    parser.add_argument("--model-dir", default="models_output")
    parser.add_argument("--output", default="models_output/evaluation.json")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, _, label_encoder = prepare_data(
        args.data, save_dir=args.model_dir
    )

    model = joblib.load(f"{args.model_dir}/mlp_model.joblib")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)

    evaluation = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "classification_report": report,
    }

    with open(args.output, "w") as f:
        json.dump(evaluation, f, indent=2, default=float)

    print(json.dumps({k: evaluation[k] for k in ["accuracy", "f1_macro", "f1_weighted"]}, indent=2))


if __name__ == "__main__":
    main()
