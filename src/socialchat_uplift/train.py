"""
Script training model MLP untuk rekomendasi peningkatan nilai pelanggan.
"""

import argparse
import json
import os

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from .models import build_mlp
from .preprocess import prepare_data


def plot_loss_curve(model, output_path="figures/training_loss.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not hasattr(model, "loss_curve_") or model.loss_curve_ is None:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(model.loss_curve_, label="Training loss")
    if hasattr(model, "validation_scores_") and model.validation_scores_ is not None:
        plt.plot(model.validation_scores_, label="Validation score")
    plt.title("MLP Training Loss Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Loss / Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved loss curve -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train SocialChat uplift MLP")
    parser.add_argument("--data", default="data/customers.csv", help="Path ke CSV dataset")
    parser.add_argument("--output", default="models_output", help="Direktori output model")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--hidden", nargs="+", type=int, default=[64, 32], help="Hidden layer sizes")
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Preparing data...")
    X_train, X_test, y_train, y_test, preprocessor, label_encoder = prepare_data(
        args.data, test_size=args.test_size, random_state=args.seed, save_dir=args.output
    )

    print(f"Training MLP with hidden layers {tuple(args.hidden)} ...")
    mlp = build_mlp(
        hidden_layers=tuple(args.hidden),
        max_iter=args.max_iter,
        random_state=args.seed,
    )
    mlp.fit(X_train, y_train)

    print(f"Training completed in {mlp.n_iter_} iterations. Loss: {mlp.loss_:.4f}")
    print(f"Validation score: {mlp.best_validation_score_:.4f}")

    y_pred = mlp.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    model_path = os.path.join(args.output, "mlp_model.joblib")
    joblib.dump(mlp, model_path)
    print(f"\nModel saved -> {model_path}")

    metrics = {
        "n_iterations": int(mlp.n_iter_),
        "final_loss": float(mlp.loss_),
        "best_validation_score": float(mlp.best_validation_score_),
        "classes": label_encoder.classes_.tolist(),
    }
    with open(os.path.join(args.output, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    plot_loss_curve(mlp)


if __name__ == "__main__":
    main()
