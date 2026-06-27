"""
Inference script untuk rekomendasi pelanggan SocialChat.
"""

import argparse

import pandas as pd

from .recommender import recommend_batch


def main():
    parser = argparse.ArgumentParser(description="Predict customer uplift recommendations")
    parser.add_argument("--input", required=True, help="Path CSV data pelanggan baru")
    parser.add_argument("--output", default="data/recommendations.csv", help="Path output rekomendasi")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    recommendations = recommend_batch(df)
    recommendations.to_csv(args.output, index=False)
    print(f"Generated {len(recommendations)} recommendations -> {args.output}")
    print(recommendations.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
