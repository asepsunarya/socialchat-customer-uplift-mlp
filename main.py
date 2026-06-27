"""
Entry point pipeline lengkap:
1. Generate data dummy
2. Train MLP
3. Evaluate
4. Generate contoh rekomendasi
"""

import os
import subprocess
import sys


def run(cmd):
    print(f"\n>> {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models_output", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    run(f"{sys.executable} -m src.socialchat_uplift.data_generator -n 2000 -o data/customers.csv")
    run(f"{sys.executable} -m src.socialchat_uplift.train --data data/customers.csv --output models_output --hidden 64 32")
    run(f"{sys.executable} -m src.socialchat_uplift.evaluate --data data/customers.csv --model-dir models_output")

    # Buat sample inference dari 5 baris pertama test data
    df = __import__("pandas").read_csv("data/customers.csv").head(10)
    df.to_csv("data/sample_predict.csv", index=False)
    run(f"{sys.executable} -m src.socialchat_uplift.predict --input data/sample_predict.csv --output data/recommendations.csv")


if __name__ == "__main__":
    main()
