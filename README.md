# SocialChat Customer Value Uplift Recommendation with MLP

Repositori ini berisi implementasi **Multilayer Perceptron (MLP)** untuk memprediksi potensi peningkatan nilai pelanggan pada platform **SocialChat** dan memberikan rekomendasi aksi bisnis yang sesuai.

## 🎯 Tujuan

Mengklasifikasikan pelanggan ke dalam tiga kelompok potensi *uplift*:
- **Low** — pelanggan dengan potensi peningkatan nilai rendah.
- **Medium** — pelanggan berpotensi ditingkatkan dengan intervensi ringan.
- **High** — pelanggan berpotensi tinggi untuk di-*upsell* atau dimaksimalkan.

Berdasarkan prediksi MLP, sistem memberikan rekomendasi aksi seperti *upsell*, *feature adoption*, atau program *retention*.

## 🗂️ Struktur Repo

```
.
├── data/                         # Dataset (dihasilkan otomatis)
├── figures/                        # Visualisasi hasil training
├── models_output/                  # Model, preprocessor, dan metrics
├── notebooks/                      # Notebook eksplorasi
├── src/socialchat_uplift/
│   ├── data_generator.py           # Generator data dummy SocialChat
│   ├── preprocess.py               # Pipeline preprocessing
│   ├── models.py                   # Arsitektur MLP
│   ├── train.py                    # Script training
│   ├── evaluate.py                 # Evaluasi model
│   ├── recommender.py              # Mesin rekomendasi
│   └── predict.py                  # Inference
├── tests/                          # Unit test
├── main.py                         # Pipeline end-to-end
├── requirements.txt
└── README.me                       # Dokumen ini
```

## ⚙️ Instalasi

```bash
git clone https://github.com/asepsunarya/socialchat-customer-uplift-mlp.git
cd socialchat-customer-uplift-mlp
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🚀 Cara Penggunaan

### 1. Jalankan pipeline lengkap

```bash
python main.py
```

Pipeline akan:
- Membuat dataset dummy `data/customers.csv`
- Melatih model MLP
- Mengevaluasi performa
- Menghasilkan contoh rekomendasi

### 2. Training manual

```bash
python -m src.socialchat_uplift.data_generator -n 5000 -o data/customers.csv
python -m src.socialchat_uplift.train --data data/customers.csv --hidden 128 64 32 --max-iter 600
python -m src.socialchat_uplift.evaluate --data data/customers.csv --model-dir models_output
```

### 3. Prediksi untuk data baru

Siapkan CSV dengan kolom yang sama seperti `data/customers.csv` (tanpa `uplift_potential`), lalu:

```bash
python -m src.socialchat_uplift.predict --input data/sample_predict.csv --output data/recommendations.csv
```

## 📊 Fitur yang Digunakan

| Fitur | Deskripsi |
|-------|-----------|
| `account_age_days` | Lama akun (hari) |
| `monthly_active_days` | Hari aktif per bulan |
| `messages_sent_per_month` | Jumlah pesan per bulan |
| `channels_joined` | Jumlah channel yang diikuti |
| `support_tickets_last_30d` | Tiket dukungan 30 hari terakhir |
| `csat_score` | Skor kepuasan pelanggan (1-5) |
| `referral_count` | Jumlah referral |
| `features_used` | Fitur aktif yang digunakan |
| `avg_session_minutes` | Durasi rata-rata sesi |
| `days_since_last_login` | Hari sejak login terakhir |
| `lifetime_value` | Nilai pelanggan seumur hidup |
| `churn_risk_score` | Skor risiko churn (0-1) |
| `current_plan` | Paket saat ini (Free/Basic/Pro) |

## 🧠 Model MLP

Model menggunakan `sklearn.neural_network.MLPClassifier` dengan:
- Hidden layers: `(64, 32)`
- Aktivasi: ReLU
- Solver: Adam
- Early stopping aktif
- Output: Softmax 3 kelas (`Low`, `Medium`, `High`)

## 📝 HASIL & METRIK

Setelah training, metrik tersimpan di `models_output/metrics.json` dan kurva loss di `figures/training_loss.png`.

## 🧠 Memahami MLP

Untuk penjelasan lengkap cara kerja MLP pada proyek ini — mulai dari input layer, hidden layer ReLU, output layer softmax, hingga backpropagation — baca:

**[`docs/MASTERING_MLP_FOR_CUSTOMER_UPLIFT.md`](docs/MASTERING_MLP_FOR_CUSTOMER_UPLIFT.md)**

Artikel tersebut disusun mengikuti gaya tutorial step-by-step seperti Medium.

## 📚 Untuk Tugas Akhir

Repositori ini bisa dikembangkan lebih lanjut dengan:
1. Mengganti data dummy dengan data riil SocialChat.
2. Menambahkan hyperparameter tuning (GridSearch/RandomSearch).
3. Membandingkan MLP dengan model lain (XGBoost, RandomForest).
4. Menambahkan SHAP/LIME untuk interpretasi model.

## 👤 Author

Asep Sunarya — untuk keperluan Tugas Akhir.
