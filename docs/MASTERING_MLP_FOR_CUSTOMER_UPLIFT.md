# Mastering Multi-Layer Perceptron (MLP) untuk Rekomendasi Peningkatan Nilai Pelanggan SocialChat

*A Step-by-Step Guide to Building and Training Neural Networks for Customer Value Uplift*

---

Dalam machine learning, salah satu tugas paling fundamental adalah **klasifikasi**. **Multi-Layer Perceptron (MLP)** memberikan fondasi yang sangat baik untuk memahami cara kerja neural network. MLP adalah jenis neural network yang tersusun dari beberapa lapisan neuron, sehingga mampu mempelajari hubungan kompleks dan non-linear dalam data.

Dalam artikel ini, kita akan menjelajahi cara kerja MLP dan menggunakannya untuk mengklasifikasikan **potensi peningkatan nilai pelanggan (uplift potential)** di aplikasi **SocialChat** ke dalam tiga kelas: `Low`, `Medium`, dan `High`.

---

## Resources

- **Repo Proyek (Full Code):** [github.com/asepsunarya/socialchat-customer-uplift-mlp](https://github.com/asepsunarya/socialchat-customer-uplift-mlp)
- **Dashboard Visualisasi:** [github.com/asepsunarya/socialchat-uplift-dashboard](https://github.com/asepsunarya/socialchat-uplift-dashboard)

---

## What is a Multi-Layer Perceptron (MLP)?

**Multi-Layer Perceptron (MLP)** adalah jenis artificial neural network yang terdiri dari beberapa lapisan neuron. Disebut "multi-layer" karena terdapat tiga jenis lapisan:

1. **Input Layer** — menerima data awal.
2. **Hidden Layer(s)** — memproses data dan mengekstrak pola.
3. **Output Layer** — menghasilkan prediksi akhir.

### Input Layer

Lapisan input menerima data mentah. Dalam kasus SocialChat, setiap neuron di lapisan input mewakili satu fitur pelanggan.

Contoh fitur yang kita gunakan:

- `account_age_days`
- `monthly_active_days`
- `messages_sent_per_month`
- `channels_joined`
- `support_tickets_last_30d`
- `csat_score`
- `referral_count`
- `features_used`
- `avg_session_minutes`
- `days_since_last_login`
- `lifetime_value`
- `churn_risk_score`
- `current_plan` (setelah one-hot encoding menjadi beberapa neuron)

Setiap baris data adalah satu vektor fitur yang merepresentasikan satu pelanggan. Lapisan input hanya menyalurkan data ke lapisan berikutnya tanpa modifikasi.

### Hidden Layers

Lapisan tersembunyi adalah tempat "kerja berat" model terjadi. Setiap neuron di hidden layer menerima input dari lapisan sebelumnya, menghitung **weighted sum**, menambahkan **bias**, lalu meneruskannya ke fungsi aktivasi seperti **ReLU**.

Fungsi ReLU memperkenalkan **non-linearitas**, sehingga jaringan dapat menangkap pola yang kompleks. Tanpa non-linearitas, MLP hanya akan menjadi model linear biasa.

Dalam repo ini, arsitektur hidden layer didefinisikan di `src/socialchat_uplift/models.py`:

```python
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),  # 2 hidden layer: 64 neuron dan 32 neuron
    activation="relu",            # Rectified Linear Unit
    solver="adam",                  # Optimizer Adam
    max_iter=500,
    early_stopping=True,
)
```

MLP bersifat **fully connected**, artinya setiap neuron di satu lapisan terhubung ke semua neuron di lapisan berikutnya.

### Output Layer

Lapisan output menghasilkan prediksi akhir. Karena kita punya 3 kelas (`Low`, `Medium`, `High`), output layer memiliki **3 neuron**. Fungsi **Softmax** mengubah nilai mentah menjadi probabilitas yang jumlahnya `1` (atau 100%).

Kelas dengan probabilitas tertinggi dipilih sebagai prediksi akhir.

```text
Low     → 0.15
Medium  → 0.65  ← prediksi
High    → 0.20
```

---

## Practical Example: Memprediksi Uplift Potential Pelanggan SocialChat

Tujuan kita: membangun model yang bisa memprediksi apakah seorang pelanggan SocialChat memiliki potensi peningkatan nilai **rendah**, **sedang**, atau **tinggi**.

### Step 1: Preparing the Data

Data preparation sangat penting sebelum data dimasukkan ke MLP.

#### 1.1 Feature Vector

Setiap pelanggan direpresentasikan sebagai vektor fitur numerik. Fitur kategorikal seperti `current_plan` perlu diubah menjadi numerik melalui **one-hot encoding**.

```text
Free  → [1, 0, 0]
Basic → [0, 1, 0]
Pro   → [0, 0, 1]
```

Implementasinya ada di `src/socialchat_uplift/preprocess.py`:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(drop="first"), CATEGORICAL_FEATURES),
    ]
)
```

#### 1.2 Normalization / Standardization

Fitur numerik seperti `lifetime_value` memiliki skala jauh lebih besar daripada `csat_score` (1–5). Untuk membuat training lebih stabil, setiap fitur numerik di-`StandardScaler` sehingga memiliki mean 0 dan standar deviasi 1.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)
```

#### 1.3 Label Encoding

Target kita adalah kategorikal (`Low`, `Medium`, `High`). Kita ubah menjadi angka menggunakan `LabelEncoder`:

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(["Low", "Medium", "High"])
# Low=0, Medium=1, High=2
```

Kemudian data dibagi menjadi **training set** dan **test set**:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

---

### Step 2: Building the MLP Model

Arsitektur model dalam proyek ini:

| Layer | Neurons | Activation | Keterangan |
|-------|---------|------------|------------|
| Input Layer | ~14 neuron | — | 12 fitur numerik + one-hot plan |
| Hidden Layer 1 | 64 neuron | ReLU | Ekstraksi pola utama |
| Hidden Layer 2 | 32 neuron | ReLU | Refinement pola |
| Output Layer | 3 neuron | Softmax | Probabilitas Low / Medium / High |

Didefinisikan di `src/socialchat_uplift/models.py`:

```python
def build_mlp(hidden_layers=(64, 32), max_iter=500):
    return MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        early_stopping=True,
    )
```

### What is ReLU?

**ReLU (Rectified Linear Unit)** didefinisikan sebagai:

```text
f(x) = max(0, x)
```

ReLU mengubah nilai negatif menjadi 0 dan membiarkan nilai positif tetap. Fungsi ini membantu jaringan fokus pada fitur yang penting dan mempercepat konvergensi training.

### What is Softmax?

Softmax digunakan di output layer untuk multi-class classification. Ia mengubah skor mentah menjadi probabilitas:

```text
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

Semua nilai berada di antara 0 dan 1, dan totalnya `1`. Kelas dengan probabilitas tertinggi dipilih sebagai hasil prediksi.

---

### Step 3: Forward Propagation (Making Predictions)

Forward propagation adalah proses saat jaringan membuat prediksi. Input data mengalir dari input layer → hidden layers → output layer.

```text
Input  →  Hidden 1 (z = W·x + b, ReLU)  →  Hidden 2 (z = W·h + b, ReLU)  →  Output (z = W·h + b, Softmax)
```

Secara matematis:

```text
hidden_sum = Σ(input_i × weight_i) + bias
hidden_activation = ReLU(hidden_sum)

output_sum = Σ(hidden_activation_j × weight_j) + bias
probabilities = Softmax(output_sum)
```

Pada repo, kita hanya perlu memanggil:

```python
y_pred = mlp.predict(X_test)
probabilities = mlp.predict_proba(X_test)
```

Semua perhitungan forward propagation diurus oleh scikit-learn.

---

### Step 4: Backpropagation (Learning from Errors)

Backpropagation adalah fase belajar. Setelah forward pass, jaringan menghitung error dengan membandingkan prediksi dengan label sebenarnya, lalu memperbarui bobot dan bias untuk mengurangi error.

#### 4.1 Calculate the Error

Untuk klasifikasi multi-kelas, kita biasanya menggunakan **Cross-Entropy Loss**, bukan MSE, karena loss ini lebih cocok untuk probabilitas kelas:

```text
CrossEntropy = -Σ y_true · log(y_pred)
```

Semakin kecil loss, semakin akurat model.

#### 4.2 Compute Gradients

Backpropagation menggunakan **gradient descent** untuk mengupdate bobot. Gradient menunjukkan seberapa besar setiap bobot harus diubah untuk mengurangi loss.

#### 4.3 Update Weights and Biases

Optimizer **Adam** memperbarui bobot secara adaptif berdasarkan gradient. Learning rate mengontrol seberapa besar perubahan tersebut:

```python
mlp = MLPClassifier(
    solver="adam",
    learning_rate_init=0.001,
    ...
)
```

Pseudo code backpropagation:

```text
for each epoch:
    for each mini-batch:
        forward pass  → hitung prediksi
        compute loss  → bandingkan dengan label asli
        backward pass → hitung gradient
        update weights with Adam
```

Kurva loss selama training dapat dilihat di file yang dihasilkan `figures/training_loss.png` dan nilai final loss tersimpan di `models_output/metrics.json`.

---

### Step 5: Training the MLP

Training berarti menjalankan forward propagation dan backpropagation berulang kali selama beberapa **epoch**. Setiap epoch adalah satu kali iterasi penuh melalui seluruh dataset training.

```python
mlp.fit(X_train, y_train)
```

Parameter penting:

- **Epochs / max_iter**: maksimal iterasi training (default 500).
- **Batch size**: scikit-learn menggunakan mini-batch otomatis sesuai solver (`lbfgs`, `sgd`, atau `adam`).
- **Early stopping**: training berhenti otomatis jika validation score tidak membaik selama 10 iterasi berturut-turut. Ini mencegah overfitting.

Setelah training selesai:

```python
print(f"Training completed in {mlp.n_iter_} iterations.")
print(f"Validation score: {mlp.best_validation_score_:.4f}")
```

---

### Step 6: Evaluation

Setelah model terlatih, kita evaluasi menggunakan test set:

```python
from sklearn.metrics import classification_report, confusion_matrix

y_pred = mlp.predict(X_test)
print(classification_report(y_test, y_pred))
```

Metrik yang penting:

- **Accuracy**: persentase prediksi benar.
- **Precision**: seberapa andal prediksi positif.
- **Recall**: seberapa banyak kasus positif yang berhasil ditemukan.
- **F1-Score**: rata-rata harmonic precision dan recall.

Hasil evaluasi dapat dijalankan melalui:

```bash
python -m src.socialchat_uplift.evaluate
```

---

## From MLP Output to Business Action

Setelah model memprediksi `uplift_potential` per pelanggan, sistem rekomendasi (`src/socialchat_uplift/recommender.py`) memetakan hasilnya ke aksi bisnis:

| Uplift | Plan | Recommended Action |
|--------|------|--------------------|
| High | Free | Upsell ke Premium/Pro |
| High | Basic | Upgrade ke Pro + fitur eksklusif |
| High | Pro | Program loyalitas & referral |
| Medium | Free | Edukasi fitur & aktivasi |
| Medium | Basic | Aktivasi fitur jarang dipakai |
| Low | Any | Program retensi & customer success |

Aksi ini kemudian divisualisasikan di dashboard dan bisa diekspor ke tim CRM.

---

## Summary

MLP yang kita bangun mengikuti alur standar neural network:

1. **Input Layer** menerima fitur pelanggan SocialChat.
2. **Hidden Layers** `(64, 32)` dengan ReLU mengekstrak pola non-linear.
3. **Output Layer** dengan Softmax menghasilkan probabilitas 3 kelas.
4. **Backpropagation** dengan Adam memperbarui bobot untuk meminimalkan Cross-Entropy Loss.
5. **Evaluation** memastikan model cukup akurat sebelum digunakan.
6. **Recommendation Engine** mengubah prediksi menjadi aksi bisnis yang konkret.

---

## Next Steps

- Ganti data dummy di `data/customers.csv` dengan data riil dari SocialChat.
- Eksperimen arsitektur hidden layer yang berbeda (misal: `(128, 64, 32)`).
- Bandingkan performa MLP dengan model lain seperti Random Forest atau XGBoost.
- Gunakan SHAP atau permutation importance untuk interpretasi fitur.

---

*Dibuat untuk keperluan Tugas Akhir Asep Sunarya.*
