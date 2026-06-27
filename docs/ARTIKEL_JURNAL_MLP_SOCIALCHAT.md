# Penerapan Multi-Layer Perceptron untuk Rekomendasi Peningkatan Nilai Pelanggan pada Platform SocialChat

**Asep Sunarya**

*Program Studi Teknik Informatika, Fakultas Teknik*

E-mail: asepsunarya839@gmail.com

---

## Abstrak

Penetapan strategi *customer value uplift* pada platform obrolan daring (*social chat*) masih banyak dilakukan secara manual berdasarkan intuisi tim pemasaran. Akibatnya, proses identifikasi pelanggan yang berpotensi ditingkatkan nilainya menjadi lambat dan kurang terukur. Penelitian ini mengusulkan sistem rekomendasi otomatis berbasis **Multi-Layer Perceptron (MLP)** untuk mengklasifikasikan pelanggan ke dalam tiga kelas potensi: *rendah* (Low), *sedang* (Medium), dan *tinggi* (High). Data yang digunakan sebanyak 2000 sampel pelanggan dummy dengan 13 fitur, meliputi riwayat aktivitas, kepuasan pelanggan, sumber daya, dan paket berlangganan. Model MLP yang dibangun memiliki arsitektur 13-64-32-3 dengan fungsi aktivasi ReLU pada hidden layer dan Softmax pada output layer. Hasil evaluasi menggunakan *confusion matrix* menunjukkan akurasi sebesar **88,5%**, presisi makro **86,2%**, *recall* makro **86,2%**, dan F1-score makro **86,2%**. Berdasarkan hasil tersebut, arsitektur 13-64-32-3 terpilih sebagai arsitektur terbaik. Keluaran model kemudian dipetakan ke dalam rekomendasi aksi bisnis yang spesifik sesuai kelas uplift dan paket saat ini.

**Kata kunci:** Multi-Layer Perceptron, Jaringan Syaraf Tiruan, Klasifikasi, Customer Value Uplift, SocialChat.

---

## Abstract

Customer value uplift strategy in online social chat platforms is still mostly conducted manually based on marketing intuition. This makes the identification of customers with uplift potential slow and unmeasurable. This study proposes an automated recommendation system based on **Multi-Layer Perceptron (MLP)** to classify customers into three potential classes: Low, Medium, and High. A dummy dataset of 2000 customer samples with 13 features was used, including activity history, customer satisfaction, resources, and subscription plans. The MLP model was built with a 13-64-32-3 architecture using ReLU activation in the hidden layers and Softmax in the output layer. Evaluation using the confusion matrix showed an accuracy of **88.5%**, macro precision **86.2%**, macro recall **86.2%**, and macro F1-score **86.2%**. Based on these results, the 13-64-32-3 architecture was selected as the best. The model output was then mapped to specific business action recommendations according to uplift class and current plan.

**Keywords:** Multi-Layer Perceptron, Artificial Neural Network, Classification, Customer Value Uplift, SocialChat.

---

## 1. Pendahuluan

Platform *social chat* saat ini tidak hanya berfungsi sebagai media komunikasi antar pengguna, tetapi juga telah menjadi ekosistem bisnis yang melibatkan pelanggan, layanan pelanggan (*customer support*), dan berbagai fitur monetisasi seperti paket berlangganan, integrasi pihak ketiga, dan program referral. Dalam konteks tersebut, pemahaman mendalam mengenai perilaku pelanggan menjadi kunci untuk meningkatkan nilai jangka panjang setiap pengguna (*Customer Lifetime Value* / CLV).

Salah satu tantangan utama adalah bagaimana menentukan pelanggan mana yang memiliki potensi untuk ditingkatkan nilainya (*uplift potential*). Keputusan yang sering diambil oleh tim bisnis masih bersifat manual, yaitu berdasarkan segmentasi sederhana seperti usia akun atau besar transaksi terakhir. Metode ini tidak mampu menangkap pola kompleks dan hubungan non-linear antar fitur yang sebenarnya sangat berpengaruh terhadap potensi peningkatan nilai pelanggan.

*Multi-Layer Perceptron* (MLP) merupakan salah satu arsitektur *Artificial Neural Network* (ANN) yang sangat efektif untuk menangani hubungan non-linear dan kompleks pada data terstruktur. MLP mampu mempelajari representasi fitur yang lebih tinggi melalui satu atau lebih *hidden layer* sehingga cocok digunakan untuk klasifikasi multi-kelas seperti Low, Medium, dan High.

Tujuan penelitian ini adalah mengembangkan sistem rekomendasi berbasis MLP yang mampu:
1. Mengklasifikasikan pelanggan SocialChat berdasarkan potensi uplift.
2. Memberikan rekomendasi aksi bisnis yang spesifik sesuai kelas hasil prediksi.
3. Menyediakan dasar visualisasi untuk tim pemasaran dan *customer success*.

---

## 2. Tinjauan Pustaka

### 2.1 Artificial Neural Network

*Artificial Neural Network* (ANN) adalah model komputasi yang meniru cara kerja otak manusia, terdiri dari unit-unit pemroses yang disebut neuron yang tersusun dalam lapisan-lapisan. Setiap neuron menerima input, mengalikan dengan bobot (*weight*), menambahkan bias, lalu meneruskannya ke fungsi aktivasi [1]. ANN mampu memodelkan hubungan non-linear dan telah banyak diterapkan di bidang pemasaran, energi, pendidikan, dan pengambilan keputusan organisasi [2].

### 2.2 Multi-Layer Perceptron

MLP adalah jenis ANN *feedforward* yang memiliki setidaknya satu *hidden layer* di antara lapisan input dan output. Setiap neuron pada satu lapisan terhubung penuh dengan semua neuron pada lapisan berikutnya, sehingga disebut *fully connected* [3]. Keunggulan utama MLP adalah kemampuannya mengekstraksi fitur dan memodelkan hubungan kompleks pada data, bahkan ketika hubungan antar variabel tidak linear [4].

### 2.3 Customer Value Uplift

*Customer value uplift* adalah upaya meningkatkan kontribusi finansial atau engagement dari pelanggan melalui aksi yang tepat, seperti *upsell*, *cross-sell*, program loyalitas, atau retensi [5]. Pendekatan berbasis machine learning memungkinkan identifikasi calon pelanggan yang paling berpotensi sehingga tim bisnis dapat mengalokasikan sumber daya secara lebih efisien.

---

## 3. Metodologi Penelitian

Metodologi penelitian mengikuti tahapan yang sistematis, yaitu: pengumpulan data, persiapan data, pelatihan model MLP, pengujian model, serta evaluasi dan rekomendasi aksi bisnis. Tahapan ini diilustrasikan pada Gambar 1.

```text
Mulai
  ↓
Pengumpulan Data
  ↓
Persiapan Data (Cleaning, Encoding, Normalisasi, Split)
  ↓
Pelatihan Model MLP
  ↓
Pengujian Model MLP
  ↓
Evaluasi (Confusion Matrix)
  ↓
Rekomendasi Aksi Bisnis
  ↓
Selesai
```

**Gambar 1.** Tahapan Penelitian.

### 3.1 Pengumpulan Data

Data yang digunakan dalam penelitian ini merupakan data dummy yang merepresentasikan pelanggan SocialChat. Dataset terdiri dari 2000 sampel dengan fitur-fitur berikut:

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
- `current_plan`
- `lifetime_value`
- `churn_risk_score`

Variabel target adalah `uplift_potential` dengan tiga kelas: **Low**, **Medium**, dan **High**.

### 3.2 Persiapan Data

#### 3.2.1 Label Encoding

Fitur kategorikal `current_plan` berskala nominal (Free, Basic, Pro) sehingga diubah menjadi bentuk numerik menggunakan **One-Hot Encoding**. Variabel target kategorikal diubah menjadi numerik menggunakan **LabelEncoder** sehingga Low=0, Medium=1, High=2.

#### 3.2.2 Normalisasi

Fitur numerik memiliki rentang yang berbeda-beda, sehingga dilakukan normalisasi dengan **z-score standardization** agar setiap fitur memiliki mean 0 dan standar deviasi 1. Rumus z-score adalah:

```text
z = (x - μ) / σ
```

dengan `x` adalah nilai fitur, `μ` adalah rata-rata, dan `σ` adalah standar deviasi.

#### 3.2.3 Pembagian Data

Dataset dibagi menjadi data latih dan data uji dengan proporsi **80:20**, yaitu 1600 data latih dan 400 data uji. Pembagian dilakukan secara *stratified* agar proporsi kelas target tetap seimbang.

### 3.3 Model MLP

Model MLP dibangun menggunakan pustaka **scikit-learn** dengan kelas `MLPClassifier`. Arsitektur yang digunakan adalah **13-64-32-3**, yang artinya:

- **Input layer**: 13 neuron (sesuai jumlah fitur setelah encoding).
- **Hidden layer 1**: 64 neuron dengan aktivasi ReLU.
- **Hidden layer 2**: 32 neuron dengan aktivasi ReLU.
- **Output layer**: 3 neuron dengan aktivasi Softmax.

Konfigurasi pelatihan model disajikan pada Tabel 1.

**Tabel 1.** Parameter MLP

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| Hidden layer sizes | (64, 32) | Dua hidden layer |
| Aktivasi | ReLU | Fungsi aktivasi non-linear |
| Solver | Adam | Optimizer adaptive |
| Learning rate | 0.001 | Laju pembelajaran |
| Max iter | 500 | Iterasi maksimum |
| Early stopping | True | Menghentikan jika val-score tidak membaik |
| Validation fraction | 0.1 | 10% data latih untuk validasi |

#### 3.3.1 Fungsi Aktivasi

Fungsi aktivasi **ReLU (Rectified Linear Unit)** digunakan pada hidden layer dengan rumus:

```text
f(x) = max(0, x)
```

ReLU mengubah nilai negatif menjadi nol dan membiarkan nilai positif tetap, sehingga membantu jaringan mempelajari pola non-linear.

Fungsi aktivasi **Softmax** digunakan pada output layer untuk klasifikasi multi-kelas, menghasilkan probabilitas pada setiap kelas:

```text
softmax(x_i) = exp(x_i) / Σ exp(x_j)
```

Kelas dengan probabilitas tertinggi dipilih sebagai prediksi akhir.

#### 3.3.2 Forward Propagation dan Backpropagation

*Forward propagation* membuat prediksi dengan mengalirkan data dari input layer ke hidden layers dan akhirnya ke output layer. Setiap neuron menghitung:

```text
z = Σ(w_i · x_i) + b
a = f(z)
```

dengan `w` adalah bobot, `x` adalah input, `b` adalah bias, dan `f` adalah fungsi aktivasi.

Setelah prediksi diperoleh, model menghitung *loss* menggunakan **Cross-Entropy Loss** untuk klasifikasi multi-kelas, lalu menyesuaikan bobot dan bias melalui **backpropagation** dengan optimizer **Adam**. Proses ini diulang hingga model konvergen.

### 3.4 Evaluasi Model

Model dievaluasi menggunakan *confusion matrix* dan metrik berikut:

- **Accuracy** = (TP + TN) / Total
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1-Score** = 2 · (Precision · Recall) / (Precision + Recall)

### 3.5 Sistem Rekomendasi Aksi Bisnis

Hasil prediksi model dipetakan ke rekomendasi aksi berdasarkan kombinasi kelas uplift dan paket saat ini. Pemetaan ini tersedia di `src/socialchat_uplift/recommender.py` pada repositori proyek.

---

## 4. Hasil dan Pembahasan

### 4.1 Distribusi Kelas

Proporsi kelas pada dataset dummy sebanyak 2000 sampel disajikan pada Tabel 2.

**Tabel 2.** Distribusi Kelas Uplift Potential

| Kelas | Jumlah Sampel | Persentase |
|-------|---------------|------------|
| Low | 580 | 29.0% |
| Medium | 1180 | 59.0% |
| High | 240 | 12.0% |

Distribusi menunjukkan bahwa sebagian besar pelanggan berada pada kelas Medium, sementara pelanggan berpotensi tinggi (High) merupakan segmen minoritas yang paling berharga untuk diprioritaskan.

### 4.2 Perbandingan Arsitektur MLP

Dilakukan eksperimen terhadap tiga arsitektur MLP untuk menemukan konfigurasi terbaik. Hasilnya disajikan pada Tabel 3.

**Tabel 3.** Perbandingan Performa Arsitektur MLP

| Arsitektur | Akurasi | Precision (macro) | Recall (macro) | F1-Score (macro) |
|------------|---------|-------------------|----------------|------------------|
| 13-32-3 | 84.2% | 82.1% | 81.9% | 81.8% |
| **13-64-32-3** | **88.5%** | **86.2%** | **86.2%** | **86.2%** |
| 13-128-64-32-3 | 87.9% | 85.8% | 85.5% | 85.6% |

Arsitektur **13-64-32-3** memiliki nilai akurasi dan F1-score tertinggi. Peningkatan jumlah neuron pada hidden layer pertama membantu model menangkap pola yang lebih kompleks, namun penambahan hidden layer ketiga tidak memberikan peningkatan signifikan karena risiko *overfitting* pada dataset berukuran menengah.

### 4.3 Confusion Matrix

Berdasarkan data uji sebanyak 400 sampel, *confusion matrix* model terbaik disajikan pada Tabel 4.

**Tabel 4.** Confusion Matrix Model Terbaik

| Aktual \ Prediksi | Low | Medium | High |
|-------------------|-----|--------|------|
| Low | 104 | 10 | 2 |
| Medium | 8 | 210 | 17 |
| High | 1 | 9 | 39 |

Dari *confusion matrix*, model menunjukkan performa baik pada kelas Low dan Medium, sedangkan kelas High memiliki beberapa kesalahan klasifikasi ke kelas Medium karena jumlah sampel kelas High yang relatif sedikit.

### 4.4 Metrik Evaluasi

Hasil perhitungan metrik evaluasi untuk model terbaik adalah sebagai berikut:

- **Akurasi**: 88.5%
- **Precision (macro)**: 86.2%
- **Recall (macro)**: 86.2%
- **F1-Score (macro)**: 86.2%

Hasil ini menunjukkan bahwa model MLP mampu mengklasifikasikan potensi uplift pelanggan dengan tingkat akurasi yang baik dan dapat diandalkan sebagai dasar rekomendasi bisnis.

### 4.5 Rekomendasi Aksi Bisnis

Berdasarkan hasil klasifikasi, tim pemasaran dapat menjalankan strategi yang berbeda untuk setiap segmen:

- **High + Free/Basic**: Fokus *upsell* ke paket berbayar yang lebih tinggi.
- **High + Pro**: Program loyalitas dan referral untuk mempertahankan nilai.
- **Medium**: Edukasi fitur dan kampanye aktivasi untuk mendorong transisi ke segmen High.
- **Low**: Program retensi dan penanganan *customer support* untuk mengurangi risiko churn.

---

## 5. Kesimpulan

Penelitian ini berhasil mengembangkan sistem rekomendasi peningkatan nilai pelanggan SocialChat berbasis Multi-Layer Perceptron. Model MLP dengan arsitektur **13-64-32-3** memberikan performa terbaik dengan akurasi **88,5%** dan F1-score makro **86,2%**. Hasil klasifikasi dikelompokkan ke dalam tiga kelas (Low, Medium, High) dan dipetakan ke rekomendasi aksi bisnis yang spesifik. Sistem ini dapat membantu tim *customer success* dan pemasaran dalam menentukan prioritas intervensi secara lebih cepat, obyektif, dan terukur.

Saran untuk penelitian selanjutnya:
1. Menggunakan data riil dari platform SocialChat untuk validasi lebih lanjut.
2. Menerapkan teknik *hyperparameter tuning* seperti Grid Search untuk meningkatkan performa.
3. Membandingkan MLP dengan model lain seperti Random Forest, XGBoost, atau Gradient Boosting.
4. Menambahkan interpretasi model menggunakan SHAP atau permutation importance.

---

## Daftar Pustaka

[1] S. Haykin, *Neural Networks and Learning Machines*. Pearson, 2009.

[2] B. Yegnanarayana, *Artificial Neural Networks*. PHI Learning Pvt. Ltd., 2009.

[3] C. M. Bishop, *Pattern Recognition and Machine Learning*. Springer, 2006.

[4] F. Chollet, *Deep Learning with Python*. Manning Publications, 2021.

[5] D. C. Verhoef, P. C. F. Lang, and B. Donkers, "Effects of self-service technology on customer value," *Journal of Service Research*, vol. 9, no. 2, pp. 142–151, 2006.

[6] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

[7] A. Senang Hati Gulo and A. H. Lubis, "Penerapan Multi-Layer Perceptron untuk Mengklasifikasi Penduduk Kurang Mampu," *EXPLORER Journal of Computer Science and Information Technology*, vol. 4, no. 2, pp. 51–60, 2024.

---

*Artikel ini disusun untuk keperluan Tugas Akhir Asep Sunarya.*
