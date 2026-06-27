# Data Preparation untuk SocialChat Customer Uplift MLP

Panduan ini menjelaskan cara menyiapkan *data riil* dari aplikasi **SocialChat** agar bisa digunakan oleh model MLP pada repository ini.

## 1. Sumber Data di SocialChat

Data pelanggan dibagi menjadi tiga sumber utama:

### A. Tabel `users` / `subscriptions`
| Kolom | Keterangan |
|-------|------------|
| `customer_id` | UUID / user ID unik |
| `email` | Opsional, untuk identifikasi |
| `created_at` | Tanggal registrasi |
| `current_plan` | Free / Basic / Pro / Enterprise |
| `first_payment_at` | Tanggal pembayaran pertama (jika ada) |

### B. Event Log (analitik)
Setiap aktivitas user dicatat satu baris per event, contohnya:

| timestamp | customer_id | event_type | event_value | meta |
|-----------|-------------|------------|-------------|------|
| 2026-06-01 08:23 | SC_001 | login | 1 | `{}` |
| 2026-06-01 08:24 | SC_001 | message_sent | 1 | `{"channel_id":"C1"}` |
| 2026-06-01 08:30 | SC_001 | session_end | 12.4 | `{"minutes":12.4}` |
| 2026-06-01 09:00 | SC_001 | csat_rating | 5 | `{}` |
| 2026-06-02 10:00 | SC_001 | support_ticket_created | 1 | `{"category":"billing"}` |
| 2026-06-05 11:00 | SC_001 | feature_used | 1 | `{"feature":"reaction"}` |

Event yang disarankan: `login`, `message_sent`, `channel_join`, `session_end`, `csat_rating`, `support_ticket_created`, `feature_used`, `referral_completed`, `payment_received`.

### C. Revenue / Payment Log
| timestamp | customer_id | amount | plan |
|-----------|-------------|--------|------|
| 2026-01-15 | SC_001 | 150000 | Basic |
| 2026-02-15 | SC_001 | 150000 | Basic |

Dari sini bisa dihitung `lifetime_value` riil.

## 2. Feature Engineering

Setelah event log tersedia, hitung fitur untuk jendela waktu tertentu. Contoh SQL (PostgreSQL) untuk menghitung fitur 30 hari terakhir:

```sql
WITH cutoff AS (
  SELECT DATE '2026-06-30' AS reference_date
),
recent_events AS (
  SELECT *
  FROM events e, cutoff c
  WHERE e.timestamp >= c.reference_date - INTERVAL '30 days'
    AND e.timestamp < c.reference_date
)
SELECT
  u.customer_id,
  u.current_plan,
  EXTRACT(DAY FROM c.reference_date - u.created_at)::int AS account_age_days,

  COUNT(DISTINCT DATE(e.timestamp))               AS monthly_active_days,
  COUNT(*) FILTER (WHERE e.event_type = 'message_sent') AS messages_sent_per_month,
  COUNT(DISTINCT e.meta->>'channel_id')             AS channels_joined,
  COUNT(*) FILTER (WHERE e.event_type = 'support_ticket_created') AS support_tickets_last_30d,
  AVG(e.event_value) FILTER (WHERE e.event_type = 'csat_rating')  AS csat_score,
  COUNT(*) FILTER (WHERE e.event_type = 'referral_completed')       AS referral_count,
  COUNT(DISTINCT e.meta->>'feature')                AS features_used,
  AVG(e.event_value) FILTER (WHERE e.event_type = 'session_end')    AS avg_session_minutes,
  EXTRACT(DAY FROM c.reference_date - MAX(DATE(e.timestamp)))::int  AS days_since_last_login,
  COALESCE(SUM(p.amount), 0)                        AS lifetime_value
FROM users u
CROSS JOIN cutoff c
LEFT JOIN recent_events e ON e.customer_id = u.customer_id
LEFT JOIN payments p ON p.customer_id = u.customer_id AND p.timestamp < c.reference_date
GROUP BY u.customer_id, u.created_at, u.current_plan, c.reference_date;
```

## 3. Label Target (`uplift_potential`)

Target dapat dihasilkan dari perbandingan nilai pelanggan sebelum dan sesudah periode prediksi.

Contoh rule sederhana:

```sql
WITH base AS (
  -- revenue 30 hari sebelum reference_date
  SELECT customer_id, SUM(amount) AS revenue_before
  FROM payments p, cutoff c
  WHERE p.timestamp >= c.reference_date - INTERVAL '60 days'
    AND p.timestamp < c.reference_date - INTERVAL '30 days'
  GROUP BY customer_id
),
future AS (
  -- revenue 30 hari sesudah reference_date
  SELECT customer_id, SUM(amount) AS revenue_after
  FROM payments p, cutoff c
  WHERE p.timestamp >= c.reference_date
    AND p.timestamp < c.reference_date + INTERVAL '30 days'
  GROUP BY customer_id
),
growth AS (
  SELECT b.customer_id,
         COALESCE(f.revenue_after, 0) - COALESCE(b.revenue_before, 0) AS delta,
         CASE
           WHEN COALESCE(b.revenue_before, 0) = 0 THEN NULL
           ELSE (COALESCE(f.revenue_after, 0) - COALESCE(b.revenue_before, 0))::float / b.revenue_before
         END AS growth_rate
  FROM base b
  LEFT JOIN future f USING (customer_id)
)
SELECT customer_id,
       CASE
         WHEN growth_rate >= 0.30 THEN 'High'
         WHEN growth_rate >= 0.10 THEN 'Medium'
         WHEN growth_rate IS NULL AND delta <= 0 THEN 'Low'
         ELSE 'Low'
       END AS uplift_potential
FROM growth;
```

Cara lain untuk dapatkan label:
- Hasil eksperimen A/B campaign.
- Label manual dari tim Customer Success.
- Gunakan *uplift modeling* dengan kelompok treatment & control.

## 4. Dari Raw Events ke CSV Siap Training

Setelah SQL query di atas dijalankan, hasilnya disimpan sebagai CSV dengan header yang sama persis seperti `data/customers.csv` pada repo.

Selain itu, bisa menggunakan script Python `src/socialchat_uplift/ingest.py` yang menerima raw event JSON/CSV dan mengubahnya menjadi format training.

## 5. Split Data

Rekomendasi pembagian data:
- **Train**: 80% pelanggan yang paling lama/tua
- **Test**: 20% pelanggan terbaru / out-of-time split

Gunakan *stratify* pada label agar proporsi `Low/Medium/High` seimbang.

## 6. Scaling & Encoding

Repo ini sudah menangani otomatis via `preprocess.py`:
- StandardScaler untuk fitur numerik.
- OneHotEncoder untuk fitur kategorikal (`current_plan`).
- LabelEncoder untuk target.

## 7. Replacing Data Dummy dengan Data Riil

Cara paling mudah:
1. Hasilkan `customers.csv` riil dari SocialChat.
2. Ganti file `data/customers.csv` di repo.
3. Jalankan ulang: `python main.py`

Model akan otomatis melatih ulang menggunakan data riil.

## 8. Catatan Penting

- Pastikan data tidak bocor: fitur hanya boleh berisi informasi yang tersedia **sebelum** periode prediksi target.
- Jika fitur `churn_risk_score` tidak tersedia di sistem, dapat diganti dengan proxy sederhana seperti `days_since_last_login / 90`.
- Semakin banyak data historis (minimal ratusan sampel per kelas), semakin stabil performa MLP.
