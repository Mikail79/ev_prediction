# EV Population Forecasting — Washington State ⚡

> Prediksi populasi kendaraan listrik di Negara Bagian Washington menggunakan **Multivariate LSTM Neural Network** dengan data infrastruktur stasiun pengisian daya sebagai variabel tambahan.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)

---

## 📖 Deskripsi Aplikasi

Aplikasi ini merupakan sistem **peramalan (forecasting)** pertumbuhan populasi kendaraan listrik (EV) di negara bagian Washington, Amerika Serikat. Sistem ini menggunakan model **deep learning LSTM (Long Short-Term Memory)** yang dilatih pada data historis resmi dari pemerintah Washington dan data infrastruktur pengisian daya dari Departemen Energi AS.

### Apa yang Membuat Ini Berbeda?

Model ini bukan sekadar *univariate* (satu variabel). Ini adalah **Multivariate LSTM** yang mempertimbangkan **4 fitur input** sekaligus:

| # | Fitur | Sumber | Deskripsi |
|---|-------|--------|-----------|
| 1 | **EV Total** | data.wa.gov | Jumlah total kendaraan listrik terdaftar per bulan |
| 2 | **Charging Stations** | AFDC (DOE) | Jumlah kumulatif stasiun pengisian daya di WA |
| 3 | **Level 2 Ports** | AFDC (DOE) | Jumlah kumulatif port pengisian Level 2 |
| 4 | **DC Fast Ports** | AFDC (DOE) | Jumlah kumulatif port DC Fast Charging |

Dengan memasukkan data infrastruktur pengisian daya, model dapat "memahami" hubungan antara pertumbuhan infrastruktur dan pertumbuhan adopsi kendaraan listrik, menghasilkan prediksi yang lebih akurat dan kontekstual.

---

## 📊 Sumber Data

### 1. Electric Vehicle Population Size History
- **URL:** [data.wa.gov/d886-d5q2](https://data.wa.gov/Transportation/Electric-Vehicle-Population-Size-History/d886-d5q2/data_preview)
- **Penyedia:** Washington State Department of Licensing
- **Format:** CSV via Socrata API
- **Isi:** Jumlah total BEV (Battery Electric Vehicle) dan PHEV (Plug-in Hybrid) terdaftar per bulan
- **Rentang:** Januari 2017 — saat ini (~113 baris)

### 2. Alternative Fuel Stations
- **URL:** [developer.nlr.gov](https://developer.nlr.gov/api/alt-fuel-stations/v1.json)
- **Penyedia:** U.S. Department of Energy — Alternative Fuels Data Center (AFDC)
- **Format:** JSON API
- **Isi:** Data seluruh stasiun pengisian daya listrik di WA (lokasi, tanggal buka, jumlah port)
- **Total:** ~3.300+ stasiun

---

## 🧠 Metode & Arsitektur Model

### Jenis: Multivariate LSTM (Long Short-Term Memory)
- **Kategori:** Time-Series Regression (Regresi Deret Waktu)
- **Bukan** klasifikasi — model ini memprediksi nilai numerik kontinu (jumlah kendaraan), bukan kategori/kelas

### Arsitektur Neural Network
```
Layer 1: LSTM(64 units, return_sequences=True) — Input Shape: (12, 4)
Layer 2: Dropout(0.2)
Layer 3: LSTM(32 units, return_sequences=False)
Layer 4: Dropout(0.2)
Layer 5: Dense(16, activation='relu')
Layer 6: Dense(1) — Output: Prediksi EV Total
```

### Hyperparameters
| Parameter | Nilai |
|-----------|-------|
| Lookback Window | 12 bulan |
| Epochs | 100 |
| Batch Size | 16 |
| Optimizer | Adam |
| Loss Function | Mean Squared Error (MSE) |
| Train/Test Split | 80/20 |
| Scaler | MinMaxScaler (0-1) |
| Dropout Rate | 0.2 |

### Proses Training
1. Data EV dan data stasiun pengisian diunduh langsung dari API resmi
2. Kedua dataset digabung (merge) berdasarkan bulan
3. Seluruh 4 fitur dinormalisasi menggunakan MinMaxScaler
4. Data dipotong menjadi sequence dengan panjang 12 bulan (lookback)
5. Model LSTM dilatih selama 100 epoch dengan validasi pada 20% data terakhir
6. Metrik evaluasi dihitung: MAPE, MAE, RMSE

---

## 📈 Output & Metrik

### Metrik Evaluasi
- **MAPE** (Mean Absolute Percentage Error) — Persentase rata-rata kesalahan prediksi
- **MAE** (Mean Absolute Error) — Rata-rata selisih absolut antara aktual dan prediksi
- **RMSE** (Root Mean Square Error) — Akar kuadrat rata-rata kesalahan kuadrat

### File yang Dihasilkan oleh Training
| File | Deskripsi |
|------|-----------|
| `ev_model.h5` | Model LSTM terlatih (bobot neural network) |
| `scaler.pkl` | Objek MinMaxScaler yang sudah di-fit |
| `metrics.json` | Nilai MAPE, MAE, RMSE, dan info stasiun |
| `ev_multivariate.csv` | Dataset gabungan yang digunakan untuk training |
| `charging_stations.csv` | Data historis stasiun pengisian bulanan |

---

## 🎯 Manfaat & Relevansi

Aplikasi ini relevan untuk:
- **Pemerintah daerah:** Merencanakan kebutuhan infrastruktur pengisian daya masa depan
- **Perusahaan utilitas listrik:** Memperkirakan peningkatan permintaan listrik
- **Investor & analis:** Memahami tren pasar kendaraan listrik
- **Akademisi & peneliti:** Studi korelasi antara infrastruktur dan adopsi teknologi

---

## 🚀 Cara Menjalankan

### Prasyarat
- Python 3.10+
- Koneksi internet (opsional, untuk mengambil data terbaru dari API)

### 1. Clone Repository
```bash
git clone https://github.com/Mikail79/ev_prediction.git
cd ev_prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Unduh Data Stasiun Pengisian (Pertama Kali)
```bash
python fetch_charging_data.py
```

### 4. Latih Model
```bash
python train_model.py
```

### 5. Jalankan Dashboard
```bash
streamlit run app.py
```

Dashboard akan terbuka di browser pada `http://localhost:8501`

---

## 📁 Struktur Proyek

```
ev_prediction/
├── app.py                    # Dashboard Streamlit (UI)
├── train_model.py            # Script training model (Python)
├── train_model.ipynb         # Notebook training (Jupyter)
├── fetch_charging_data.py    # Script pengambilan data stasiun pengisian
├── ev_population.csv         # Data historis populasi EV (backup lokal)
├── charging_stations.csv     # Data historis stasiun pengisian (backup lokal)
├── ev_multivariate.csv       # Dataset gabungan (dihasilkan saat training)
├── ev_model.h5               # Model LSTM terlatih
├── scaler.pkl                # Scaler yang sudah di-fit
├── metrics.json              # Metrik evaluasi model
├── requirements.txt          # Daftar dependensi Python
└── README.md                 # Dokumentasi ini
```

---

## 🔄 Pembaruan Data

Aplikasi ini dirancang untuk **auto-sync** dengan API resmi:
- **Dashboard (`app.py`):** Secara otomatis mengambil data terbaru dari API setiap 24 jam
- **Training (`train_model.py`):** Mengambil data terbaru dari kedua API saat dijalankan
- **Fallback:** Jika API tidak tersedia, sistem menggunakan file CSV lokal sebagai cadangan

---

## 📝 Lisensi

Proyek ini dibuat untuk keperluan edukasi dan penelitian.
Data bersumber dari API publik pemerintah AS (data.wa.gov & AFDC).
