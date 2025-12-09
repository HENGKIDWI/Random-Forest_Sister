# 🧠 Distributed Random Forest System

Sistem pembelajaran mesin terdistribusi (*Distributed Machine Learning*) berbasis Python yang mengimplementasikan arsitektur **Parameter Server**. Sistem ini memungkinkan pelatihan model *Random Forest* secara paralel pada banyak *worker node*, dengan fitur *Dynamic Sharding*, *Auto-Discovery*, dan *Ensemble Voting*.

Dibuat sebagai Tugas Besar Mata Kuliah Sistem Terdistribusi.

---

## 🚀 Fitur Utama

* **Dynamic Discovery:** Worker otomatis mendaftarkan diri ke Coordinator saat dinyalakan (*Plug-and-Play*).
* **Auto Data Sharding:** Dataset CSV besar dipecah secara otomatis dan didistribusikan rata ke seluruh worker aktif.
* **Robust Preprocessing:** Menangani data kotor secara otomatis (mengisi nilai kosong/NaN dan konversi teks ke angka/Label Encoding).
* **Parameter Server Architecture:** Sentralisasi penyimpanan model dengan mekanisme *State Reset* antar sesi training.
* **Ensemble Voting:** Prediksi akhir dilakukan menggunakan *Soft Voting* dari seluruh model yang tersebar.
* **Real-time Dashboard:** Antarmuka Web (HTML/Tailwind) untuk upload data dan memantau akurasi serta *Confusion Matrix*.

---

## 🛠️ Arsitektur Sistem

Sistem ini terdiri dari 4 komponen utama yang berjalan pada port berbeda:

| Node Role | File Python | Port Default | Fungsi Utama |
| :--- | :--- | :--- | :--- |
| **Coordinator** | `coordinator.py` | `8000` | Gateway User, Data Cleaning, Sharding, Load Balancer. |
| **Parameter Server** | `param_server.py` | `8001` | Model Registry (Menyimpan & Menghapus Model). |
| **Worker(s)** | `worker.py` | `8002`, `8003`, ... | Training Model (Random Forest) & Push Model ke PS. |
| **Inference Node** | `inference.py` | `8004` | Validasi, Voting, & Perhitungan Akurasi. |

---

## 📋 Prasyarat

Pastikan Python 3.9+ sudah terinstall. Install library yang dibutuhkan:

```bash
pip install fastapi uvicorn pandas scikit-learn requests python-multipart


# **running**
uvicorn coordinator:app --port 8000
uvicorn param_server:app --port 8001
uvicorn inference:app --port 8004

python worker.py 8002
python worker.py 8003

woker bisa diperbanyak sesuai kebutuhan, tapi harus dijalankan setelah coordinator berjalan
