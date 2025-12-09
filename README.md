

# 🧠 Distributed Random Forest System

Sistem pembelajaran mesin terdistribusi (*Distributed Machine Learning*) berbasis Python yang mengimplementasikan arsitektur **Parameter Server**. Sistem ini memungkinkan pelatihan model *Random Forest* secara paralel melalui banyak *worker node*, dengan fitur *Dynamic Sharding*, *Auto-Discovery*, dan *Ensemble Voting*.

Dibuat sebagai Tugas Besar Mata Kuliah **Sistem Terdistribusi**.

---

## 🚀 Fitur Utama

* **Dynamic Worker Discovery** — Worker otomatis mendaftarkan diri ke Coordinator (*plug-and-play*).
* **Automatic Data Sharding** — Dataset CSV dipecah otomatis dan didistribusikan merata ke seluruh worker.
* **Robust Preprocessing Pipeline** — Menangani NaN, missing values, dan label encoding secara otomatis.
* **Parameter Server Architecture** — Model worker disimpan secara terpusat dan dapat di-reset antar sesi.
* **Ensemble Soft Voting** — Inference dilakukan dengan voting dari seluruh model worker.
* **Interactive Web Dashboard** — Upload CSV, monitoring worker, log training, akurasi, dan confusion matrix.

---

## 🛠️ Arsitektur Sistem

Sistem terdiri dari 4 node yang berjalan pada port berbeda:

| Node Role            | File              | Port                | Fungsi                                              |
| -------------------- | ----------------- | ------------------- | --------------------------------------------------- |
| **Coordinator**      | `coordinator.py`  | `8000`              | UI gateway, preprocessing, sharding, load balancing |
| **Parameter Server** | `param_server.py` | `8001`              | Penyimpanan model global & state reset              |
| **Worker(s)**        | `worker.py`       | `8002`, `8003`, ... | Pelatihan model Random Forest dan push model ke PS  |
| **Inference Node**   | `inference.py`    | `8004`              | Ensemble voting, evaluasi akurasi, confusion matrix |

---

## 📋 Prasyarat

* Python **3.9+**
* Install semua dependensi:

```bash
pip install fastapi uvicorn pandas numpy scikit-learn requests python-multipart
```

---

## ▶️ Cara Menjalankan Sistem

Setiap komponen dijalankan pada terminal berbeda.
Ikuti urutan agar tidak terjadi error.

---

### 1️⃣ Jalankan Coordinator (PORT 8000)

```bash
uvicorn coordinator:app --port 8000
```

---

### 2️⃣ Jalankan Parameter Server (PORT 8001)

```bash
uvicorn param_server:app --port 8001
```

---

### 3️⃣ Jalankan Inference Node (PORT 8004)

```bash
uvicorn inference:app --port 8004
```

---

### 4️⃣ Jalankan Worker Node (PORT 8002, 8003, dst)

Worker 1:

```bash
python worker.py 8002
```

Worker 2 (opsional):

```bash
python worker.py 8003
```

Worker dapat diperbanyak sesuai kebutuhan.
**Coordinator harus dijalankan terlebih dahulu** agar worker bisa melakukan auto-registration.

---

