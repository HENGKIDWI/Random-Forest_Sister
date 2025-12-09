🧠 Distri-Brain: Distributed Random Forest System

Sistem pembelajaran mesin terdistribusi sederhana menggunakan algoritma Random Forest. Sistem ini mendemonstrasikan konsep Parallel Computing dan Distributed Systems dengan membagi tugas pelatihan (training) ke beberapa Worker, mengumpulkan model di Parameter Server, dan melakukan prediksi melalui Inference Node.

📋 Prasyarat (Requirements)

Pastikan Python 3.8+ sudah terinstal di komputer Anda.

Install library yang dibutuhkan dengan menjalankan perintah berikut di terminal:

pip install fastapi uvicorn pandas numpy scikit-learn requests python-multipart


🚀 Cara Menjalankan (How to Run)

Sistem ini terdiri dari 4 komponen yang harus dijalankan secara bersamaan pada terminal yang berbeda. Ikuti urutan di bawah ini agar sistem berjalan lancar:

1. Jalankan Parameter Server (Terminal 1)

Server ini bertugas menyimpan model global yang dikirim oleh worker.

python param_server.py
# Berjalan di Port 8001


2. Jalankan Coordinator (Terminal 2)

Server utama yang melayani antarmuka web (Frontend) dan membagi dataset (sharding).

python coordinator.py
# Berjalan di Port 8000


3. Jalankan Inference Node (Terminal 3)

Node ini bertugas mengambil model dari Parameter Server dan melakukan pengujian akurasi (Ensemble Voting).

python inference.py
# Berjalan di Port 8004


4. Jalankan Worker (Terminal 4, 5, dst)

Worker adalah unit yang melakukan pelatihan model. Anda bisa menjalankan satu atau lebih worker untuk melihat efek distribusi beban.

Worker 1:

python worker.py 8002


Worker 2 (Opsional - Buka Terminal Baru):

python worker.py 8003


Catatan: Pastikan coordinator.py sudah menyala sebelum menjalankan worker.py agar worker bisa mendaftarkan diri secara otomatis.

💻 Cara Penggunaan

Buka browser dan akses http://localhost:8000.

Cek indikator Active Workers di pojok kanan atas. Pastikan jumlahnya sesuai dengan worker yang Anda jalankan.

Upload Dataset:

Klik "Choose File" dan pilih file CSV dataset Anda (misalnya: dataset penyakit jantung, diabetes, dll).

Isi kolom Target dengan nama kolom yang ingin diprediksi (contoh: target, outcome, cardio, diagnosis). Nama ini Case-Sensitive!

Klik tombol 🚀 Distribute & Train.

Lihat prosesnya:

Log: Menampilkan status pengiriman data ke worker.

Performance Metric: Menampilkan akurasi global setelah semua worker selesai melatih dan model digabungkan.

Confusion Matrix: Detail prediksi Benar vs Salah.

📂 Struktur Project

File

Deskripsi

Port

coordinator.py

Backend utama, UI Server, & Sharding Data

8000

param_server.py

Penyimpanan model global (State Manager)

8001

worker.py

Unit pemroses training (Distributed Node)

8002++

inference.py

Unit evaluasi & prediksi (Aggregator)

8004

index.html

Antarmuka Dashboard (Frontend)

-

⚠️ Troubleshooting

Error "Address already in use":
Artinya port sedang dipakai. Pastikan tidak ada terminal lain yang sedang menjalankan script yang sama. Matikan terminal lama dengan Ctrl+C.

Worker tidak terdeteksi:
Matikan worker.py (Ctrl+C), lalu jalankan lagi. Worker perlu handshake ulang dengan Coordinator.

Akurasi 0% atau Error:

Cek nama kolom target. Apakah Anda mengetik Target padahal di CSV namanya target (huruf kecil)?

Pastikan dataset bersih (walaupun sistem sudah memiliki fitur auto-cleaning dasar).
