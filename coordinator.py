import pandas as pd
import requests
import io
import numpy as np
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pydantic import BaseModel

app = FastAPI()

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
TEST_DATA = {"X": [], "y": []}
WORKERS = set()

class WorkerRegistration(BaseModel):
    worker_url: str

@app.get("/")
def read_index():
    return FileResponse("index.html")

@app.post("/register-worker")
def register_worker(data: WorkerRegistration):
    WORKERS.add(data.worker_url)
    print(f"📡 Worker Baru Terdaftar: {data.worker_url}. Total: {len(WORKERS)}")
    return {"status": "registered", "total_workers": len(WORKERS)}

@app.post("/upload-csv")
async def upload_csv(file: UploadFile, target: str = Form(...)):
    global TEST_DATA
    
    if not WORKERS:
        return {"error": "❌ Tidak ada Worker aktif! Jalankan 'python worker.py 8002' dulu."}

    # 1. Baca File ke Memory
    content = await file.read()
    decoded_content = content.decode('utf-8')
    
    # Deteksi Separator
    first_line = decoded_content.split('\n')[0]
    separator = ';' if ';' in first_line else ','
    
    print(f"📂 Mencoba baca CSV dengan separator: '{separator}'")

    # 2. Coba Load DataFrame
    try:
        df = pd.read_csv(io.StringIO(decoded_content), sep=separator)
        df.columns = df.columns.str.strip() # Bersihkan nama kolom
    except Exception as e:
        return {"error": f"Gagal baca CSV: {str(e)}"}

    # --- DEBUGGING AMAN (Hanya jalan jika df sukses dibuat) ---
    print(f"🔄 DATASET DITERIMA! Ukuran: {len(df)} baris")
    print(f"   - Kolom: {list(df.columns)}")
    # ---------------------------------------------------------

    # 3. Validasi Target
    if target not in df.columns:
        return {"error": f"Kolom target '{target}' tidak ditemukan. Kolom tersedia: {list(df.columns)}"}

    # ==========================================
    # 🧹 DATA CLEANING "ANTI-PELURU"
    # ==========================================
    
    # Pisahkan Fitur (X) dan Target (y)
    X_raw = df.drop(columns=[target])
    
    # a. HANYA ambil kolom angka untuk Fitur (Buang kolom teks/nama/gender di fitur)
    X_numeric = X_raw.select_dtypes(include=[np.number])
    
    # b. Isi Data Kosong (NaN) dengan 0
    X_numeric = X_numeric.fillna(0)
    X = X_numeric.values # Convert ke Numpy

    # c. Proses Target (y)
    y_raw = df[target]
    
    # Jika target berupa Teks (misal: 'Female', 'Male'), ubah jadi Angka (0, 1)
    if y_raw.dtype == 'object' or y_raw.dtype == 'string':
        print(f"🔄 Encoding Target '{target}' dari Teks ke Angka...")
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
    else:
        # Jika target sudah angka, pastikan tidak ada NaN dan jadi Integer
        y = y_raw.fillna(0).values.astype(int)

    # 4. Reset Model Lama
    try:
        requests.delete("http://localhost:8001/clear-models", timeout=1)
    except: pass

    # 5. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Simpan Test Data (Convert ke pure Python list agar JSON aman)
    TEST_DATA = {
        "X": [[float(val) for val in row] for row in X_test],
        "y": [int(val) for val in y_test]
    }

    # 6. SHARDING & KIRIM (Paranoid Mode)
    active_workers_list = list(WORKERS)
    num_workers = len(active_workers_list)
    
    X_chunks = np.array_split(X_train, num_workers)
    y_chunks = np.array_split(y_train, num_workers)
    
    print(f"📦 Mengirim tugas ke {num_workers} worker...")

    for i, worker_url in enumerate(active_workers_list):
        # KONVERSI MANUAL KE PYTHON LIST (PENTING UNTUK MENGHINDARI ERROR 422)
        # Kita paksa setiap angka jadi 'float' atau 'int' murni Python
        
        safe_features = [[float(x) for x in row] for row in X_chunks[i]]
        safe_targets = [int(x) for x in y_chunks[i]]
        
        chunk_data = {
            "worker_id": f"Worker-{i+1}",
            "features": safe_features,
            "targets": safe_targets
        }
        
        try:
            res = requests.post(f"{worker_url}/train", json=chunk_data, timeout=60)
            
            # --- CEK ERROR 422 DISINI ---
            if res.status_code != 200:
                print(f"⚠️ Worker {worker_url} MENOLAK Data ({res.status_code})")
                print(f"   Pesan Error: {res.text}") # <--- INI KUNCINYA
            else:
                print(f"✅ Sukses kirim ke {worker_url}")
                
        except Exception as e:
            print(f"⚠️ Gagal koneksi ke {worker_url}: {e}")

    return {
        "info": f"Training Sukses! Data dikirim ke {num_workers} Worker.",
        "active_workers": num_workers,
        "columns_used": list(X_numeric.columns),
        "target": target
    }

@app.get("/check-accuracy")
def check_accuracy():
    if not TEST_DATA["X"]:
        return {"accuracy": 0, "message": "Upload data dulu"}
    try:
        payload = {"features": TEST_DATA["X"], "targets": TEST_DATA["y"]}
        res = requests.post("http://localhost:8004/predict_accuracy", json=payload)
        return res.json()
    except:
        return {"accuracy": 0, "message": "Inference Node Mati"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)