import requests
import pickle
import base64
import pandas as pd
import sys
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

# --- CONFIG ---
COORDINATOR_URL = "http://localhost:8000"
PARAM_SERVER_URL = "http://localhost:8001/push-model"

# Tentukan Port dari argumen terminal (Contoh: python worker.py 8002)
# Jika tidak ada argumen, default ke 8002
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8002
MY_URL = f"http://localhost:{PORT}"

# --- EVENT: SAAT WORKER NYALA ---
@app.on_event("startup")
def startup_event():
    """Otomatis lapor ke Coordinator bahwa saya hidup"""
    print(f"🚀 Worker berjalan di Port {PORT}")
    try:
        print(f"📞 Menghubungi Coordinator ({COORDINATOR_URL})...")
        requests.post(f"{COORDINATOR_URL}/register-worker", json={"worker_url": MY_URL})
        print("✅ Berhasil terdaftar di Coordinator!")
    except Exception as e:
        print(f"❌ Gagal connect ke Coordinator: {e}")
        print("   -> Pastikan coordinator.py sudah jalan duluan!")

# --- MODEL DATA ---
class TrainPayload(BaseModel):
    worker_id: str
    features: list[list[float]]
    targets: list[int]

# --- ENDPOINT TRAINING ---
@app.post("/train")
def train(payload: TrainPayload):
    print(f"⚙️ [{payload.worker_id}] Menerima tugas: {len(payload.targets)} data.")

    # 1. Konversi data JSON ke Pandas
    X = pd.DataFrame(payload.features)
    y = pd.Series(payload.targets)
    
    # 2. Training Random Forest (Versi Mini / Weak Learner)
    # Kita buat n_estimators kecil (10) karena ini distributed ensemble
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)

    # 3. Bungkus Model jadi Paket (Pickle -> Base64)
    model_bytes = pickle.dumps(clf)
    model_b64 = base64.b64encode(model_bytes).decode()

    # 4. Setor Model ke Parameter Server
    try:
        requests.post(PARAM_SERVER_URL, json={
            "worker_id": payload.worker_id,
            "model_b64": model_b64
        })
        print(f"✅ [{payload.worker_id}] Model dikirim ke Parameter Server.")
    except Exception as e:
        print(f"❌ [{payload.worker_id}] Gagal lapor ke PS: {e}")

    return {"status": "success"}

# --- MAIN BLOCK ---
if __name__ == "__main__":
    # Jalankan Uvicorn sesuai PORT yang didapat dari sys.argv
    uvicorn.run(app, host="0.0.0.0", port=PORT)