import pickle
import base64
import requests
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.metrics import confusion_matrix

app = FastAPI()

class PredictPayload(BaseModel):
    features: list[list[float]]
    targets: list[int] = []

@app.post("/predict_accuracy")
def predict_accuracy(payload: PredictPayload):
    # 1. Ambil Model dari PS
    try:
        r = requests.get("http://localhost:8001/get-models")
        data = r.json()
        if data['count'] == 0:
            return {"accuracy": 0, "message": "Belum ada model di PS"}
        
        # Decode models
        models = pickle.loads(base64.b64decode(data['models_b64']))
    except Exception as e:
        print(f"❌ Error fetch model: {e}")
        return {"accuracy": 0, "message": "Gagal ambil model"}

    X = payload.features
    y_true = payload.targets

    if not X:
        return {"accuracy": 0, "message": "Data kosong"}

    # 2. ENSEMBLE VOTING (VERSI DINAMIS / MULTI-CLASS)
    # Kita tidak lagi hardcode 2 kelas. Kita cek output model pertama.
    
    try:
        # Cek probabilitas dari model pertama untuk tahu dimensi array
        first_probs = models[0].predict_proba(X)
        num_classes = first_probs.shape[1]
        
        # Buat penampung probabilitas rata-rata
        avg_probs = np.zeros((len(X), num_classes))
        valid_models = 0

        for model in models:
            probs = model.predict_proba(X)
            
            # Pastikan model ini punya jumlah kelas yang sama dengan model pertama
            # (Kadang worker A melihat 3 kelas, worker B cuma lihat 2 kelas -> ini bisa bikin error)
            if probs.shape[1] == num_classes:
                avg_probs += probs
                valid_models += 1
            else:
                # Handle kasus langka: padding jika kelas kurang
                # (Kompleks, untuk demo kita skip model yang beda dimensi)
                print(f"⚠️ Model di-skip karena dimensi beda: {probs.shape[1]} vs {num_classes}")

        if valid_models == 0:
            return {"accuracy": 0, "message": "Tidak ada model valid"}

        # Hitung Rata-rata
        avg_probs /= valid_models
        
        # Ambil kelas dengan probabilitas tertinggi
        y_pred = np.argmax(avg_probs, axis=1)

        # 3. Hitung Akurasi
        correct = np.sum(y_pred == y_true)
        accuracy = (correct / len(y_true)) * 100
        
        # 4. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Print Debug di Terminal Inference biar kelihatan
        print(f"📊 Prediksi Selesai. Akurasi: {accuracy:.2f}% (Dari {valid_models} model)")

        return {
            "accuracy": round(accuracy, 2), 
            "total_models_voting": valid_models,
            "confusion_matrix": cm.tolist()
        }

    except Exception as e:
        print(f"❌ CRASH saat Voting: {e}")
        # Return error agar kelihatan di frontend/log coordinator
        return {"accuracy": 0, "message": f"Voting Error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)