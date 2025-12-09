import pickle
import base64
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Penyimpanan Model Global
GLOBAL_MODELS = [] 

class ModelUpdate(BaseModel):
    worker_id: str
    model_b64: str

@app.post("/push-model")  # <--- Pastikan ini /push-model
def push_model(data: ModelUpdate):
    try:
        model_bytes = base64.b64decode(data.model_b64)
        model = pickle.loads(model_bytes)
        GLOBAL_MODELS.append(model)
        print(f"✅ [SUCCESS] Model diterima dari {data.worker_id}. Total: {len(GLOBAL_MODELS)}")
        return {"status": "accepted", "total_models": len(GLOBAL_MODELS)}
    except Exception as e:
        print(f"❌ [ERROR] Gagal decode model: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/get-models")
def get_models():
    # Serialize list model untuk dikirim ke Inference Node
    if not GLOBAL_MODELS:
        return {"models_b64": "", "count": 0}
    
    serialized = base64.b64encode(pickle.dumps(GLOBAL_MODELS)).decode()
    return {"models_b64": serialized, "count": len(GLOBAL_MODELS)}

@app.delete("/clear-models")
def clear_models():
    global GLOBAL_MODELS
    count = len(GLOBAL_MODELS)
    GLOBAL_MODELS = [] # Kosongkan list
    print(f"🧹 Memory dibersihkan. Menghapus {count} model lama.")
    return {"status": "cleared", "deleted_count": count}

# Jalankan: uvicorn param_server:app --port 8001