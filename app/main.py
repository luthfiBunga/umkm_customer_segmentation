
import logging
import joblib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import (
    prepare_features,
    CLUSTER_NAMES
)

# Membuat aplikasi FastAPI
app = FastAPI(
    title="UMKM Customer Segmentation API",
    description="API untuk segmentasi pelanggan UMKM menggunakan K-Means",
    version="1.0.0"
)

# Konfigurasi logging
logging.basicConfig(
    filename="prediction.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Memuat model
model = joblib.load("artifacts/model.joblib")


# Validasi input pelanggan
class CustomerData(BaseModel):
    umur: float | None = Field(default=None, ge=13, le=100)
    jumlah_transaksi_6bln: float | None = Field(default=None, ge=0)
    total_belanja_rp: float | None = Field(default=None, ge=0)
    hari_sejak_transaksi_terakhir: float | None = Field(default=None, ge=0)
    persen_pakai_promo: float | None = Field(default=None, ge=0, le=100)
    rating_rata: float | None = Field(default=None, ge=1, le=5)


@app.get("/")
def home():
    return {
        "message": "UMKM Customer Segmentation API aktif"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(customer: CustomerData):

    try:
        # Mengubah input menjadi DataFrame
        data = customer.model_dump()

        features = prepare_features(data)

        # Prediksi cluster
        prediction = model.predict(features)

        cluster = int(prediction[0])

        # Mengambil nama cluster
        cluster_name = CLUSTER_NAMES.get(
            cluster,
            "Cluster tidak dikenal"
        )

        result = {
            "cluster": cluster,
            "segment": cluster_name
        }

        # Mencatat prediksi
        logging.info(
            "Input=%s | Hasil=%s",
            data,
            result
        )

        return result

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:
        logging.exception("Terjadi kesalahan saat prediksi")

        raise HTTPException(
            status_code=500,
            detail="Terjadi kesalahan pada proses prediksi"
        )
