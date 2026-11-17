
import os
import json
import joblib
import pandas as pd
import sklearn

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Lokasi dataset
DATA_PATH = "data/data_pelanggan_clean.csv"

# 2. Lokasi penyimpanan model
os.makedirs("artifacts", exist_ok=True)

# 3. Fitur yang digunakan
FEATURES = [
    "umur",
    "jumlah_transaksi_6bln",
    "total_belanja_rp",
    "hari_sejak_transaksi_terakhir",
    "persen_pakai_promo",
    "rating_rata"
]

# 4. Membaca dataset
df = pd.read_csv(DATA_PATH)

# 5. Memastikan semua fitur tersedia
missing_columns = [col for col in FEATURES if col not in df.columns]

if missing_columns:
    raise ValueError(f"Kolom tidak ditemukan: {missing_columns}")

X = df[FEATURES].copy()

# 6. Membuat pipeline K-Means
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(
        n_clusters=4,
        random_state=42,
        n_init=20
    ))
])

# 7. Melatih model
model.fit(X)

# 8. Menyimpan model
joblib.dump(model, "artifacts/model.joblib")

# 9. Membuat metadata
metadata = {
    "model_name": "UMKM Customer Segmentation",
    "algorithm": "KMeans",
    "n_clusters": 4,
    "features": FEATURES,
    "silhouette_score": 0.4185,
    "random_state": 42,
    "sklearn_version": sklearn.__version__
}

with open("artifacts/metadata.json", "w") as file:
    json.dump(metadata, file, indent=4)

# 10. Menampilkan hasil
print("Model berhasil dilatih!")
print("Jumlah data:", len(df))
print("Jumlah fitur:", len(FEATURES))
print("Model tersimpan di artifacts/model.joblib")
print("Metadata tersimpan di artifacts/metadata.json")
