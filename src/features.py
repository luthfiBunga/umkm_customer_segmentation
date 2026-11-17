
import pandas as pd

# Fitur yang digunakan model
FEATURES = [
    "umur",
    "jumlah_transaksi_6bln",
    "total_belanja_rp",
    "hari_sejak_transaksi_terakhir",
    "persen_pakai_promo",
    "rating_rata"
]

# Nama setiap cluster berdasarkan hasil analisis
CLUSTER_NAMES = {
    0: "Pelanggan Loyal dan Bernilai Tinggi",
    1: "Pelanggan Aktif dan Sensitif Promo",
    2: "Pelanggan Tidak Aktif / Berisiko Churn",
    3: "Pelanggan Belanja Tinggi dengan Frekuensi Rendah"
}

def prepare_features(data):
    """
    Mengubah input pelanggan menjadi DataFrame
    dengan urutan fitur yang sesuai model.
    """

    if isinstance(data, dict):
        data = [data]

    df = pd.DataFrame(data)

    # Memastikan semua fitur tersedia
    missing = [col for col in FEATURES if col not in df.columns]

    if missing:
        raise ValueError(
            f"Fitur berikut belum diisi: {missing}"
        )

    # Mengambil fitur sesuai urutan model
    df = df[FEATURES].copy()

    # Memastikan semua data berupa angka
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Validasi nilai yang tidak masuk akal
    if (df["umur"].dropna() < 13).any() or \
       (df["umur"].dropna() > 100).any():
        raise ValueError("Umur harus berada pada rentang 13–100 tahun.")

    if (df["jumlah_transaksi_6bln"].dropna() < 0).any():
        raise ValueError("Jumlah transaksi tidak boleh negatif.")

    if (df["total_belanja_rp"].dropna() < 0).any():
        raise ValueError("Total belanja tidak boleh negatif.")

    if (df["hari_sejak_transaksi_terakhir"].dropna() < 0).any():
        raise ValueError("Hari sejak transaksi tidak boleh negatif.")

    if ((df["persen_pakai_promo"].dropna() < 0) |
        (df["persen_pakai_promo"].dropna() > 100)).any():
        raise ValueError("Persentase promo harus 0–100.")

    if ((df["rating_rata"].dropna() < 1) |
        (df["rating_rata"].dropna() > 5)).any():
        raise ValueError("Rating harus berada pada rentang 1–5.")

    return df
