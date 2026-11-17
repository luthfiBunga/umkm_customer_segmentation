
import streamlit as st
import requests

st.set_page_config(
    page_title="Segmentasi Pelanggan UMKM",
    page_icon="📊",
    layout="centered"
)

API_URL = "http://127.0.0.1:8000/predict"

st.title("📊 Segmentasi Pelanggan UMKM")
st.write(
    "Sistem pengelompokan pelanggan UMKM "
    "menggunakan algoritma K-Means."
)

st.divider()

st.subheader("Input Data Pelanggan")

with st.form("form_pelanggan"):

    umur = st.number_input(
        "Umur pelanggan",
        min_value=13,
        max_value=100,
        value=27
    )

    transaksi = st.number_input(
        "Jumlah transaksi selama 6 bulan",
        min_value=0,
        value=10
    )

    belanja = st.number_input(
        "Total belanja (Rp)",
        min_value=0,
        value=500000
    )

    recency = st.number_input(
        "Hari sejak transaksi terakhir",
        min_value=0,
        value=10
    )

    promo = st.number_input(
        "Persentase penggunaan promo (%)",
        min_value=0.0,
        max_value=100.0,
        value=20.0
    )

    rating = st.number_input(
        "Rating rata-rata pelanggan",
        min_value=1.0,
        max_value=5.0,
        value=4.0,
        step=0.1
    )

    submit = st.form_submit_button(
        "Prediksi Segmentasi Pelanggan",
        use_container_width=True
    )

if submit:

    data = {
        "umur": umur,
        "jumlah_transaksi_6bln": transaksi,
        "total_belanja_rp": belanja,
        "hari_sejak_transaksi_terakhir": recency,
        "persen_pakai_promo": promo,
        "rating_rata": rating
    }

    try:
        with st.spinner("Memproses prediksi..."):

            response = requests.post(
                API_URL,
                json=data,
                timeout=15
            )

        if response.status_code == 200:

            result = response.json()

            st.success("Prediksi berhasil!")

            st.subheader("Hasil Segmentasi")

            st.metric(
                "Cluster Pelanggan",
                result["cluster"]
            )

            st.info(
                "Segmen: " + result["segment"]
            )

            st.write("Data pelanggan:")
            st.json(data)

        else:
            st.error(
                f"Prediksi gagal: {response.text}"
            )

    except requests.exceptions.ConnectionError:
        st.error(
            "API tidak aktif. Pastikan FastAPI berjalan."
        )

    except requests.exceptions.RequestException as e:
        st.error(f"Terjadi kesalahan: {e}")

st.divider()

st.caption(
    "Machine Learning | K-Means | 4 Cluster"
)
