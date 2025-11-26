import streamlit as st
import pandas as pd
import plotly.express as px

# Konfigurasi halaman
st.set_page_config(page_title="DSS Iklim Kepulauan Yapen", layout="wide")
st.title("🌦️ Decision Support System Iklim - Kepulauan Yapen")
st.markdown("Dashboard Prediksi Iklim berdasarkan data cuaca harian Kepulauan Yapen.")

# =======================================================================
#  BAGIAN 1 — Load Data Excel (DATA ASLI KAMU)
# =======================================================================

def load_data():
    """Membaca file Excel asli tanpa fallback ke data dummy."""
    data_path = "data_yapen.xlsx"

    try:
        df = pd.read_excel(data_path)
    except Exception as e:
        st.error(f"❌ File Excel tidak dapat dibaca. Kesalahan: {e}")
        st.stop()

    # Validasi kolom wajib
    required_cols = [
        "Tanggal",
        "curah_hujan",
        "Tavg",
        "Tn",
        "Tx",
        "kelembaban",
        "matahari",
        "kecepatan_angin"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        st.error(f"❌ Kolom berikut tidak ada di data kamu: {missing}")
        st.stop()

    # Konversi tanggal
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")

    return df

# =======================================================================
#  BAGIAN 2 — Fungsi Analisis
# =======================================================================

def klasifikasi_cuaca(ch, matahari):
    if ch > 20:
        return "Hujan"
    elif ch > 5:
        return "Berawan"
    elif matahari > 4:
        return "Cerah"
    else:
        return "Berawan"

def risiko_kekeringan(ch, matahari):
    if ch < 1 and matahari > 6:
        return "Risiko Tinggi"
    elif ch < 5:
        return "Risiko Sedang"
    else:
        return "Risiko Rendah"

def hujan_ekstrem(ch):
    return "Ya" if ch > 50 else "Tidak"

# =======================================================================
#  BAGIAN 3 — Load Data
# =======================================================================

df = load_data()

# Tambahkan kolom analisis
df["Prediksi Cuaca"] = df.apply(lambda r: klasifikasi_cuaca(r["curah_hujan"], r["matahari"]), axis=1)
df["Risiko Kekeringan"] = df.apply(lambda r: risiko_kekeringan(r["curah_hujan"], r["matahari"]), axis=1)
df["Hujan Ekstrem"] = df["curah_hujan"].apply(hujan_ekstrem)

# =======================================================================
#  BAGIAN 4 — Filter Tanggal
# =======================================================================

st.sidebar.header("📅 Filter")
selected_date = st.sidebar.date_input(
    "Pilih Tanggal",
    value=df["Tanggal"].min(),
    min_value=df["Tanggal"].min(),
    max_value=df["Tanggal"].max()
)

row = df[df["Tanggal"] == pd.to_datetime(selected_date)]

# =======================================================================
#  BAGIAN 5 — Info Harian
# =======================================================================

if not row.empty:
    info = row.iloc[0]

    st.subheader(f"📊 Data Iklim — {selected_date.strftime('%d %B %Y')}")
    st.write(f"- Suhu rata-rata: **{info['Tavg']}°C**")
    st.write(f"- Kelembaban: **{info['kelembaban']}%**")
    st.write(f"- Curah hujan: **{info['curah_hujan']} mm**")
    st.write(f"- Matahari: **{info['matahari']} jam**")
    st.write(f"- Kecepatan angin: **{info['kecepatan_angin']} km/jam**")

    st.markdown("---")
    st.subheader("🤖 Hasil Analisis Sistem")
    st.success(f"**Prediksi Cuaca:** {info['Prediksi Cuaca']}")
    st.info(f"**Risiko Kekeringan:** {info['Risiko Kekeringan']}")
    st.warning(f"**Hujan Ekstrem:** {info['Hujan Ekstrem']}")
else:
    st.error("Data tidak ditemukan untuk tanggal tersebut.")

# =======================================================================
#  BAGIAN 6 — Grafik
# =======================================================================

st.markdown("---")
st.subheader("📈 Grafik Tren Iklim")

col1, col2 = st.columns(2)

with col1:
    fig_suhu = px.line(df, x="Tanggal", y=["Tavg", "Tn", "Tx"], title="Tren Suhu Harian")
    st.plotly_chart(fig_suhu, use_container_width=True)

with col2:
    fig_hujan = px.line(df, x="Tanggal", y="curah_hujan", title="Tren Curah Hujan Harian")
    st.plotly_chart(fig_hujan, use_container_width=True)

# =======================================================================
#  BAGIAN 7 — Tabel
# =======================================================================

with st.expander("📁 Lihat Data Lengkap"):
    st.dataframe(df)

# =======================================================================
#  BAGIAN 8 — Unduh Excel
# =======================================================================

st.markdown("⬇️ **Unduh Hasil Analisis:**")
excel_path = "hasil_dss_iklim.xlsx"
df.to_excel(excel_path, index=False)
st.download_button("Download Excel", data=open(excel_path, "rb").read(), file_name=excel_path)






