import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ===========================================
# KONFIGURASI AWAL
# ===========================================
st.set_page_config(page_title="Dashboard Cuaca Yapen", layout="wide")

LOCAL_PATH = "/mnt/data/KYAPEN.xlsx"

# ===========================================
# FUNGSI MEMUAT DATA
# ===========================================
@st.cache_data
def load_data(path=LOCAL_PATH):
    try:
        df = pd.read_excel(path)

        # Pastikan kolom tanggal terbaca
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')

        # Gabungkan jika ada 2 kolom kecepatan angin
        if 'Kecepatan_angin' in df.columns and 'kecepatan_angin' in df.columns:
            df['kecepatan_angin'] = df['Kecepatan_angin'].fillna(df['kecepatan_angin'])
        elif 'Kecepatan_angin' in df.columns:
            df.rename(columns={'Kecepatan_angin': 'kecepatan_angin'}, inplace=True)

        # Jika kolom yang dibutuhkan tidak ada → isi default
        required_cols = ['Tn', 'Tx', 'Tavg', 'kelembaban', 'curah_hujan', 'matahari', 'kecepatan_angin']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0

        # Bersihkan data numeric
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        df = df.sort_values("Tanggal").reset_index(drop=True)
        return df

    except Exception as e:
        st.error(f"Gagal memuat file Excel: {e}")
        return pd.DataFrame()

# ===========================================
# LOAD DATA
# ===========================================
df = load_data()

st.title("📊 Dashboard Data Cuaca - Kabupaten Yapen")

if df.empty:
    st.stop()

# ===========================================
# SIDEBAR FILTER
# ===========================================
st.sidebar.header("Filter Data")

tahun_list = sorted(df['Tanggal'].dt.year.dropna().unique())
tahun = st.sidebar.selectbox("Pilih Tahun", tahun_list)

df_filtered = df[df['Tanggal'].dt.year == tahun]

# ===========================================
# METRIC RINGKASAN
# ===========================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rata-rata Suhu (°C)", f"{df_filtered['Tavg'].mean():.2f}")
col2.metric("Total Curah Hujan (mm)", f"{df_filtered['curah_hujan'].sum():.1f}")
col3.metric("Rata-rata Kelembaban (%)", f"{df_filtered['kelembaban'].mean():.1f}")
col4.metric("Rata-rata Kecepatan Angin", f"{df_filtered['kecepatan_angin'].mean():.1f}")

# ===========================================
# GRAFIK SUHU
# ===========================================
st.subheader("📈 Grafik Suhu Harian")

chart_temp = (
    alt.Chart(df_filtered)
    .mark_line()
    .encode(
        x="Tanggal:T",
        y=alt.Y("Tavg:Q", title="Suhu (°C)"),
        tooltip=["Tanggal", "Tn", "Tx", "Tavg"]
    )
    .properties(height=300)
)

st.altair_chart(chart_temp, use_container_width=True)

# ===========================================
# GRAFIK CURAH HUJAN
# ===========================================
st.subheader("🌧 Curah Hujan Harian")

chart_hujan = (
    alt.Chart(df_filtered)
    .mark_bar()
    .encode(
        x="Tanggal:T",
        y=alt.Y("curah_hujan:Q", title="Curah Hujan (mm)"),
        tooltip=["Tanggal", "curah_hujan"]
    )
    .properties(height=300)
)

st.altair_chart(chart_hujan, use_container_width=True)

# ===========================================
# GRAFIK KELEMBABAN & MATAHARI
# ===========================================
st.subheader("🌤 Kelembaban & Lama Penyinaran Matahari")

colA, colB = st.columns(2)

with colA:
    chart_humidity = (
        alt.Chart(df_filtered)
        .mark_line()
        .encode(
            x="Tanggal:T",
            y=alt.Y("kelembaban:Q", title="Kelembaban (%)"),
            tooltip=["Tanggal", "kelembaban"]
        )
        .properties(height=300)
    )
    st.altair_chart(chart_humidity, use_container_width=True)

with colB:
    chart_matahari = (
        alt.Chart(df_filtered)
        .mark_line()
        .encode(
            x="Tanggal:T",
            y=alt.Y("matahari:Q", title="Jam Matahari"),
            tooltip=["Tanggal", "matahari"]
        )
        .properties(height=300)
    )
    st.altair_chart(chart_matahari, use_container_width=True)

# ===========================================
# DATAFRAME
# ===========================================
st.subheader("📋 Data Lengkap")
st.dataframe(df_filtered, use_container_width=True)
