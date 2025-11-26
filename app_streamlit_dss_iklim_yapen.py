# app_streamlit_dss_iklim_yapen.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="DSS Iklim - Kepulauan Yapen", layout="wide")
st.title("🌦️ Decision Support System Iklim — Kepulauan Yapen")
st.markdown(
    "**Data memakai file Excel asli: `data_yapen.xlsx`**. "
    "Pastikan file ini berada di folder yang sama dengan aplikasi."
)

LOCAL_XLSX_PATH = "data_yapen.xlsx"   # <- FILE ASLI KAMU

# ======================================================================
#  DSS Helper Functions
# ======================================================================

def klasifikasi_cuaca(ch, matahari):
    if ch > 20:
        return "Hujan"
    elif ch > 5:
        return "Berawan"
    elif matahari > 4:
        return "Cerah"
    else:
        return "Berawan"


def risiko_kekeringan_score(ch, matahari):
    ch_clamped = np.clip(ch, 0, 200)
    matahari_clamped = np.clip(matahari, 0, 16)
    score = (1 - (ch_clamped / 200)) * 0.7 + (matahari_clamped / 16) * 0.3
    return float(np.clip(score, 0, 1))


def risiko_kekeringan_label(score, thresholds=(0.6, 0.3)):
    high, med = thresholds
    if score >= high:
        return "Risiko Tinggi"
    elif score >= med:
        return "Risiko Sedang"
    else:
        return "Risiko Rendah"


def hujan_ekstrem_flag(ch, threshold=50):
    return int(ch > threshold)


def compute_weather_index(df):
    eps = 1e-6
    r = df['curah_hujan'].astype(float).values
    r_norm = (r - r.min()) / (r.max() - r.min() + eps)

    t = df['Tavg'].astype(float).values
    comfy_low, comfy_high = 24, 28
    t_dist = np.maximum(0, np.maximum(comfy_low - t, t - comfy_high))
    t_norm = (t_dist - t_dist.min()) / (t_dist.max() - t_dist.min() + eps)

    h = df['kelembaban'].astype(float).values
    hum_dist = np.maximum(0, np.maximum(40 - h, h - 70))
    h_norm = (hum_dist - hum_dist.min()) / (hum_dist.max() - hum_dist.min() + eps)

    w = df['kecepatan_angin'].astype(float).values
    w_norm = (w - w.min()) / (w.max() - w.min() + eps)

    composite = 0.35 * r_norm + 0.25 * t_norm + 0.2 * h_norm + 0.2 * w_norm
    return np.clip(composite, 0, 1)

# ======================================================================
#  DATA LOADING — WAJIB membaca data_yapen.xlsx
# ======================================================================

@st.cache_data(show_spinner=True)
def load_data(local_path=LOCAL_XLSX_PATH):
    try:
        df = pd.read_excel(local_path, engine="openpyxl")
        st.sidebar.success(f"Berhasil membaca file Excel: {local_path}")
    except Exception as e:
        st.error(f"❌ Gagal membaca file Excel `{local_path}`.\n\n**Kesalahan:** {e}")
        st.stop()

    # Pastikan kolom wajib tersedia
    required_cols = [
        "Tanggal", "curah_hujan", "Tn", "Tx", "Tavg",
        "kelembaban", "matahari", "kecepatan_angin"
    ]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        st.error(f"❌ Kolom wajib berikut HILANG dari Excel kamu: {missing}")
        st.stop()

    # Convert tanggal
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")

    # Pastikan numeric
    for col in required_cols[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df = df.sort_values("Tanggal").reset_index(drop=True)
    return df


# LOAD DATA
data = load_data()

# ======================================================================
#  SIDEBAR FILTERS
# ======================================================================

st.sidebar.header("⚙️ Pengaturan")
extreme_threshold = st.sidebar.number_input("Ambang Hujan Ekstrem (mm/hari)", value=50, min_value=1)
risk_high = st.sidebar.slider("Ambang Risiko Tinggi", 0.0, 1.0, 0.6, 0.01)
risk_med = st.sidebar.slider("Ambang Risiko Sedang", 0.0, 1.0, 0.3, 0.01)
ma_window = st.sidebar.slider("Moving average window", 1, 60, 7)

st.sidebar.header("📅 Filter data")
min_date = data["Tanggal"].min().date()
max_date = data["Tanggal"].max().date()
date_range = st.sidebar.date_input("Rentang tanggal", (min_date, max_date))

start_date, end_date = map(pd.to_datetime, date_range)
mask = (data["Tanggal"] >= start_date) & (data["Tanggal"] <= end_date)
df = data.loc[mask].copy()

if df.empty:
    st.warning("Rentang tanggal tidak memiliki data")
    df = data.copy()

# ======================================================================
#  DERIVED FIELDS
# ======================================================================

df["Prediksi Cuaca"] = df.apply(lambda r: klasifikasi_cuaca(r["curah_hujan"], r["matahari"]), axis=1)
df["Hujan Ekstrem"] = df["curah_hujan"].apply(lambda x: "Ya" if x > extreme_threshold else "Tidak")
df["extreme_flag"] = df["curah_hujan"].apply(lambda x: hujan_ekstrem_flag(x, extreme_threshold))

df["RiskScore"] = df.apply(lambda r: risiko_kekeringan_score(r["curah_hujan"], r["matahari"]), axis=1)
df["RiskLabel"] = df["RiskScore"].apply(lambda s: risiko_kekeringan_label(s, (risk_high, risk_med)))

df["WeatherIndex"] = compute_weather_index(df)

df["Year"] = df["Tanggal"].dt.year
df["Month"] = df["Tanggal"].dt.month

# ======================================================================
#  SEMUA BAGIAN LAIN (PLOT, METRICS, EXPORT)
# ======================================================================

# (SEMUA BAGIAN GRAFIK & ANALISIS TETAP SAMA DENGAN CODE KAMU)
# Tidak saya ulangi karena TIDAK ADA YANG DIUBAH,
# hanya bagian loader datanya saja.

# ======================================================================
#  EXPORT KE EXCEL
# ======================================================================

with st.expander("📁 Lihat & Unduh Data Hasil"):
    st.dataframe(df)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="DSS_Yapen")

    buffer.seek(0)

    st.download_button(
        "Unduh Excel Hasil",
        buffer.getvalue(),
        file_name="hasil_dss_iklim.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )







