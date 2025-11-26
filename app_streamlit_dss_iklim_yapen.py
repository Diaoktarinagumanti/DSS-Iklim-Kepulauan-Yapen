# app_streamlit_dss_iklim_yapen.py

import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ---------- CONFIG ----------
st.set_page_config(page_title="DSS Iklim - Kepulauan Yapen", layout="wide")
st.title("🌦️ Decision Support System Iklim — Kepulauan Yapen")
st.markdown("Dashboard prediksi & analisis iklim menggunakan file **data_yapen.xlsx**.")

LOCAL_XLSX_PATH = "data_yapen.xlsx"


# ---------- DSS FUNCTIONS ----------
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
    ch = np.clip(ch, 0, 200)
    matahari = np.clip(matahari, 0, 12)
    score = (1 - ch/200) * 0.7 + (matahari/12) * 0.3
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
    r = df["curah_hujan"].values
    r_norm = (r - r.min()) / (r.max() - r.min() + eps)

    t = df["Tavg"].values
    comfy_low, comfy_high = 24, 28
    t_dist = np.maximum(0, np.maximum(comfy_low - t, t - comfy_high))
    t_norm = (t_dist - t_dist.min()) / (t_dist.max() - t_dist.min() + eps)

    h = df["kelembaban"].values
    hum_dist = np.maximum(0, np.maximum(40 - h, h - 70))
    h_norm = (hum_dist - hum_dist.min()) / (hum_dist.max() - hum_dist.min() + eps)

    w = df["kecepatan_angin"].values
    w_norm = (w - w.min()) / (w.max() - w.min() + eps)

    return 0.35*r_norm + 0.25*t_norm + 0.2*h_norm + 0.2*w_norm



# === EXPORT EXCEL TANPA ERROR ===
import pandas as pd
import streamlit as st
from io import BytesIO

def export_excel(df):
    buffer = BytesIO()

    # Gunakan engine default (openpyxl). Aman dan tidak butuh instal library tambahan.
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Data_Iklim")

    st.download_button(
        label="⬇️ Download Data dalam Excel",
        data=buffer.getvalue(),
        file_name="data_iklim.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# ---------- SIDEBAR ----------
st.sidebar.header("⚙️ Pengaturan")
extreme_threshold = st.sidebar.number_input("Ambang Hujan Ekstrem (mm)", value=50, min_value=1)
risk_high = st.sidebar.slider("Ambang Risiko Tinggi", 0.0, 1.0, 0.6, 0.01)
risk_med = st.sidebar.slider("Ambang Risiko Sedang", 0.0, 1.0, 0.3, 0.01)
ma_window = st.sidebar.slider("Moving Average (hari)", 1, 60, 7)

# Filter tanggal
st.sidebar.header("📅 Filter Data")
min_date = data["tanggal"].min().date()
max_date = data["tanggal"].max().date()
date_range = st.sidebar.date_input("Rentang Tanggal", (min_date, max_date))

start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
df = data[(data["tanggal"] >= start) & (data["tanggal"] <= end)].copy()


# ---------- FEATURE ENGINEERING ----------
df["prediksi_cuaca"] = df.apply(lambda r: klasifikasi_cuaca(r["curah_hujan"], r["matahari"]), axis=1)
df["hujan_ekstrem"] = df["curah_hujan"].apply(lambda x: "Ya" if x > extreme_threshold else "Tidak")
df["extreme_flag"] = df["curah_hujan"].apply(lambda x: hujan_ekstrem_flag(x, extreme_threshold))
df["risk_score"] = df.apply(lambda r: risiko_kekeringan_score(r["curah_hujan"], r["matahari"]), axis=1)
df["risk_label"] = df["risk_score"].apply(lambda s: risiko_kekeringan_label(s, (risk_high, risk_med)))
df["weather_index"] = compute_weather_index(df)
df["year"] = df["tanggal"].dt.year
df["month"] = df["tanggal"].dt.month


# ---------- DASHBOARD ----------
st.markdown("---")
st.subheader("Ringkasan Data")
c1, c2, c3 = st.columns(3)
c1.metric("Avg Rainfall", f"{df['curah_hujan'].mean():.2f} mm")
c2.metric("Avg Temperature", f"{df['tavg'].mean():.2f} °C")
c3.metric("Avg Risk Score", f"{df['risk_score'].mean():.2f}")

# ========== 1. CURAH HUJAN ==========
st.header("1. Prediksi Curah Hujan")
fig_rain = px.line(df, x="tanggal", y="curah_hujan")
st.plotly_chart(fig_rain, use_container_width=True)

# ========== 2. TEMPERATURE ==========
st.header("2. Temperatur")
fig_temp = px.line(df, x="tanggal", y=["tn", "tavg", "tx"])
st.plotly_chart(fig_temp, use_container_width=True)

# ========== 3. RISIKO KEKERINGAN ==========
st.header("3. Risiko Kekeringan")
df["risk_ma"] = df["risk_score"].rolling(ma_window).mean()
fig_risk = px.line(df, x="tanggal", y="risk_score")
fig_risk.add_scatter(x=df["tanggal"], y=df["risk_ma"], mode="lines", name="Moving Average")
st.plotly_chart(fig_risk, use_container_width=True)

# ========== 4. HUJAN EKSTREM ==========
st.header("4. Hujan Ekstrem")
fig_extreme = px.scatter(df, x="tanggal", y="curah_hujan", color="hujan_ekstrem")
st.plotly_chart(fig_extreme, use_container_width=True)

# ========== 5. WEATHER INDEX ==========
st.header("5. Weather Index")
fig_index = px.line(df, x="tanggal", y="weather_index")
st.plotly_chart(fig_index, use_container_width=True)

# ========== EXPORT ==========
st.markdown("---")
with st.expander("📁 Unduh Data"):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)

    buffer.seek(0)
    st.download_button(
        "Unduh Excel",
        buffer.getvalue(),
        "hasil_dss_iklim.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.success("Dashboard siap digunakan dengan file data_yapen.xlsx ✔")









