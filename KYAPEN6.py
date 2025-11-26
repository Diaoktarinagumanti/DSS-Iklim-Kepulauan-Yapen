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
st.markdown(
    "Dashboard prediksi & analisis iklim. Data otomatis dimuat dari `data_yapen.csv` "
    "jika tersedia; jika tidak aplikasi membuat data contoh."
)

LOCAL_CSV_PATH = "data_yapen.csv"

# ---------- DSS helper functions ----------
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
    r = df['curah_hujan'].fillna(0).astype(float).values
    r_norm = (r - r.min()) / (r.max() - r.min() + eps)

    t = df['Tavg'].fillna(0).astype(float).values
    comfy_low, comfy_high = 24, 28
    t_dist = np.maximum(0, np.maximum(comfy_low - t, t - comfy_high))
    t_norm = (t_dist - t_dist.min()) / (t_dist.max() - t_dist.min() + eps)

    h = df['kelembaban'].fillna(0).astype(float).values
    hum_dist = np.maximum(0, np.maximum(40 - h, h - 70))
    h_norm = (hum_dist - hum_dist.min()) / (hum_dist.max() - hum_dist.min() + eps)

    w = df['kecepatan_angin'].fillna(0).astype(float).values
    w_norm = (w - w.min()) / (w.max() - w.min() + eps)

    composite = 0.35 * r_norm + 0.25 * t_norm + 0.2 * h_norm + 0.2 * w_norm
    return np.clip(composite, 0, 1)

# ---------- Data loading ----------
@st.cache_data(show_spinner=False)
def load_data(local_path=LOCAL_CSV_PATH):
    try:
        df = pd.read_csv(local_path, parse_dates=['Tanggal'])
        st.sidebar.success(f"Loaded local CSV: {local_path}")
    except Exception:
        st.sidebar.info("Local CSV tidak ditemukan — membuat data contoh 2 tahun.")
        end = pd.Timestamp.today()
        start = end - pd.Timedelta(days=730)
        rng = pd.date_range(start=start, end=end, freq='D')
        np.random.seed(42)
        df = pd.DataFrame({
            'Tanggal': rng,
            'curah_hujan': np.random.gamma(1.5, 8, len(rng)).round(1),
            'Tn': np.random.normal(22, 2, len(rng)).round(1),
            'Tx': np.random.normal(31, 2.5, len(rng)).round(1),
            'Tavg': np.random.normal(26.5, 1.8, len(rng)).round(1),
            'kelembaban': np.random.randint(50, 95, len(rng)),
            'matahari': np.clip(np.random.normal(5, 2, len(rng)), 0, 12).round(1),
            'kecepatan_angin': np.random.uniform(0, 20, len(rng)).round(1),
        })

    df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')

    for col in ['curah_hujan', 'Tn', 'Tx', 'Tavg', 'kelembaban', 'matahari', 'kecepatan_angin']:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df.sort_values('Tanggal').reset_index(drop=True)

# Load
data = load_data()

# Sidebar controls
st.sidebar.header("⚙️ Pengaturan")
extreme_threshold = st.sidebar.number_input("Ambang Hujan Ekstrem (mm/hari)", value=50, min_value=1)
risk_high = st.sidebar.slider("Ambang Risiko Tinggi", 0.0, 1.0, 0.6, 0.01)
risk_med = st.sidebar.slider("Ambang Risiko Sedang", 0.0, 1.0, 0.3, 0.01)
ma_window = st.sidebar.slider("Moving average window (hari)", 1, 60, 7)

# Filters
st.sidebar.header("📅 Filter Data")
min_date = data['Tanggal'].min().date()
max_date = data['Tanggal'].max().date()

date_range = st.sidebar.date_input("Rentang tanggal", value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date)

start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
mask = (data['Tanggal'] >= start_date) & (data['Tanggal'] <= end_date)
df = data.loc[mask].copy()

if df.empty:
    st.warning("Tidak ada data pada rentang tanggal — menampilkan seluruh dataset.")
    df = data.copy()

# Derived fields
df['Prediksi Cuaca'] = df.apply(lambda r: klasifikasi_cuaca(r['curah_hujan'], r['matahari']), axis=1)
df['Hujan Ekstrem'] = df['curah_hujan'].apply(lambda x: "Ya" if x > extreme_threshold else "Tidak")
df['extreme_flag'] = df['curah_hujan'].apply(lambda x: hujan_ekstrem_flag(x, extreme_threshold))
df['RiskScore'] = df.apply(lambda r: risiko_kekeringan_score(r['curah_hujan'], r['matahari']), axis=1)
df['RiskLabel'] = df['RiskScore'].apply(lambda s: risiko_kekeringan_label(s, thresholds=(risk_high, risk_med)))
df['WeatherIndex'] = compute_weather_index(df)
df['Year'] = df['Tanggal'].dt.year
df['Month'] = df['Tanggal'].dt.month

# ---------------- Summary Metrics ----------------
st.markdown("---")
st.subheader("Ringkasan Cepat")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Periode", f"{df['Tanggal'].min().date()} — {df['Tanggal'].max().date()}")
c2.metric("Avg Rain (mm)", f"{df['curah_hujan'].mean():.2f}")
c3.metric("Avg Temp (°C)", f"{df['Tavg'].mean():.2f}")
c4.metric("Avg RiskScore", f"{df['RiskScore'].mean():.2f}")

# ---------------- 1. Rainfall Forecast ----------------
st.markdown("---")
st.header("1. Prediksi Curah Hujan")
r1, r2 = st.columns([2, 1])

with r1:
    st.plotly_chart(px.line(df, x='Tanggal', y='curah_hujan', title="Rainfall (Line Chart)"), use_container_width=True)
    st.plotly_chart(px.area(df, x='Tanggal', y='curah_hujan', title="Rainfall (Area Chart)"), use_container_width=True)

with r2:
    st.write("**Kegunaan:** Identifikasi musim hujan & potensi banjir.")
    monthly_sum = df.set_index('Tanggal').resample('M')['curah_hujan'].sum().reset_index()
    st.plotly_chart(px.bar(monthly_sum, x='Tanggal', y='curah_hujan', title="Monthly Rainfall Sum"), use_container_width=True)

# ---------------- 2. Temperature Forecast ----------------
st.markdown("---")
st.header("2. Prediksi Temperatur")
t1, t2 = st.columns([2, 1])

with t1:
    st.plotly_chart(px.line(df, x='Tanggal', y=['Tn', 'Tavg', 'Tx'], title="Temperature Trends"), use_container_width=True)

    heat_df = df.copy()
    heat_df['Day'] = heat_df['Tanggal'].dt.day
    heat_df['MonthName'] = heat_df['Tanggal'].dt.strftime('%b')
    pivot = heat_df.pivot_table(index='MonthName', columns='Day', values='Tavg', aggfunc='mean')

    if pivot.size > 0:
        st.plotly_chart(px.imshow(pivot, title="Temperature Heatmap"), use_container_width=True)

with t2:
    st.write("**Kegunaan:** Deteksi gelombang panas & variasi musiman.")

# ---------------- 3. Risiko Kekeringan ----------------
st.markdown("---")
st.header("3. Prediksi Risiko Kekeringan")
d1, d2 = st.columns([2, 1])

with d1:
    latest_score = df['RiskScore'].mean()
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest_score,
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "blue"},
            'steps': [
                {'range': [0, risk_med], 'color': "lightgreen"},
                {'range': [risk_med, risk_high], 'color': "yellow"},
                {'range': [risk_high, 1], 'color': "red"},
            ]
        },
        title={'text': "Average Drought Risk"}
    ))
    st.plotly_chart(gauge, use_container_width=True)

    df['Risk_MA'] = df['RiskScore'].rolling(ma_window).mean()
    fig_risk_line = px.line(df, x='Tanggal', y='RiskScore', title="Risk Score Over Time")
    fig_risk_line.add_scatter(x=df['Tanggal'], y=df['Risk_MA'], mode='lines', name=f'MA({ma_window})')
    st.plotly_chart(fig_risk_line, use_container_width=True)

with d2:
    st.write("**Kegunaan:** Pemantauan risiko jangka panjang.")

# ---------------- 4. Hujan Ekstrem ----------------
st.markdown("---")
st.header("4. Prediksi Hujan Ekstrem")
e1, e2 = st.columns([2, 1])

with e1:
    freq = df[df['Hujan Ekstrem'] == 'Ya'].groupby(df['Tanggal'].dt.to_period('M')).size().reset_index(name='count')

    if not freq.empty:
        freq['Tanggal'] = freq['Tanggal'].dt.to_timestamp()
        st.plotly_chart(px.bar(freq, x='Tanggal', y='count', title="Frekuensi Hujan Ekstrem per Bulan"),
                        use_container_width=True)

    st.plotly_chart(px.scatter(df, x='Tanggal', y='curah_hujan', color='Hujan Ekstrem',
                               title="Curah Hujan vs Waktu"), use_container_width=True)

    df['extreme_prob_30d'] = df['extreme_flag'].rolling(30, min_periods=1).mean()
    st.plotly_chart(px.line(df, x='Tanggal', y='extreme_prob_30d',
                            title="Probabilitas Hujan Ekstrem 30 Hari"),
                    use_container_width=True)

with e2:
    st.write("**Kegunaan:** Peringatan dini potensi banjir.")

# ---------------- 5. Weather Index ----------------
st.markdown("---")
st.header("5. Indeks Cuaca Gabungan")
w1, w2 = st.columns([2, 1])

with w1:
    comp = pd.DataFrame({
        'rain': (df['curah_hujan'] - df['curah_hujan'].min()) /
                (df['curah_hujan'].max() - df['curah_hujan'].min() + 1e-6),
        'temp_stress': np.maximum(0, np.maximum(24 - df['Tavg'], df['Tavg'] - 28)),
        'hum_stress': np.maximum(0, np.maximum(40 - df['kelembaban'], df['kelembaban'] - 70)),
        'wind': (df['kecepatan_angin'] - df['kecepatan_angin'].min()) /
                (df['kecepatan_angin'].max() - df['kecepatan_angin'].min() + 1e-6),
    })

    for c in ['temp_stress', 'hum_stress']:
        comp[c] = (comp[c] - comp[c].min()) / (comp[c].max() - comp[c].min() + 1e-6)

    avg_comp = comp.mean()
    cats = ['Rain', 'TempStress', 'HumStress', 'Wind']
    vals = avg_comp.values.tolist()
    vals += vals[:1]

    st.plotly_chart(
        go.Figure(
            data=go.Scatterpolar(r=vals, theta=cats + [cats[0]], fill='toself')
        ).update_layout(polar=dict(radialaxis=dict(range=[0, 1])), showlegend=False,
                        title="Weather Index Components"),
        use_container_width=True,
    )

    st.plotly_chart(px.line(df, x='Tanggal', y='WeatherIndex',
                            title="Composite Weather Index Over Time"),
                    use_container_width=True)

with w2:
    st.write("**Kegunaan:** Ringkasan kondisi iklim keseluruhan.")

# ---------------- 6. Tren Bulanan/Tahunan ----------------
st.markdown("---")
st.header("6. Tren Bulanan & Tahunan")
m1, m2 = st.columns(2)

with m1:
    years = sorted(df['Year'].unique())
    fig_multi = go.Figure()
    for y in years:
        tmp = df[df['Year'] == y]
        monthly = tmp.groupby(tmp['Tanggal'].dt.month)['curah_hujan'].mean().reset_index(name='curah_hujan')
        fig_multi.add_trace(go.Scatter(x=monthly['Tanggal'], y=monthly['curah_hujan'],
                                       mode='lines+markers', name=str(y)))
    st.plotly_chart(fig_multi.update_layout(title="Rainfall by Year (Monthly Avg)"), use_container_width=True)

with m2:
    df['Rain_MA'] = df['curah_hujan'].rolling(ma_window).mean()
    st.plotly_chart(px.line(df, x='Tanggal', y=['curah_hujan', 'Rain_MA'],
                            title=f"Moving Average Rainfall (window={ma_window})"),
                    use_container_width=True)

# ---------------- 7. Anomali Iklim ----------------
st.markdown("---")
st.header("7. Prediksi Anomali Iklim")
a1, a2 = st.columns([2, 1])

with a1:
    baseline_temp = data.groupby(data['Tanggal'].dt.month)['Tavg'].mean()
    df['baseline_Tavg'] = df['Tanggal'].dt.month.map(baseline_temp)
    df['Tavg_anom'] = df['Tavg'] - df['baseline_Tavg']

    anom_pivot = df.pivot_table(index=df['Tanggal'].dt.year,
                                columns=df['Tanggal'].dt.month,
                                values='Tavg_anom', aggfunc='mean')

    if anom_pivot.size > 0:
        st.plotly_chart(px.imshow(anom_pivot, title="Temperature Anomaly Heatmap"),
                        use_container_width=True)

    fig_anom_line = go.Figure()
    fig_anom_line.add_trace(go.Scatter(x=df['Tanggal'], y=df['Tavg'], name='Tavg'))
    fig_anom_line.add_trace(go.Scatter(x=df['Tanggal'], y=df['baseline_Tavg'], name='Baseline'))
    st.plotly_chart(fig_anom_line.update_layout(title="Temperature vs Baseline"),
                    use_container_width=True)

with a2:
    st.write("**Kegunaan:** Deteksi pergeseran musiman.")

# ---------------- Data Viewer & Download ----------------
st.markdown("---")
with st.expander("📁 Lihat dan Unduh Data Lengkap"):
    st.dataframe(df)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Hasil_DSS', index=False)

    buffer.seek(0)

    st.download_button(
        "Unduh Excel Hasil Analisis",
        data=buffer.getvalue(),
        file_name="hasil_dss_iklim.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.caption(
    "Catatan: Pastikan CSV memiliki kolom minimal: Tanggal, curah_hujan, "
    "Tn, Tx, Tavg, kelembaban, matahari, kecepatan_angin. "
    "Nama file: data_yapen.csv"
)


