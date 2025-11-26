import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# =====================================================
# JUDUL APLIKASI
# =====================================================
st.title("🌦️ Prediksi Iklim di Wilayah Kepulauan Yapen dengan Machine Learning")
st.write("Data dimuat otomatis, tanpa perlu upload file.")

# =====================================================
# AUTO LOAD DATA
# =====================================================
DATA_PATH = "KYAPEN.xlsx"   # GANTI JIKA PERLU
SHEET_NAME = "Data Harian - Table"

try:
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    st.success("✅ Data berhasil dimuat otomatis!")
except Exception as e:
    st.error(f"❌ Gagal memuat data: {e}")
    st.stop()

# =====================================================
# 1. CLEANING & PREPARASI DATA
# =====================================================

# Hapus kolom duplikat
df = df.loc[:, ~df.columns.duplicated()]

# Mapping nama kolom
if "kecepatan_angin" in df.columns:
    df = df.rename(columns={"kecepatan_angin": "FF_X"})

# Konversi tanggal
df["Tanggal"] = pd.to_datetime(df["Tanggal"], dayfirst=True)
df["Tahun"] = df["Tanggal"].dt.year
df["Bulan"] = df["Tanggal"].dt.month

# =====================================================
# 2. PILIH VARIABEL
# =====================================================
possible_vars = [
    "Tn", "Tx", "Tavg", "kelembaban",
    "curah_hujan", "matahari",
    "FF_X", "DDD_X"
]

available_vars = [v for v in possible_vars if v in df.columns]

akademis_label = {
    "Tn": "Suhu Minimum (°C)",
    "Tx": "Suhu Maksimum (°C)",
    "Tavg": "Suhu Rata-rata (°C)",
    "kelembaban": "Kelembaban Udara (%)",
    "curah_hujan": "Curah Hujan (mm)",
    "matahari": "Durasi Penyinaran Matahari (jam)",
    "FF_X": "Kecepatan Angin Maksimum (m/s)",
    "DDD_X": "Arah Angin saat Kecepatan Maksimum (°)"
}

st.subheader("🔹 Variabel Cuaca yang Tersedia")
for var in available_vars:
    st.write(f"- {akademis_label[var]}")

# =====================================================
# 3. AGREGASI DATA BULANAN
# =====================================================
agg_dict = {v: 'mean' for v in available_vars}

if "curah_hujan" in available_vars:
    agg_dict["curah_hujan"] = "sum"

cuaca_df = df[['Tahun', 'Bulan'] + available_vars]

monthly_df = cuaca_df.groupby(["Tahun", "Bulan"]).agg(agg_dict).reset_index()

st.subheader("📊 Data Cuaca Bulanan (Hasil Agregasi)")
st.dataframe(monthly_df)

# =====================================================
# 4. TRAIN MODEL ML
# =====================================================
X = monthly_df[["Tahun", "Bulan"]]

models = {}
metrics = {}

st.subheader("🧠 Training Model Machine Learning...")

for var in available_vars:
    y = monthly_df[var]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    models[var] = model
    metrics[var] = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred)
    }

# =====================================================
# 5. EVALUASI MODEL
# =====================================================
st.subheader("📈 Evaluasi Model")

for var, m in metrics.items():
    st.write(
        f"**{akademis_label[var]}** → RMSE: {m['RMSE']:.3f} | R²: {m['R2']:.3f}"
    )

# =====================================================
# 6. PREDIKSI MANUAL
# =====================================================
st.subheader("🔮 Prediksi Manual (Per Bulan)")

tahun_input = st.number_input(
    "Masukkan Tahun",
    min_value=2025,
    max_value=2100,
    value=2030
)

bulan_input = st.selectbox(
    "Pilih Bulan",
    list(range(1, 13))
)

input_data = pd.DataFrame([[tahun_input, bulan_input]], columns=["Tahun", "Bulan"])

st.write("### ✅ Hasil Prediksi:")

for var in available_vars:
    hasil = models[var].predict(input_data)[0]
    st.success(f"{akademis_label[var]}: **{hasil:.2f}**")

# =====================================================
# 7. PREDIKSI OTOMATIS 2025 - 2075
# =====================================================
st.subheader("📆 Prediksi Otomatis Tahun 2025 - 2075")

future_years = range(2025, 2076)
future_months = range(1, 13)

future_data = pd.DataFrame(
    [(y, m) for y in future_years for m in future_months],
    columns=["Tahun", "Bulan"]
)

for var in available_vars:
    future_data[f"Pred_{var}"] = models[var].predict(future_data[["Tahun", "Bulan"]])

st.dataframe(future_data.head(12))

# =====================================================
# 8. VISUALISASI GRAFIK
# =====================================================
st.subheader("📈 Grafik Tren Variabel Cuaca")

monthly_df["Sumber"] = "Data Historis"
future_data["Sumber"] = "Prediksi"

gabungan = []

for var in available_vars:
    hist = monthly_df[["Tahun", "Bulan", var, "Sumber"]].rename(columns={var: "Nilai"})
    hist["Variabel"] = akademis_label[var]

    fut = future_data[["Tahun", "Bulan", f"Pred_{var}", "Sumber"]].rename(
        columns={f"Pred_{var}": "Nilai"}
    )
    fut["Variabel"] = akademis_label[var]

    gabungan.append(pd.concat([hist, fut]))

final_df = pd.concat(gabungan)

final_df["Tanggal"] = pd.to_datetime(
    final_df["Tahun"].astype(str) + "-" +
    final_df["Bulan"].astype(str) + "-01"
)

selected_var = st.selectbox(
    "Pilih Variabel",
    final_df["Variabel"].unique()
)

fig = px.line(
    final_df[final_df["Variabel"] == selected_var],
    x="Tanggal",
    y="Nilai",
    color="Sumber",
    title=f"Tren {selected_var} (Historis vs Prediksi)"
)

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 9. DOWNLOAD HASIL PREDIKSI
# =====================================================
st.subheader("💾 Download Hasil Prediksi")

csv = future_data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download CSV Prediksi 2025–2075",
    data=csv,
    file_name="prediksi_iklim_2025_2075.csv",
    mime="text/csv"
)
