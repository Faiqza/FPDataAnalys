import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import os
import glob
import math

# Konfigurasi halaman
st.set_page_config(page_title="Analisis Kualitas Udara dari Stasiun Wanshouxigong oleh maliki_borneo")

# Memuat dataset
# Load folder from drive
folder_data = "Data" 

dfs = []
for name in os.listdir(folder_data):
    file_path = os.path.join(folder_data, name)
    df = pd.read_csv(file_path)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# Judul dashboard
st.title('Dashboard Analisis Kualitas Udara')

# Deskripsi
st.write('Dashboard ini menyediakan cara interaktif untuk menjelajahi data kualitas udara, terutama fokus pada tingkat PM2.5 dan hubungannya dengan berbagai kondisi cuaca.')

# Tentang saya
st.markdown("""
### Tentang Saya
- **Nama**: faiq fahreza
- **Alamat Email**: m004d4ky2473@bangkit.academy
- **ID Dicoding**: Faiq Fahreza M004D4KY2473

### Pertanyaan :
- Bagaimanakah hubungan antara pm2.5 dan pm10?
- Tahun berapakah yang memiliki rerata bulanan pm2.5 paling tinggi?
""")

# Sidebar untuk interaksi pengguna
st.sidebar.header('Fitur Input Pengguna')

# Memilih tahun dan bulan untuk melihat data
selected_year = st.sidebar.selectbox('Pilih Tahun', sorted(data['year'].unique()))
selected_month = st.sidebar.selectbox('Pilih Bulan', sorted(data['month'].unique()))

# Filter data berdasarkan tahun dan bulan yang dipilih
data_filtered = data[(data['year'] == selected_year) & (data['month'] == selected_month)].copy()

# Menampilkan statistik data
st.subheader('Gambaran Data untuk Periode yang Dipilih')
st.write(data_filtered.describe())

# Grafik garis untuk tingkat PM2.5 selama bulan yang dipilih
st.subheader('Tingkat PM2.5 Harian')
fig, ax = plt.subplots()
ax.plot(data_filtered['day'], data_filtered['PM2.5'])
plt.xlabel('Hari dalam Bulan')
plt.ylabel('Konsentrasi PM2.5')
st.pyplot(fig)

# Analisis Trend Musiman
st.subheader('Analisis Trend Musiman')
trend_musiman = data.groupby('month')['PM2.5'].mean()
fig, ax = plt.subplots()
trend_musiman.plot(kind='bar', color='skyblue', ax=ax)
plt.title('Rata-rata Tingkat PM2.5 Bulanan')
plt.xlabel('Bulan')
plt.ylabel('Rata-rata PM2.5')
st.pyplot(fig)

# Tingkat PM2.5 Harian
st.subheader('Tingkat PM2.5 Harian')
fig, ax = plt.subplots()
ax.plot(data_filtered['day'], data_filtered['PM2.5'])
plt.xlabel('Hari dalam Bulan')
plt.ylabel('Konsentrasi PM2.5')
st.pyplot(fig)

# Distribusi Polutan
st.subheader('Distribusi Polutan')
selected_pollutant = st.selectbox('Pilih Polutan', ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO'])
fig, ax = plt.subplots()
sns.boxplot(x='month', y=selected_pollutant, data=data[data['year'] == selected_year], ax=ax)
st.pyplot(fig)

# Dekomposisi Deret Waktu PM2.5
st.subheader('Dekomposisi Deret Waktu PM2.5')
try:
    data_filtered['PM2.5'].ffill(inplace=True)
    decomposed = seasonal_decompose(data_filtered['PM2.5'], model='additive', period=24) # Sesuaikan periode jika perlu
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    decomposed.trend.plot(ax=ax1, title='Tren')
    decomposed.seasonal.plot(ax=ax2, title='Musiman')
    decomposed.resid.plot(ax=ax3, title='Residual')
    plt.tight_layout()
    st.pyplot(fig)
except ValueError as e:
    st.error("Tidak dapat melakukan dekomposisi deret waktu: " + str(e))

# Curah Hujan vs. Kualitas Udara
st.subheader('Curah Hujan vs. Tingkat PM2.5')
fig, ax = plt.subplots()
sns.scatterplot(x='RAIN', y='PM2.5', data=data_filtered, ax=ax)
plt.xlabel('Curah Hujan')
plt.ylabel('Konsentrasi PM2.5')
st.pyplot(fig)

# Histogram for Specific Variables
st.subheader('Histogram for Specific Variables')
num_cols = len(data_filtered.columns[5:-1])
num_rows = math.ceil(num_cols / 4)
plt.figure(figsize=(15, 10))

for i, col in enumerate(data_filtered.columns[5:-1], 1):
    row_num = math.ceil(i / 4)
    col_num = i % 4 if i % 4 != 0 else 4
    plt.subplot(num_rows, 4, i)
    sns.histplot(data_filtered[col], kde=True)
    plt.title(col)

plt.tight_layout()
st.pyplot()

# Scatter Plot PM2.5 vs. PM10
st.write('Pertanyaan 1 : Bagaimanakah hubungan antara pm2.5 dan PM10?')
st.subheader('Scatter Plot PM2.5 vs. PM10')
correlation_matrix = data_filtered[['PM2.5', 'PM10']].corr()
plt.scatter(correlation_matrix['PM2.5'], correlation_matrix['PM10'], s=32, alpha=0.8)
plt.title('Scatter Plot of PM2.5 vs. PM10')
plt.xlabel('PM2.5')
plt.ylabel('PM10')
plt.grid(True)
st.pyplot()

st.write('Pertanyaan 2 : Tahun berapakah yang memiliki rerata bulanan pm2.5 paling tinggi?')
# Plotting Monthly Average Concentrations of NO2
st.subheader('Monthly Average Concentrations of NO2')
data_time_series =
