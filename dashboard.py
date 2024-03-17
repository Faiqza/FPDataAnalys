import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import os
import glob

# Konfigurasi halaman
st.set_page_config(page_title="Analisis Kualitas Udara dari Stasiun Wanshouxigong oleh maliki_borneo")

# Memuat dataset
# Load folder dari drive
folder_data = "C:/FP Data Analys bangkit/Data"  

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

# Heatmap korelasi untuk bulan yang dipilih
st.subheader('Heatmap Korelasi Indikator Kualitas Udara')
corr = data_filtered[['PM2.5', 'NO2', 'SO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP']].corr()
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, ax=ax)
plt.title('Heatmap Korelasi')
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

# Heatmap Rata-rata Harian
st.subheader('Rata-rata Harian PM2.5')
try:
    # Pastikan tipe data yang benar dan tangani nilai yang hilang
    data['hour'] = data['hour'].astype(int)
    data['PM2.5'] = pd.to_numeric(data['PM2.5'], errors='coerce')
    data['PM2.5'].ffill(inplace=True)

    # Hitung rata-rata harian
    rata_harian = data.groupby('hour')['PM2.5'].mean()

    # Plot
    fig, ax = plt.subplots()
    sns.heatmap([rata_harian.values], ax=ax, cmap='coolwarm')
    plt.title('Rata-rata Harian PM2.5')
    st.pyplot(fig)
except Exception as e:
    st.error(f"Error dalam plotting rata-rata harian: {e}")

# Analisis Arah Angin
st.subheader('Analisis Arah Angin')
data_angin = data_filtered.groupby('wd')['PM2.5'].mean()
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, polar=True)
theta = np.linspace(0, 2 * np.pi, len(data_angin))
bars = ax.bar(theta, data_angin.values, align='center', alpha=0.5)
plt.title('Tingkat PM2.5 berdasarkan Arah Angin')
st.pyplot(fig)

# Curah Hujan vs. Kualitas Udara
st.subheader('Curah Hujan vs. Tingkat PM2.5')
fig, ax = plt.subplots()
sns.scatterplot(x='RAIN', y='PM2.5', data=data_filtered, ax=ax)
plt.title('Curah Hujan vs. Tingkat PM2.5')
st.pyplot(fig)

# Heatmap Korelasi - Interaktif
st.subheader('Heatmap Korelasi Interaktif')
selected_columns = st.multiselect('Pilih Kolom untuk Korelasi', data.columns, default=['PM2.5', 'NO2', 'TEMP', 'PRES', 'DEWP'])
corr = data[selected_columns].corr()
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, ax=ax)
st.pyplot(fig)

# Kesimpulan
st.subheader('Kesimpulan')
st.write("""
- Dashboard ini menyediakan analisis data kualitas udara yang mendalam dan interaktif.
- Berbagai visualisasi menawarkan wawasan tentang tingkat PM2.5, distribusinya, dan faktor-faktor yang memengaruhinya.
- Tren musiman dan dampak kondisi cuaca dan polutan yang berbeda terhadap kualitas udara tergambar dengan jelas.
- Pengguna dapat menjelajahi data secara dinamis untuk mendapatkan pemahaman yang lebih dalam tentang tren kualitas udara.
""")
