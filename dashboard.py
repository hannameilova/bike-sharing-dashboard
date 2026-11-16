import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------------
# Konfigurasi halaman
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Bike Sharing Dashboard",
    page_icon="🚲",
    layout="wide",
)

# ---------------------------------------------------------------------
# Konstanta dan label
# ---------------------------------------------------------------------
WEATHER_LABELS = {
    1: "Cerah",
    2: "Berawan / Kabut",
    3: "Hujan Ringan",
    4: "Hujan Lebat",
}

MONTH_LABELS = {
    1: "Januari", 2: "Februari", 3: "Maret", 4: "April",
    5: "Mei", 6: "Juni", 7: "Juli", 8: "Agustus",
    9: "September", 10: "Oktober", 11: "November", 12: "Desember",
}

REQUIRED_COLUMNS = {
    "dteday", "hr", "mnth", "workingday", "weathersit",
    "temp", "hum", "windspeed", "casual", "registered", "cnt",
}


# ---------------------------------------------------------------------
# Fungsi pemuatan dan persiapan data
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(file_path):
    """Membaca dataset Bike Sharing dari file hour.csv di folder project."""
    return pd.read_csv(file_path)


def prepare_data(raw_df):
    """Membersihkan dan menambahkan kolom turunan sesuai notebook analisis."""
    missing_columns = REQUIRED_COLUMNS.difference(raw_df.columns)
    if missing_columns:
        raise ValueError(
            "Kolom berikut tidak ditemukan pada CSV: "
            + ", ".join(sorted(missing_columns))
        )

    data = raw_df.copy()

    # Pastikan kolom-kolom numerik terbaca sebagai angka.
    numeric_columns = [
        "hr", "mnth", "workingday", "weathersit", "temp", "hum",
        "windspeed", "casual", "registered", "cnt",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # Mengubah tanggal dan menyusun urutan waktu.
    data["dteday"] = pd.to_datetime(data["dteday"], errors="coerce")
    data = data.dropna(subset=["dteday", "hr", "cnt"])
    data["datetime"] = (
        data["dteday"] + pd.to_timedelta(data["hr"], unit="h")
    )
    data = data.sort_values("datetime").reset_index(drop=True)

    # Mengikuti proses cleaning pada notebook:
    # nilai hum = 0 dianggap tidak valid, lalu diimputasi dengan median
    # berdasarkan kombinasi bulan (mnth) dan jam (hr).
    data["hum"] = pd.to_numeric(data["hum"], errors="coerce")
    data.loc[data["hum"] == 0, "hum"] = np.nan

    median_hum = data.groupby(["mnth", "hr"])["hum"].transform("median")
    data["hum"] = data["hum"].fillna(median_hum)
    data["hum"] = data["hum"].fillna(data["hum"].median())

    # Konversi suhu aktual dan pengelompokan suhu per interval 5°C.
    data["temp"] = pd.to_numeric(data["temp"], errors="coerce")
    data["temp_c"] = data["temp"] * 41

    bins = np.arange(0, 46, 5)
    labels = [f"{bins[i]}–{bins[i + 1]}°C" for i in range(len(bins) - 1)]
    data["kelompok_temp"] = pd.cut(
        data["temp_c"],
        bins=bins,
        labels=labels,
        right=False,
        ordered=True,
    )

    # Label untuk kebutuhan filter dan visualisasi.
    data["tahun"] = data["dteday"].dt.year
    data["nama_bulan"] = data["mnth"].map(MONTH_LABELS)
    data["kondisi_cuaca"] = data["weathersit"].map(WEATHER_LABELS).fillna(
        data["weathersit"].astype(str)
    )
    data["jenis_hari"] = data["workingday"].map({
        0: "Hari Non-Kerja",
        1: "Hari Kerja",
    }).fillna("Tidak diketahui")

    return data


def format_number(value):
    """Format angka dengan pemisah ribuan Indonesia."""
    return f"{value:,.0f}".replace(",", ".")


# ---------------------------------------------------------------------
# Memuat data dan menyiapkan sidebar filter
# ---------------------------------------------------------------------
st.title("🚲 Bike Sharing Dashboard")
st.caption(
    "Dashboard analisis pola penyewaan sepeda berdasarkan waktu, "
    "jenis hari, kondisi cuaca, dan suhu."
)

# Dataset hour.csv dibaca langsung dari folder yang sama dengan dashboard.py.
# Dengan cara ini, aplikasi tidak memerlukan unggahan file melalui sidebar.
DATA_FILE = Path(__file__).resolve().parent / "hour.csv"

if not DATA_FILE.is_file():
    st.error(
        "File **hour.csv** tidak ditemukan. Pastikan file tersebut berada "
        "di folder yang sama dengan `dashboard.py`. "
        f"Lokasi yang diperiksa: `{DATA_FILE}`"
    )
    st.stop()

try:
    raw_df = load_csv(str(DATA_FILE))
    df = prepare_data(raw_df)
except Exception as error:
    st.error(f"Dataset tidak dapat diproses: {error}")
    st.stop()

with st.sidebar:
    st.header("Pengaturan Dashboard")
    st.subheader("Filter Data")

if df.empty:
    st.warning("Dataset tidak memiliki baris data yang dapat dianalisis.")
    st.stop()

min_date = df["dteday"].min().date()
max_date = df["dteday"].max().date()

with st.sidebar:
    selected_dates = st.date_input(
        "Rentang tanggal",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(selected_dates, (tuple, list)):
        if len(selected_dates) == 2:
            start_date, end_date = selected_dates
        elif len(selected_dates) == 1:
            start_date = end_date = selected_dates[0]
        else:
            start_date, end_date = min_date, max_date
    else:
        start_date = end_date = selected_dates

    available_years = sorted(df["tahun"].dropna().unique().tolist())
    year_options = ["Semua"] + [int(year) for year in available_years]
    selected_year = st.selectbox("Tahun", year_options, index=0)

    month_options = list(range(1, 13))
    selected_months = st.multiselect(
        "Bulan",
        options=month_options,
        default=month_options,
        format_func=lambda month: MONTH_LABELS[month],
    )

    working_options = ["Semua", "Hari Kerja", "Hari Non-Kerja"]
    selected_workingday = st.selectbox("Jenis hari", working_options)

    weather_options = [
        label for code, label in WEATHER_LABELS.items()
        if code in df["weathersit"].dropna().unique()
    ]
    selected_weather = st.multiselect(
        "Kondisi cuaca",
        options=weather_options,
        default=weather_options,
    )

# ---------------------------------------------------------------------
# Terapkan filter
# ---------------------------------------------------------------------
filtered_df = df[
    df["dteday"].dt.date.between(start_date, end_date)
].copy()

if selected_year != "Semua":
    filtered_df = filtered_df[filtered_df["tahun"] == selected_year]

if selected_months:
    filtered_df = filtered_df[filtered_df["mnth"].isin(selected_months)]
else:
    filtered_df = filtered_df.iloc[0:0]

if selected_workingday != "Semua":
    filtered_df = filtered_df[
        filtered_df["jenis_hari"] == selected_workingday
    ]

if selected_weather:
    filtered_df = filtered_df[
        filtered_df["kondisi_cuaca"].isin(selected_weather)
    ]
else:
    filtered_df = filtered_df.iloc[0:0]

if filtered_df.empty:
    st.warning(
        "Tidak ada data untuk kombinasi filter yang dipilih. "
        "Silakan ubah rentang tanggal atau pilihan filter."
    )
    st.stop()

# ---------------------------------------------------------------------
# Ringkasan metrik
# ---------------------------------------------------------------------
st.subheader("Ringkasan Penyewaan")

total_rentals = int(filtered_df["cnt"].sum())
average_rentals = filtered_df["cnt"].mean()
casual_total = int(filtered_df["casual"].sum())
registered_total = int(filtered_df["registered"].sum())

metric_cols = st.columns(4)
metric_cols[0].metric("Total Penyewaan", format_number(total_rentals))
metric_cols[1].metric("Rata-rata per Jam", f"{average_rentals:,.2f}")
metric_cols[2].metric("Pengguna Kasual", format_number(casual_total))
metric_cols[3].metric("Pengguna Terdaftar", format_number(registered_total))

date_text = (
    f"{filtered_df['dteday'].min():%d %B %Y} – "
    f"{filtered_df['dteday'].max():%d %B %Y}"
)
st.caption(
    f"Menampilkan {format_number(len(filtered_df))} catatan per jam "
    f"untuk periode {date_text}."
)

st.divider()

# ---------------------------------------------------------------------
# Visualisasi 1: rata-rata penyewaan berdasarkan jam
# ---------------------------------------------------------------------
st.subheader("1. Rata-rata Penyewaan Sepeda Berdasarkan Jam")

hourly_df = (
    filtered_df.groupby("hr", as_index=False)["cnt"]
    .mean()
    .rename(columns={"cnt": "rata_rata_penyewaan"})
    .sort_values("hr")
)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(
    hourly_df["hr"],
    hourly_df["rata_rata_penyewaan"],
    marker="o",
    markersize=6,
    linewidth=2,
)
ax.set_title("Rata-rata Jumlah Penyewaan Sepeda Berdasarkan Jam")
ax.set_xlabel("Jam")
ax.set_ylabel("Rata-rata Penyewaan")
ax.set_xticks(range(24))
ax.grid(axis="both", linestyle="--", alpha=0.4)

if not hourly_df.empty:
    max_row = hourly_df.loc[hourly_df["rata_rata_penyewaan"].idxmax()]
    min_row = hourly_df.loc[hourly_df["rata_rata_penyewaan"].idxmin()]
    ax.annotate(
        f"Puncak: {max_row['rata_rata_penyewaan']:.2f}",
        (max_row["hr"], max_row["rata_rata_penyewaan"]),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        fontsize=10,
        fontweight="bold",
    )
    ax.annotate(
        f"Terendah: {min_row['rata_rata_penyewaan']:.2f}",
        (min_row["hr"], min_row["rata_rata_penyewaan"]),
        xytext=(8, 10),
        textcoords="offset points",
        ha="left",
        fontsize=9,
    )

fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

if not hourly_df.empty:
    max_hour = int(max_row["hr"])
    min_hour = int(min_row["hr"])
    st.info(
        f"Pada data yang sedang ditampilkan, rata-rata penyewaan tertinggi "
        f"terjadi pukul **{max_hour:02d}.00** "
        f"({max_row['rata_rata_penyewaan']:.2f} sepeda per jam), sedangkan "
        f"terendah terjadi pukul **{min_hour:02d}.00** "
        f"({min_row['rata_rata_penyewaan']:.2f} sepeda per jam)."
    )

st.divider()

# ---------------------------------------------------------------------
# Visualisasi 2: hari kerja vs hari non-kerja
# ---------------------------------------------------------------------
st.subheader("2. Distribusi Total Penyewaan Berdasarkan Jenis Hari")

workingday_df = (
    filtered_df.groupby("jenis_hari", as_index=False)["cnt"]
    .sum()
)
working_order = ["Hari Kerja", "Hari Non-Kerja"]
workingday_df["jenis_hari"] = pd.Categorical(
    workingday_df["jenis_hari"],
    categories=working_order,
    ordered=True,
)
workingday_df = workingday_df.sort_values("jenis_hari")

col_chart, col_info = st.columns([1.35, 1])

with col_chart:
    fig, ax = plt.subplots(figsize=(7, 5))
    if workingday_df["cnt"].sum() > 0:
        ax.pie(
            workingday_df["cnt"],
            labels=workingday_df["jenis_hari"].astype(str),
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"width": 0.42},
            pctdistance=0.78,
        )
        ax.set_title("Proporsi Total Penyewaan")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with col_info:
    st.write("**Total penyewaan per jenis hari**")
    st.dataframe(
        workingday_df.assign(
            total_penyewaan=workingday_df["cnt"].map(format_number)
        )[["jenis_hari", "total_penyewaan"]].rename(
            columns={"jenis_hari": "Jenis Hari"}
        ),
        hide_index=True,
        use_container_width=True,
    )

    if len(workingday_df) > 0 and workingday_df["cnt"].sum() > 0:
        top_day = workingday_df.loc[workingday_df["cnt"].idxmax()]
        share = top_day["cnt"] / workingday_df["cnt"].sum() * 100
        st.info(
            f"Kontribusi terbesar berasal dari **{top_day['jenis_hari']}**, "
            f"yaitu **{share:.1f}%** dari total penyewaan pada filter ini."
        )

st.divider()

# ---------------------------------------------------------------------
# Visualisasi 3: kondisi cuaca
# ---------------------------------------------------------------------
st.subheader("3. Rata-rata Penyewaan Berdasarkan Kondisi Cuaca")

weather_df = (
    filtered_df.groupby("kondisi_cuaca", as_index=False)["cnt"]
    .mean()
    .sort_values("cnt", ascending=False)
)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(weather_df["kondisi_cuaca"], weather_df["cnt"])
ax.set_title("Rata-rata Penyewaan Sepeda Berdasarkan Kondisi Cuaca")
ax.set_xlabel("Kondisi Cuaca")
ax.set_ylabel("Rata-rata Penyewaan per Jam")
ax.tick_params(axis="x", rotation=12)

for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

if not weather_df.empty:
    ax.set_ylim(0, weather_df["cnt"].max() * 1.16)

fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

if not weather_df.empty:
    lowest_weather = weather_df.loc[weather_df["cnt"].idxmin()]
    highest_weather = weather_df.loc[weather_df["cnt"].idxmax()]
    st.info(
        f"Rata-rata penyewaan tertinggi terdapat pada kondisi "
        f"**{highest_weather['kondisi_cuaca']}** "
        f"({highest_weather['cnt']:.2f}), sedangkan terendah pada kondisi "
        f"**{lowest_weather['kondisi_cuaca']}** "
        f"({lowest_weather['cnt']:.2f}) sepeda per jam."
    )

st.divider()

# ---------------------------------------------------------------------
# Visualisasi 4: kelompok suhu
# ---------------------------------------------------------------------
st.subheader("4. Rata-rata Penyewaan Berdasarkan Kelompok Suhu")

temp_df = (
    filtered_df.dropna(subset=["kelompok_temp"])
    .groupby("kelompok_temp", observed=False)["cnt"]
    .mean()
    .reset_index()
    .dropna(subset=["cnt"])
)
temp_df["kelompok_temp"] = temp_df["kelompok_temp"].astype(str)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(
    temp_df["kelompok_temp"],
    temp_df["cnt"],
    marker="o",
    linewidth=2,
)
for index, row in enumerate(temp_df.itertuples(index=False)):
    ax.annotate(
        f"{row.cnt:.2f}",
        (index, row.cnt),
        xytext=(0, 9),
        textcoords="offset points",
        ha="center",
        fontsize=8,
    )
ax.set_title("Rata-rata Penyewaan Sepeda Berdasarkan Kelompok Suhu")
ax.set_xlabel("Kelompok Suhu")
ax.set_ylabel("Rata-rata Penyewaan per Jam")
ax.tick_params(axis="x", rotation=15)
ax.grid(axis="both", linestyle="--", alpha=0.4)
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

if not temp_df.empty:
    warmest_group = temp_df.loc[temp_df["cnt"].idxmax()]
    coolest_group = temp_df.loc[temp_df["cnt"].idxmin()]
    st.caption(
        f"Kelompok dengan rata-rata penyewaan tertinggi adalah "
        f"**{warmest_group['kelompok_temp']}** "
        f"({warmest_group['cnt']:.2f}), sedangkan terendah adalah "
        f"**{coolest_group['kelompok_temp']}** "
        f"({coolest_group['cnt']:.2f})."
    )

st.divider()

# ---------------------------------------------------------------------
# Analisis lanjutan: histogram dan correlation matrix
# ---------------------------------------------------------------------
st.header("Analisis Lanjutan")
st.caption(
    "Distribusi variabel numerik dan hubungan linear antarvariabel "
    "pada data yang telah difilter."
)

hist_columns = [
    column for column in
    ["temp", "atemp", "hum", "windspeed", "casual", "registered", "cnt"]
    if column in filtered_df.columns
]

with st.expander("Lihat Histogram Distribusi Variabel", expanded=True):
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()

    for index, column in enumerate(hist_columns):
        sns.histplot(
            data=filtered_df,
            x=column,
            bins=30,
            kde=True,
            ax=axes[index],
        )
        axes[index].set_title(f"Distribusi {column}")
        axes[index].set_xlabel(column)
        axes[index].set_ylabel("Frekuensi")

    for index in range(len(hist_columns), len(axes)):
        fig.delaxes(axes[index])

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with st.expander("Lihat Correlation Matrix", expanded=True):
    numeric_df = filtered_df.select_dtypes(include=["number"]).copy()
    numeric_df = numeric_df.drop(
        columns=["instant"], errors="ignore"
    )

    if numeric_df.shape[1] >= 2:
        correlation_matrix = numeric_df.corr()

        fig, ax = plt.subplots(
            figsize=(max(10, len(correlation_matrix.columns) * 0.8),
                     max(8, len(correlation_matrix.columns) * 0.7))
        )
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.6,
            center=0,
            square=True,
            vmin=-1,
            vmax=1,
            ax=ax,
            annot_kws={"size": 9},
        )
        ax.set_title("Correlation Matrix Dataset Penyewaan Sepeda")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        corr_with_cnt = (
            correlation_matrix["cnt"]
            .drop(labels=["cnt"], errors="ignore")
            .sort_values(key=lambda values: values.abs(), ascending=False)
        )
        if not corr_with_cnt.empty:
            strongest = corr_with_cnt.index[0]
            st.caption(
                f"Korelasi dengan `cnt` yang paling kuat pada data terfilter "
                f"adalah **{strongest}** dengan nilai "
                f"**{corr_with_cnt.iloc[0]:.2f}**. "
                "Korelasi menunjukkan hubungan, bukan sebab-akibat."
            )
    else:
        st.warning("Belum cukup kolom numerik untuk menghitung korelasi.")

# ---------------------------------------------------------------------
# Data tabel dan catatan
# ---------------------------------------------------------------------
with st.expander("Lihat Data yang Digunakan"):
    display_columns = [
        column for column in [
            "dteday", "hr", "workingday", "kondisi_cuaca",
            "temp_c", "hum", "casual", "registered", "cnt"
        ]
        if column in filtered_df.columns
    ]
    st.dataframe(
        filtered_df[display_columns].rename(columns={
            "dteday": "Tanggal",
            "hr": "Jam",
            "workingday": "Hari Kerja (0/1)",
            "kondisi_cuaca": "Kondisi Cuaca",
            "temp_c": "Suhu (°C)",
            "hum": "Kelembapan",
            "casual": "Pengguna Kasual",
            "registered": "Pengguna Terdaftar",
            "cnt": "Total Penyewaan",
        }),
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "Catatan: Dashboard ini menggunakan data per jam. "
    "Filter dan seluruh visualisasi akan diperbarui mengikuti pilihan pada sidebar."
)
