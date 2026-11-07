# 🎓 PrediksiKu - Sistem Prediksi Kelulusan Mahasiswa

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Decision%20Tree-F7931E?logo=scikitlearn&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?logo=sqlite&logoColor=white)

Aplikasi web sederhana untuk memprediksi apakah seorang mahasiswa **lulus tepat waktu** berdasarkan
**IPK** dan **total SKS**. Model Machine Learning dilatih sekali, disimpan ke file `.pkl`, lalu dipakai
oleh antarmuka **Streamlit**. Setiap hasil prediksi disimpan ke database **SQLite** sebagai riwayat.

> Proyek ini dibuat untuk **Praktikum 10 - Rekayasa Perangkat Lunak (SSD1030)**
> Program Studi Sains Data, UIN K.H. Abdurrahman Wahid Pekalongan.
> Topik: MLOps Dasar (Deployment Model & Arsitektur Sistem Berbasis AI).

<!--
Setelah menambahkan screenshot ke folder docs/, hapus tanda komentar di bawah ini.

## 🖼️ Tampilan

![Halaman Prediksi](docs/prediksi.png)
![Halaman Riwayat](docs/riwayat.png)
-->

## ✨ Fitur

- **Prediksi kelulusan** dari input nama, IPK (0.0 - 4.0), dan total SKS (0 - 200).
- **Kartu status** yang berubah hijau (lulus) atau merah (tidak lulus) setelah prediksi,
  lengkap dengan alasan jika ditolak oleh aturan IPK minimal.
- **Aturan IPK minimal**: IPK di bawah 2.00 otomatis dinyatakan tidak lulus, apa pun
  jumlah SKS-nya.
- **Riwayat prediksi** tersimpan permanen di SQLite dan tampil dalam tabel berlabel warna.
- **Navigasi tiga halaman**: Prediksi, Riwayat, dan Tentang.
- **Validasi input**: semua kolom wajib diisi, dan nama dilindungi dari injeksi HTML.
- **Pemisahan UI dan model**: model dimuat dari file `.pkl` dan di-cache dengan `st.cache_resource`.

## 🏗️ Arsitektur

```
Pengguna → UI Streamlit → Logika Prediksi → Model (.pkl) → Database SQLite
```

| Lapisan | Peran | Implementasi |
|---|---|---|
| Presentation Layer | Form input dan tampilan hasil | Streamlit (`app.py`) |
| Business Logic Layer | Validasi input, aturan IPK minimal, dan memanggil model | Fungsi `prediksi_status()` |
| Machine Learning Layer | Melakukan prediksi | `model_kelulusan.pkl` (Decision Tree) |
| Database Layer | Menyimpan riwayat prediksi | SQLite (`riwayat_prediksi.db`) |

## 📁 Struktur Proyek

```
Praktik10_PrediksiKu/
├── 1_latih_model.py        # Tahap 2: melatih dan menyimpan model
├── model_kelulusan.pkl     # Model hasil latih (serialization dengan joblib)
├── app.py                  # Tahap 3: aplikasi Streamlit
├── requirements.txt        # Daftar library
├── riwayat_prediksi.db     # Dibuat otomatis saat aplikasi pertama dijalankan
└── README.md
```

## 🚀 Cara Menjalankan

**Prasyarat:** Python 3.10 atau lebih baru.

1. Clone repositori dan masuk ke foldernya:

   ```bash
   git clone https://github.com/USERNAME/NAMA-REPO.git
   cd NAMA-REPO
   ```

2. (Opsional) Buat virtual environment:

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. Pasang library:

   ```bash
   pip install -r requirements.txt
   ```

4. Latih model dan simpan ke file `.pkl`:

   ```bash
   python 1_latih_model.py
   ```

5. Jalankan aplikasi:

   ```bash
   streamlit run app.py
   ```

   Browser akan terbuka otomatis di `http://localhost:8501`.

## 🧪 Contoh Penggunaan

| IPK | SKS | Hasil |
|---|---|---|
| 3.80 | 150 | ✅ Lulus Tepat Waktu |
| 3.00 | 144 | ✅ Lulus Tepat Waktu |
| 3.00 | 120 | ❌ Tidak Lulus |
| 3.20 | 72 | ❌ Tidak Lulus |
| 1.00 | 144 | ❌ Tidak Lulus (IPK di bawah batas minimal 2.00) |
| 0.00 | 144 | ❌ Tidak Lulus (IPK di bawah batas minimal 2.00) |

## 📚 Konsep MLOps yang Dipelajari

- **Serialization**: menyimpan model terlatih ke file dengan `joblib.dump(model, 'model_kelulusan.pkl')`
  agar tidak perlu dilatih ulang setiap aplikasi dijalankan.
- **Deserialization**: memuat kembali model dengan `joblib.load('model_kelulusan.pkl')`.
- **Deployment**: menyediakan model lewat antarmuka web sehingga pengguna bisa memakainya
  tanpa menjalankan kode Python.
- **Arsitektur sistem AI**: UI, logika bisnis, model, dan database dipisahkan agar mudah
  dikembangkan dan dirawat.

## ⚠️ Catatan

- **Aturan IPK minimal:** model contoh hanya dilatih dari 5 data sehingga hanya belajar
  dari SKS dan mengabaikan IPK. Karena itu aturan IPK minimal 2.00 ditambahkan di lapisan
  logika bisnis, sebelum model dipanggil. Nilainya dapat diubah lewat konstanta
  `IPK_MINIMAL` di `app.py`.
- Model dilatih dari **5 data contoh** saja, sehingga hanya untuk keperluan belajar dan
  bukan untuk keputusan akademik yang sebenarnya.
- `random_state=0` pada `DecisionTreeClassifier` dipakai agar hasil latih selalu sama.
  Tanpa itu, pohon bisa memilih IPK atau SKS secara acak karena keduanya sama-sama
  memisahkan data latih dengan sempurna. Dengan pengaturan ini, model memakai batas
  **SKS 125**.
- Jangan memuat file `.pkl` dari sumber yang tidak dikenal, karena format ini dapat
  menjalankan kode saat dibuka.

## 🛠️ Teknologi

- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)
- [joblib](https://joblib.readthedocs.io/)
- [SQLite](https://www.sqlite.org/)

## 👤 Pembuat

- **Nama:** [Nama Anda]
- **NIM:** [NIM Anda]
- **Mata Kuliah:** Rekayasa Perangkat Lunak (SSD1030)
- **Program Studi:** Sains Data, UIN K.H. Abdurrahman Wahid Pekalongan