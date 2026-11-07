"""PrediksiKu - Sistem Prediksi Kelulusan Mahasiswa.

Praktikum 10 RPL: MLOps Dasar (Streamlit UI).
Alur: UI (Streamlit) -> logika prediksi -> model .pkl -> database SQLite.
"""
import html
import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import streamlit as st

# ----------------------------------------------------------------------------
# Konfigurasi
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model_kelulusan.pkl"
DB_PATH = BASE_DIR / "riwayat_prediksi.db"

STATUS_LULUS = "Lulus Tepat Waktu"
STATUS_TIDAK = "Tidak Lulus"
IPK_MINIMAL = 2.0  # syarat minimal IPK kelulusan (ubah sesuai aturan kampus)
MENU = ["Prediksi", "Riwayat", "Tentang"]

st.set_page_config(page_title="PrediksiKu", page_icon="🎓", layout="wide")

# ----------------------------------------------------------------------------
# Tampilan (CSS)
# ----------------------------------------------------------------------------
CSS = """
<style>
.stApp {
    background:
        radial-gradient(900px 500px at 0% 0%, #cfe3ff 0%, transparent 60%),
        radial-gradient(800px 500px at 55% 0%, #ece7ff 0%, transparent 55%),
        radial-gradient(900px 600px at 100% 100%, #cfe3ff 0%, transparent 60%),
        #f6f9ff;
}
.block-container { max-width: 1100px; padding-top: 3.2rem; padding-bottom: 3rem; }

/* Logo */
.brand { display:flex; align-items:center; gap:.7rem; padding-top:.35rem; }
.brand-dot { width:26px; height:26px; border-radius:50%;
    background: linear-gradient(135deg,#2563eb,#60a5fa); }
.brand-name { font-weight:800; color:#10244f; font-size:1.02rem; }

/* Header Streamlit transparan (tombol Deploy tetap tampil) */
[data-testid="stHeader"] { background: transparent; }

/* Navigasi pill (st.radio) */
[data-testid="stElementContainer"]:has([data-testid="stRadio"]) {
    width:100% !important; display:flex; justify-content:flex-end;
}
[data-testid="stRadio"] { width:auto !important; }
[data-testid="stRadioGroup"] {
    display:inline-flex !important; flex-direction:row; gap:2px; background:#fff;
    border:1px solid #dfe8f8; border-radius:999px; padding:4px;
    box-shadow:0 6px 18px rgba(37,99,235,.08);
}
[data-testid="stRadioOption"] {
    margin:0; padding:.38rem 1.1rem; border-radius:999px; cursor:pointer;
}
[data-testid="stRadioOption"] > div > div:first-child { display:none; }
[data-testid="stRadioOption"] p { color:#10244f; font-weight:700; font-size:.92rem; margin:0; }
[data-testid="stRadioOption"][data-selected="true"] { background:#2563eb; }
[data-testid="stRadioOption"][data-selected="true"] p { color:#fff; }

/* Hero */
.badge { display:inline-block; background:#fff; border:1px solid #dfe8f8; color:#2563eb;
    font-weight:700; font-size:.74rem; padding:.42rem .9rem; border-radius:999px;
    margin-top:1.2rem; }
.hero-title { font-size:2rem; font-weight:800; color:#10244f; margin:.9rem 0 .4rem;
    line-height:1.15; }
.hero-sub { color:#6a7a99; font-size:1.02rem; line-height:1.6; margin-bottom:1.4rem; }

/* Form sebagai kartu */
[data-testid="stForm"] {
    background:#fff; border:1px solid #e3ebf9; border-radius:22px; padding:1.6rem;
    box-shadow:0 18px 40px rgba(37,99,235,.09);
}
[data-testid="stForm"] label p { color:#10244f; font-weight:700; font-size:.88rem; }
[data-testid="stTextInputRootElement"], [data-testid="stNumberInputContainer"] {
    background:#eaf1ff !important; border:none !important; border-radius:12px !important;
}
[data-testid="stTextInputRootElement"] input, [data-testid="stNumberInputContainer"] input,
[data-testid="stNumberInputContainer"] button {
    background:transparent !important; color:#10244f;
}
[data-testid="stElementContainer"]:has([data-testid="stFormSubmitButton"]),
[data-testid="stFormSubmitButton"] { width:100% !important; }
[data-testid="stFormSubmitButton"] button {
    width:100% !important; height:3.1rem; border:none; border-radius:14px; color:#fff;
    background:linear-gradient(90deg,#2563eb,#3b82f6);
    box-shadow:0 10px 22px rgba(37,99,235,.28); transition:transform .15s;
}
[data-testid="stFormSubmitButton"] button:hover { transform:translateY(-2px); color:#fff; }
[data-testid="stFormSubmitButton"] button p { color:#fff; font-weight:800; }

/* Kartu */
.card { background:#fff; border:1px solid #e3ebf9; border-radius:22px; padding:1.4rem 1.5rem;
    box-shadow:0 18px 40px rgba(37,99,235,.08); margin-bottom:1.1rem; }
.card-head { display:flex; justify-content:space-between; align-items:center;
    margin-bottom:1rem; }
.card-title { font-weight:800; color:#10244f; font-size:1.05rem; }
.count { background:#e8efff; color:#2563eb; font-weight:800; font-size:.74rem;
    padding:.3rem .75rem; border-radius:999px; }

/* Kartu status */
.status { display:flex; gap:1rem; align-items:center; }
.status-icon { width:46px; height:46px; border-radius:14px; display:flex;
    align-items:center; justify-content:center; font-weight:900; font-size:1.2rem;
    background:#e8efff; color:#2563eb; letter-spacing:1px; }
.status-title { font-weight:800; color:#10244f; font-size:1rem; }
.status-sub { color:#6a7a99; font-size:.86rem; margin-top:.1rem; }
.status.ok .status-icon { background:#dcfce7; color:#15803d; }
.status.no .status-icon { background:#fee2e2; color:#b91c1c; }
.status.ok .status-title { color:#15803d; }
.status.no .status-title { color:#b91c1c; }

/* Tabel */
.tbl { width:100%; border-collapse:collapse; border:1px solid #e3ebf9; font-size:.88rem; }
.tbl th { background:#eef3ff; color:#6a7a99; text-align:left; font-weight:700;
    padding:.7rem .8rem; font-size:.78rem; }
.tbl td { color:#10244f; padding:.78rem .8rem; border-top:1px solid #e9eefb; }
.tbl tr:first-child td { background:#f6f9ff; }
.pill { display:inline-block; font-weight:800; font-size:.72rem; padding:.28rem .7rem;
    border-radius:999px; }
.pill.ok { background:#dcfce7; color:#15803d; }
.pill.no { background:#fee2e2; color:#b91c1c; }
.kosong { text-align:center; color:#6a7a99; padding:2rem 0; font-size:.92rem; }

/* Halaman Tentang */
.flow { display:flex; flex-wrap:wrap; gap:.6rem; align-items:center; margin:.6rem 0 1rem; }
.flow span.step { background:#eef3ff; color:#2563eb; font-weight:800; font-size:.85rem;
    padding:.5rem .9rem; border-radius:12px; }
.flow span.arrow { color:#9db2de; font-weight:800; }
.about p, .about li { color:#3b4a69; font-size:.95rem; line-height:1.65; }
</style>
"""


def h(teks: str) -> str:
    """Rapatkan HTML satu baris agar tidak dianggap blok kode oleh Markdown."""
    return " ".join(baris.strip() for baris in teks.splitlines() if baris.strip())


# ----------------------------------------------------------------------------
# Database (SQLite)
# ----------------------------------------------------------------------------
def koneksi() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with koneksi() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS riwayat (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nama TEXT NOT NULL,
                ipk REAL NOT NULL,
                sks INTEGER NOT NULL,
                status TEXT NOT NULL,
                waktu TEXT NOT NULL
            )"""
        )


def simpan_riwayat(nama: str, ipk: float, sks: int, status: str) -> None:
    with koneksi() as conn:
        conn.execute(
            "INSERT INTO riwayat (nama, ipk, sks, status, waktu) VALUES (?, ?, ?, ?, ?)",
            (nama, ipk, sks, status, datetime.now().isoformat(timespec="seconds")),
        )


def ambil_riwayat(batas: int | None = None) -> list[sqlite3.Row]:
    sql = "SELECT nama, ipk, sks, status FROM riwayat ORDER BY id DESC"
    if batas:
        sql += f" LIMIT {int(batas)}"
    with koneksi() as conn:
        return conn.execute(sql).fetchall()


def hitung_riwayat() -> int:
    with koneksi() as conn:
        return conn.execute("SELECT COUNT(*) FROM riwayat").fetchone()[0]


def hapus_semua_riwayat() -> None:
    with koneksi() as conn:
        conn.execute("DELETE FROM riwayat")


# ----------------------------------------------------------------------------
# Model (memisahkan UI dari model)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def prediksi_status(model, ipk: float, sks: int) -> tuple[str, str]:
    """Kembalikan (status, alasan). Alasan hanya terisi jika aturan IPK minimal berlaku.

    Aturan IPK minimal berada di lapisan logika bisnis, di atas model. Model contoh
    hanya dilatih dari 5 data sehingga belajar memakai SKS saja dan mengabaikan IPK.
    """
    if ipk < IPK_MINIMAL:
        return STATUS_TIDAK, f"IPK di bawah batas minimal {IPK_MINIMAL:.2f}"
    hasil = model.predict([[ipk, sks]])
    return (STATUS_LULUS if hasil[0] == 1 else STATUS_TIDAK), ""


# ----------------------------------------------------------------------------
# Komponen tampilan
# ----------------------------------------------------------------------------
def badge_status(status: str) -> str:
    kelas = "ok" if status == STATUS_LULUS else "no"
    return f'<span class="pill {kelas}">{html.escape(status)}</span>'


def tabel_riwayat(rows) -> str:
    if not rows:
        return '<div class="kosong">Belum ada riwayat. Coba lakukan prediksi pertama Anda.</div>'
    baris = "".join(
        f"<tr><td>{html.escape(r['nama'])}</td><td>{r['ipk']:.2f}</td>"
        f"<td>{r['sks']}</td><td>{badge_status(r['status'])}</td></tr>"
        for r in rows
    )
    return (
        '<table class="tbl"><thead><tr><th>Nama</th><th>IPK</th><th>SKS</th>'
        f"<th>Status</th></tr></thead><tbody>{baris}</tbody></table>"
    )


def kartu_riwayat(rows, total: int) -> str:
    return h(
        f"""<div class="card"><div class="card-head">
        <div class="card-title">Riwayat Prediksi</div>
        <div class="count">{total} data</div></div>"""
    ) + tabel_riwayat(rows) + "</div>"


def kartu_status(hasil: dict | None) -> str:
    if hasil is None:
        kelas, ikon = "", "•••"
        judul = "Menunggu input"
        sub = "Isi form lalu tekan Prediksi &amp; Simpan untuk melihat status."
    else:
        lulus = hasil["status"] == STATUS_LULUS
        kelas, ikon = ("ok", "✓") if lulus else ("no", "✕")
        judul = html.escape(hasil["status"])
        sub = (
            f"{html.escape(hasil['nama'])} • IPK {hasil['ipk']:.2f} • "
            f"{hasil['sks']} SKS • tersimpan ke database."
        )
        if hasil.get("alasan"):
            sub += f"<br>Alasan: {html.escape(hasil['alasan'])}."
    return h(
        f"""<div class="card"><div class="status {kelas}">
        <div class="status-icon">{ikon}</div>
        <div><div class="status-title">{judul}</div>
        <div class="status-sub">{sub}</div></div></div></div>"""
    )


# ----------------------------------------------------------------------------
# Halaman
# ----------------------------------------------------------------------------
def halaman_prediksi(model) -> None:
    kiri, kanan = st.columns([1, 1.1], gap="large")

    with kiri:
        st.markdown(
            h(
                """<span class="badge">Praktikum RPL 10 • MLOps &amp; Streamlit</span>
                <div class="hero-title">Sistem Prediksi Kelulusan</div>
                <div class="hero-sub">Masukkan data mahasiswa, sistem akan menentukan
                kelulusan dan menyimpannya ke database.</div>"""
            ),
            unsafe_allow_html=True,
        )
        with st.form("form_prediksi", clear_on_submit=True):
            st.markdown('<div class="card-title">Data Mahasiswa</div>', unsafe_allow_html=True)
            nama = st.text_input("Nama Mahasiswa", placeholder="Contoh: Budi Santoso", max_chars=60)
            ipk = st.number_input(
                "IPK (0.0 - 4.0)", min_value=0.0, max_value=4.0, value=None,
                step=0.01, format="%.2f", placeholder="0.00",
            )
            sks = st.number_input(
                "Total SKS", min_value=0, max_value=200, value=None, step=1, placeholder="0",
            )
            tombol = st.form_submit_button("Prediksi & Simpan")

        if tombol:
            nama = nama.strip()
            if not nama or ipk is None or sks is None:
                st.warning("Lengkapi nama, IPK, dan total SKS terlebih dahulu.")
            else:
                status, alasan = prediksi_status(model, float(ipk), int(sks))
                simpan_riwayat(nama, float(ipk), int(sks), status)
                st.session_state["hasil"] = {
                    "nama": nama, "ipk": float(ipk), "sks": int(sks),
                    "status": status, "alasan": alasan,
                }

    with kanan:
        st.markdown(kartu_status(st.session_state.get("hasil")), unsafe_allow_html=True)
        st.markdown(kartu_riwayat(ambil_riwayat(batas=6), hitung_riwayat()), unsafe_allow_html=True)


def halaman_riwayat() -> None:
    st.markdown(kartu_riwayat(ambil_riwayat(), hitung_riwayat()), unsafe_allow_html=True)
    with st.expander("Opsi data"):
        st.caption("Hapus seluruh riwayat, misalnya setelah selesai uji coba.")
        if st.button("Hapus semua riwayat"):
            hapus_semua_riwayat()
            st.session_state.pop("hasil", None)
            st.rerun()


def halaman_tentang() -> None:
    st.markdown(
        h(
            """<div class="card about"><div class="card-title">Tentang PrediksiKu</div>
            <p>PrediksiKu adalah aplikasi latihan MLOps dasar: model Machine Learning yang
            sudah dilatih disimpan ke file <b>.pkl</b>, lalu dipanggil oleh antarmuka web
            Streamlit sehingga pengguna tidak perlu menjalankan kode Python.</p>
            <div class="flow"><span class="step">Pengguna</span><span class="arrow">→</span>
            <span class="step">UI Streamlit</span><span class="arrow">→</span>
            <span class="step">Logika Prediksi</span><span class="arrow">→</span>
            <span class="step">Model .pkl</span><span class="arrow">→</span>
            <span class="step">Database SQLite</span></div>
            <ul>
            <li><b>Model:</b> Decision Tree dengan fitur IPK dan Total SKS.</li>
            <li><b>Serialization:</b> model disimpan dengan <code>joblib.dump</code>
            dan dimuat kembali dengan <code>joblib.load</code>.</li>
            <li><b>Aturan IPK minimal:</b> IPK di bawah 2.00 otomatis dinyatakan tidak
            lulus, karena model contoh hanya belajar dari SKS.</li>
            <li><b>Database:</b> setiap prediksi tersimpan di SQLite sebagai riwayat.</li>
            <li><b>Catatan:</b> model dilatih dari data contoh yang sangat kecil, jadi
            hasilnya hanya untuk keperluan belajar.</li></ul></div>"""
        ),
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)

    try:
        model = load_model()
    except FileNotFoundError:
        st.error("File model tidak ditemukan! Jalankan `python 1_latih_model.py` terlebih dahulu.")
        st.stop()

    init_db()

    kolom_logo, kolom_menu = st.columns([1, 1])
    with kolom_logo:
        st.markdown(
            '<div class="brand"><div class="brand-dot"></div>'
            '<div class="brand-name">PrediksiKu</div></div>',
            unsafe_allow_html=True,
        )
    with kolom_menu:
        menu = st.radio("Menu", MENU, horizontal=True, label_visibility="collapsed", key="menu")

    if menu == "Prediksi":
        halaman_prediksi(model)
    elif menu == "Riwayat":
        halaman_riwayat()
    else:
        halaman_tentang()


main()
