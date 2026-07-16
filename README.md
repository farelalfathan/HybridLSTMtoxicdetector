# ClarityNLP — Toxic Comment Detector (Streamlit)

Dashboard untuk model **Hybrid LSTM-GRU** (klasifikasi biner *toxic* / *non-toxic*).
Semua angka di dashboard diambil langsung dari `artifacts/metrics.json` hasil evaluasi
di data uji — bukan angka contoh/hardcode.

## Struktur folder

```
.
├── app.py
├── requirements.txt
├── README.md
└── artifacts/              <- WAJIB diisi sebelum dijalankan (lihat langkah di bawah)
    ├── model.keras
    ├── tokenizer.json
    ├── config.json
    ├── threshold.json
    └── metrics.json
```

## Langkah 1 — Ambil artifact dari notebook training

1. Jalankan notebook training (`Hybrid_LSTM_GRU_Toxic_Comment_Classification_FIXED.ipynb`)
   dari Cell 1 sampai **Cell 14** ("EXPORT ARTIFACTS UNTUK STREAMLIT").
   Cell 14 otomatis mendownload `artifacts_for_streamlit.zip` ke komputermu (lewat
   `google.colab.files.download`) — tidak perlu mount Google Drive.
2. Ekstrak isi zip tersebut ke folder `artifacts/` di repo ini, sehingga persis seperti
   struktur folder di atas (nama file **jangan diubah**: `model.keras`, `tokenizer.json`,
   `config.json`, `threshold.json`, `metrics.json`).

## Langkah 2 — Jalankan lokal (opsional, untuk cek dulu)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Langkah 3 — Push ke GitHub

```bash
git init
git add .
git commit -m "Initial commit: ClarityNLP toxic comment dashboard"
git branch -M main
git remote add origin https://github.com/<username>/<repo-name>.git
git push -u origin main
```

**Catatan ukuran file:** `model.keras` untuk arsitektur ini biasanya sekitar 30-40MB,
masih di bawah batas keras GitHub (100MB/file). Kalau model kamu ternyata lebih besar
dari 50MB, GitHub akan memperingatkan — pakai [Git LFS](https://git-lfs.com/) untuk
file tersebut:
```bash
git lfs install
git lfs track "artifacts/*.keras"
git add .gitattributes
```

## Langkah 4 — Deploy ke Streamlit Community Cloud

1. Buka [share.streamlit.io](https://share.streamlit.io), login dengan akun GitHub.
2. Klik **New app**, pilih repo dan branch di atas, set **Main file path** ke `app.py`.
3. Klik **Deploy**. Build pertama biasanya makan waktu beberapa menit karena TensorFlow
   cukup besar untuk di-install.

## Kalau model dilatih ulang

Ulangi Langkah 1 (download ulang `artifacts_for_streamlit.zip`, timpa folder `artifacts/`),
commit & push lagi. Karena semua angka dashboard dibaca dari `metrics.json`, tidak ada
kode di `app.py` yang perlu diubah manual.

## Konsistensi dengan Penulisan Ilmiah

- Label yang dipakai konsisten **Non-Toxic / Toxic** (bukan "Hate Speech Terdeteksi"),
  sesuai pembahasan BAB II mengenai perbedaan *hate speech* sebagai subkategori spesifik
  vs *toxic* secara umum (Davidson et al., 2017).
- Fungsi `clean_text` di `app.py` disalin persis dari Cell 4 notebook training, supaya
  teks yang diproses saat inference identik dengan saat training (tidak ada *train-serve
  skew*).
- Arsitektur: Hybrid LSTM-GRU **paralel** (dua cabang terpisah dari embedding yang sama,
  masing-masing di-*pooling*, digabung lewat *concatenate*) — sesuai Gambar 3.4 di BAB III.
