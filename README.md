# Intelligent Steel Surface Fault Detection (Kelompok 16)
Hybrid Intelligent System: **GA (seleksi fitur) - PSO (hyperparameter) - MLP (prediksi 7 defect) - Fuzzy Mamdani (keputusan inspeksi) - Streamlit**.
Dataset: UCI Steel Plates Faults (ID 198). TE2514031 Kecerdasan Buatan, ITK, Gasal 2026/2027.

## Struktur
```
app.py               dashboard Streamlit (8 halaman wajib)
train.py             menjalankan eksperimen A-E dan menyimpan artefak
generate_report.py   membuat Laporan_Kelompok16.pdf dari artefak
src/config.py        semua parameter GA/PSO/NN/severity (edit di sini)
src/data.py          loading UCI, label argmax, split stratified, scaler
src/nn.py            MLP NumPy (ReLU, dropout, Adam, class weight, early stopping)
src/ga.py  src/pso.py  src/fuzzy.py   GA, PSO, Fuzzy (implementasi manual)
src/pipeline.py      fitness, independent runs, evaluasi model A-E
artifacts/           hasil training (dibaca dashboard) - WAJIB ikut di-commit
data/                steel_plates.csv (dibuat otomatis saat train)
docs/DEMO_SCRIPT.md  naskah video demo
```

## Cara menjalankan
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py --quick          # cek cepat (~1-2 menit); hasilnya akan ditimpa
python train.py --jobs -1        # eksperimen penuh (5 run GA, 10 run PSO, 5 seed per model)
streamlit run app.py
python generate_report.py        # laporan PDF
```
`train.py` mengunduh dataset via `ucimlrepo` dan menyimpannya ke `data/steel_plates.csv`. Jika tidak ada internet,
unduh manual dari UCI dan simpan sebagai CSV berisi 27 fitur + 7 kolom biner kelas (nama kolom kelas persis seperti di `src/config.py`).
Opsi `--synthetic` hanya untuk uji kode, **jangan** untuk laporan.

## Eksperimen
| Model | Komponen |
|---|---|
| A | Baseline MLP, semua fitur, hyperparameter default |
| B | GA + MLP |
| C | PSO + MLP (semua fitur) |
| D | GA + PSO + MLP |
| E | D + Fuzzy (sistem final dashboard) |

Anti-leakage: split stratified train/val/test (60/20/20); scaler hanya di-fit pada sub-train; GA & PSO hanya memakai
sub-train + validation; test hanya untuk evaluasi akhir. Hasil stokastik GA/PSO dilaporkan dari beberapa independent run (best/mean/std);
metrik model dari beberapa seed (mean +- std).

## Deploy ke Streamlit Community Cloud (gratis, publik)
1. Jalankan `python train.py` lokal sampai selesai; pastikan `artifacts/` dan `data/steel_plates.csv` terisi.
2. Push seluruh folder ke repo GitHub **publik** (termasuk `artifacts/` dan `data/`).
3. Buka share.streamlit.io, login GitHub, **New app**, pilih repo, branch, file utama `app.py`, Deploy.
4. Buka URL di jendela incognito untuk memastikan bisa diakses tanpa login, lalu tulis URL di laporan.

## Catatan penting untuk kelompok
- Nilai severity per kelas (`SEVERITY` di `src/config.py`) adalah asumsi; ubah dan justifikasi di laporan.
- Bagian `[ISI]` di laporan wajib ditulis kelompok sendiri (analisis & kesimpulan). Setiap anggota harus bisa menjelaskan
  fitness GA, update PSO, backprop MLP, rule fuzzy, dan menjalankan dashboard saat demo.
