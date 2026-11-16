"""Buat laporan PDF otomatis dari artifacts/ -> Laporan_Kelompok16.pdf
Angka, tabel, dan grafik terisi otomatis; bagian bertanda [ISI] adalah analisis yang harus kalian tulis sendiri."""
import io, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from src import config as C

A = lambda n: json.load(open(C.ART / n))
meta, ga, pso, res = A("meta.json"), A("ga.json"), A("pso.json"), A("results.json")
ss = getSampleStyleSheet(); H1, H2, B = ss["Heading1"], ss["Heading2"], ss["BodyText"]
S = []
P = lambda t, s=B: S.append(Paragraph(t, s))


def img(fig, w=14):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150, bbox_inches="tight"); plt.close(fig); buf.seek(0)
    im = Image(buf); r = im.imageHeight / im.imageWidth; im.drawWidth = w * cm; im.drawHeight = w * cm * r; S.append(im)


def table(rows, widths=None):
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17375e")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("GRID", (0, 0), (-1, -1), .4, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 8)]))
    S.append(t); S.append(Spacer(1, 8))


def curve(series, title):
    fig, ax = plt.subplots(figsize=(6, 3))
    for k, v in series.items(): ax.plot(v, label=k, alpha=.8)
    ax.set_title(title); ax.set_xlabel("generasi / iterasi"); ax.set_ylabel("fitness"); ax.legend(fontsize=6); return fig


# ---- Cover ----
P("TUGAS BESAR KECERDASAN BUATAN", ss["Title"])
P("Intelligent Steel Surface Fault Detection<br/>Hybrid GA - PSO - MLP - Fuzzy Logic - Streamlit", H2)
P("Kelompok 16: Marsa Naufal, Syahmi, Muhammad Irvan Hannif, Muhammad Ramdi Pratama<br/>"
  "Mata Kuliah TE2514031 Kecerdasan Buatan, Institut Teknologi Kalimantan, Gasal 2026/2027<br/>"
  "URL Dashboard: [ISI URL STREAMLIT] | Repository: [ISI URL GITHUB]")
if meta.get("synthetic"): P("<b>PERINGATAN: artefak berasal dari data sintetis (uji coba).</b>")

# ---- 1 ----
P("1. Pendahuluan", H1)
P("Inspeksi cacat permukaan pelat baja secara manual lambat dan subjektif. Sistem ini mengklasifikasikan tujuh jenis cacat "
  "(Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps, Other_Faults) dan mengubah keluaran model menjadi keputusan "
  "inspeksi Accept/Monitor/Rework/Reject. Alur: Dataset UCI -> preprocessing -> GA -> PSO -> MLP -> Fuzzy -> Streamlit.")

# ---- 2 ----
P("2. Data dan Preprocessing", H1)
sp = meta["split"]
P(f"Dataset UCI Steel Plates Faults (ID 198), 27 fitur numerik. Tujuh kolom biner digabung menjadi satu label kelas. "
  f"Pembagian stratified: sub-train {sp['train']}, validation {sp['val']}, test {sp['test']}. Scaler (z-score) di-fit hanya pada "
  f"sub-train. GA dan PSO hanya memakai sub-train dan validation; test hanya dipakai pada evaluasi akhir (tanpa data leakage). "
  f"Ketidakseimbangan kelas ditangani dengan class-weighted cross-entropy dan metrik F1-macro.")
cm_ = np.array(res["A: Baseline NN"]["confusion_matrix"]); sup = cm_.sum(1)
table([["Kelas"] + C.CLASSES, ["Jumlah di test"] + [str(x) for x in sup]])

# ---- 3 ----
P("3. Metodologi", H1)
c = ga["config"]
P("3.1 Genetic Algorithm", H2)
P(f"Kromosom biner panjang 27 (1 = fitur dipakai). Fitness = F1-macro validasi - {c['penalty']} x rasio fitur terpilih, dievaluasi dengan MLP "
  f"hyperparameter default. Seleksi tournament (k={c['tournament_k']}), uniform crossover (p={c['crossover_rate']}), bit-flip mutation "
  f"(p={c['mutation_rate']}), elitism {c['elitism']}, populasi {c['pop_size']}, generasi {c['generations']}.")
P("3.2 Particle Swarm Optimization", H2)
p = pso["config"]
P(f"Ruang pencarian: log10(lr) in [-4,-1.5], hidden units in [8,128], dropout in [0,0.5], log2(batch) in [4,7]. "
  f"Swarm {p['n_particles']} partikel, {p['iterations']} iterasi, w={p['w']}, c1={p['c1']}, c2={p['c2']}, velocity clamping 20% rentang. "
  f"Fitness = F1-macro validasi.")
P("3.3 Neural Network", H2)
P("MLP dua hidden layer (ReLU, dropout), output softmax 7 kelas, optimizer Adam, early stopping. Dipilih MLP karena data tabular "
  "tanpa struktur urutan/spasial, sehingga 1D-CNN tidak memberi keuntungan struktural. [ISI: tambahkan argumen kelompok]")
P("3.4 Fuzzy Inference System", H2)
P("Mamdani dengan input severity (rata-rata tertimbang severity kelas dengan bobot probabilitas NN, 0-10) dan confidence "
  "(probabilitas maksimum). Output priority 0-100 dengan himpunan Accept, Monitor, Rework, Reject. 9 rule, AND=min, agregasi=max, centroid. "
  f"Severity kelas (asumsi kelompok): {meta['severity']}.")
table([["Severity", "Confidence", "Priority"]] + [list(r) for r in __import__("src.fuzzy", fromlist=["RULES"]).RULES])

# ---- 4 ----
P("4. Hasil Eksperimen", H1)
P("4.1 Hasil GA", H2)
st = ga["stats"]
P(f"{len(ga['runs'])} independent run: best={st['best']:.4f}, mean={st['mean']:.4f}, std={st['std']:.4f}. "
  f"Fitur terpilih ({len(ga['selected'])}): {', '.join(ga['selected'])}. Kromosom terbaik: {''.join(map(str, ga['best_chromosome']))}")
img(curve({f"run {i+1}": r["hist_best"] for i, r in enumerate(ga["runs"])}, "Konvergensi GA"))
P("4.2 Hasil PSO", H2)
rows = [["Skenario", "Best", "Mean", "Std", "lr", "hidden", "dropout", "batch"]]
for k, nm in [("all_features", "PSO semua fitur (C)"), ("ga_features", "PSO fitur GA (D)")]:
    b = max(pso[k]["runs"], key=lambda r: r["fitness"]); s_ = pso[k]["stats"]; h = b["hp"]
    rows.append([nm, f"{s_['best']:.4f}", f"{s_['mean']:.4f}", f"{s_['std']:.4f}", f"{h['lr']:.5f}", h["hidden"], f"{h['dropout']:.2f}", h["batch_size"]])
table(rows)
img(curve({f"run {i+1}": r["hist_best"] for i, r in enumerate(pso["ga_features"]["runs"])}, "Konvergensi PSO (gbest, fitur GA)"))
P("4.3 Perbandingan Model (test set, mean +- std)", H2)
rows = [["Model", "Fitur", "Acc", "Prec(m)", "Rec(m)", "F1(m)", "F1 std", "Param"]]
for k, v in res.items():
    m = v["metrics"]
    rows.append([k, v["n_features"], f"{m['accuracy']['mean']:.3f}", f"{m['precision_macro']['mean']:.3f}", f"{m['recall_macro']['mean']:.3f}",
                 f"{m['f1_macro']['mean']:.3f}", f"{m['f1_macro']['std']:.3f}", v["n_params"]])
table(rows)
fig, ax = plt.subplots(figsize=(6, 3)); ks = list(res)
ax.bar([k.split(":")[0] for k in ks], [res[k]["metrics"]["f1_macro"]["mean"] for k in ks], yerr=[res[k]["metrics"]["f1_macro"]["std"] for k in ks], capsize=4)
ax.set_ylabel("F1-macro"); img(fig, 10)
P("4.4 Model Final: Confusion Matrix dan Metrik per Kelas", H2)
r = res["D: GA + PSO + NN"]; cm_ = np.array(r["confusion_matrix"])
fig, ax = plt.subplots(figsize=(5, 4.3)); ax.imshow(cm_, cmap="Blues"); ax.set_xticks(range(7)); ax.set_yticks(range(7))
ax.set_xticklabels(C.CLASSES, rotation=45, ha="right", fontsize=7); ax.set_yticklabels(C.CLASSES, fontsize=7)
for i in range(7):
    for j in range(7): ax.text(j, i, cm_[i, j], ha="center", va="center", fontsize=7, color="white" if cm_[i, j] > cm_.max() / 2 else "black")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); img(fig, 9)
table([["Kelas", "Precision", "Recall", "F1", "Support"]] + [[k, *(f"{r['per_class'][k][m]:.3f}" for m in ("precision", "recall", "f1-score")), int(r["per_class"][k]["support"])] for k in C.CLASSES])
P("4.5 Hasil Fuzzy pada Test Set", H2)
fz = res["E: GA + PSO + NN + Fuzzy"]["fuzzy"]
table([["Kelas aktual", "Accept", "Monitor", "Rework", "Reject"]] + [[k] + [fz["crosstab"][k][l] for l in ("Accept", "Monitor", "Rework", "Reject")] for k in C.CLASSES])

# ---- 5 ----
P("5. Analisis", H1)
f = {k: v["metrics"]["f1_macro"]["mean"] for k, v in res.items()}
base = f["A: Baseline NN"]
P(f"Terhadap baseline (F1-macro {base:.3f}): GA+NN {f['B: GA + NN']-base:+.3f}, PSO+NN {f['C: PSO + NN']-base:+.3f}, "
  f"GA+PSO+NN {f['D: GA + PSO + NN']-base:+.3f}. GA mengurangi fitur dari 27 menjadi {res['B: GA + NN']['n_features']}. "
  "Selisih perlu dibandingkan dengan standard deviation antar-seed sebelum disimpulkan bermakna.")
P("[ISI: apakah GA membantu? apakah PSO membantu? apakah integrasi D lebih baik dari C? mengapa? kelas mana yang sulit (lihat confusion matrix, "
  "terutama Other_Faults dan kelas minoritas)? apa efek class imbalance? keterbatasan: validasi holdout, jumlah run, severity ditentukan manual.]")
P("[ISI: analisis fuzzy: apakah keputusan masuk akal per kelas? bagaimana efek confidence rendah pada keputusan?]")

# ---- 6 ----
P("6. Kesimpulan", H1)
P("[ISI: kesimpulan 1 paragraf dan saran pengembangan]")
P("Daftar Pustaka", H2)
P("UCI Machine Learning Repository, Steel Plates Faults, ID 198. https://archive.ics.uci.edu/dataset/198/steel+plates+faults")

SimpleDocTemplate("Laporan_Kelompok16.pdf", pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm).build(S)
print("Dibuat: Laporan_Kelompok16.pdf")
