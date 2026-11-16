"""Dashboard Streamlit - Intelligent Steel Surface Fault Detection (Kelompok 16)."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from src import config as C
from src.nn import MLP
from src.data import Scaler
from src import fuzzy as FZ

st.set_page_config(page_title="Steel Fault Detection - Kel. 16", layout="wide")
MEMBERS = ["Syahmi", "Marsa Naufal",
           "Muhammad Irvan Hannif", "Muhammad Ramdi Pratama"]


@st.cache_data
def load_json(name):
    p = C.ART / name
    return json.load(open(p)) if p.exists() else None


@st.cache_data
def load_df():
    if not C.DATA_CSV.exists():
        return None
    df = pd.read_csv(C.DATA_CSV)
    df["Class"] = df[C.CLASSES].values.argmax(1)
    df["Class"] = df["Class"].map(dict(enumerate(C.CLASSES)))
    return df


@st.cache_resource
def load_model():
    return MLP.load(C.ART / "model_final.npz")


meta, ga, pso, res = load_json("meta.json"), load_json(
    "ga.json"), load_json("pso.json"), load_json("results.json")
df = load_df()

PAGES = ["Project Overview", "Dataset Explorer", "GA Optimization", "PSO Optimization",
         "Neural Network", "Fuzzy Inference", "Prediction / Simulation", "Model Comparison"]
page = st.sidebar.radio("Menu", PAGES)
st.sidebar.caption("TE2514031 Kecerdasan Buatan - Gasal 2026/2027")
if meta is None or df is None:
    st.error(
        "Artefak belum ada. Jalankan `python train.py` terlebih dahulu (lihat README).")
    st.stop()
if meta.get("synthetic"):
    st.warning(
        "Artefak ini dibuat dari DATA SINTETIS (uji coba). Jalankan `python train.py` dengan data UCI asli.")


def confusion_fig(cm, title=""):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(7))
    ax.set_yticks(range(7))
    ax.set_xticklabels(C.CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(C.CLASSES)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def mf_fig(mfs, lo, hi, title, xlabel):
    x = np.linspace(lo, hi, 400)
    fig, ax = plt.subplots(figsize=(5, 2.6))
    for k, v in mfs.items():
        ax.plot(x, FZ.mf(v, x), label=k)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


# ======================= 1. OVERVIEW =======================
if page == PAGES[0]:
    st.title("Intelligent Steel Surface Fault Detection")
    st.caption(
        "Hybrid Intelligent System: GA - PSO - Neural Network (MLP) - Fuzzy Logic - Streamlit")
    st.markdown("**Kelompok 16:** " + ", ".join(MEMBERS))
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Problem")
        st.write("Inspeksi permukaan pelat baja secara manual lambat dan subjektif. Sistem ini mengklasifikasikan "
                 "7 jenis cacat dari 27 fitur geometri/luminositas, lalu menerjemahkan hasil prediksi menjadi "
                 "keputusan inspeksi yang mudah dipahami: **Accept - Monitor - Rework - Reject**.")
        st.subheader("Dataset")
        st.write("UCI Steel Plates Faults (ID 198): 1.941 sampel, 27 fitur, 7 kelas "
                 "(Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps, Other_Faults).")
        st.subheader("Alasan memilih MLP")
        st.write("Fitur bersifat tabular tanpa urutan/spasial, sehingga 1D-CNN tidak mendapat keuntungan struktural. "
                 "MLP lebih sederhana, cepat dilatih untuk ribuan evaluasi fitness GA/PSO, dan mudah dijelaskan.")
    with c2:
        st.subheader("Arsitektur sistem")
        st.graphviz_chart("""digraph G { rankdir=LR; node [shape=box, style=rounded];
          D[label="Dataset UCI 198"]; P[label="Preprocessing\\n(split + scaling)"]; GA[label="GA\\nSeleksi fitur"];
          PSO[label="PSO\\nHyperparameter"]; NN[label="MLP\\nPrediksi 7 kelas"]; F[label="Fuzzy Mamdani\\nPrioritas inspeksi"]; S[label="Streamlit"];
          D->P->GA->PSO->NN->F->S; }""")
        st.write(f"Split: train {meta['split']['train']} / validation {meta['split']['val']} / test {meta['split']['test']} "
                 "(stratified). GA dan PSO hanya memakai train+validation; test hanya untuk evaluasi akhir.")

# ======================= 2. DATASET =======================
elif page == PAGES[1]:
    st.title("Dataset Explorer")
    feats = meta["all_features"]
    a, b, c = st.columns(3)
    a.metric("Jumlah sampel", len(df))
    b.metric("Jumlah fitur", len(feats))
    c.metric("Missing values", int(df[feats].isna().sum().sum()))
    st.subheader("Distribusi kelas (target)")
    vc = df["Class"].value_counts().reindex(C.CLASSES)
    st.bar_chart(vc)
    st.caption("Kelas tidak seimbang -> evaluasi memakai F1-macro, precision/recall per kelas, dan confusion matrix; "
               "training memakai class weight.")
    st.subheader("Statistik deskriptif")
    st.dataframe(df[feats].describe().T)
    st.subheader("Korelasi fitur")
    corr = df[feats].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats)))
    ax.set_yticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=90, fontsize=6)
    ax.set_yticklabels(feats, fontsize=6)
    fig.colorbar(im)
    fig.tight_layout()
    st.pyplot(fig)
    st.subheader("Visualisasi fitur per kelas")
    f = st.selectbox("Pilih fitur", feats)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.boxplot([df.loc[df.Class == k, f].dropna()
               for k in C.CLASSES], showfliers=False)
    ax.set_xticklabels(C.CLASSES, rotation=30)
    fig.tight_layout()
    st.pyplot(fig)
    with st.expander("Data mentah"):
        st.dataframe(df.head(200))

# ======================= 3. GA =======================
elif page == PAGES[2]:
    st.title("Genetic Algorithm - Feature Selection")
    cfg = ga["config"]
    s = ga["stats"]
    st.markdown(f"**Kromosom:** vektor biner panjang {len(ga['all_features'])} (1 = fitur dipakai). "
                f"**Fitness:** F1-macro validasi - {cfg['penalty']} x (fitur terpilih / total fitur).")
    st.json({k: v for k, v in cfg.items()})
    st.subheader("Convergence curve (run terbaik)")
    best = max(ga["runs"], key=lambda r: r["fitness"])
    st.line_chart(pd.DataFrame(
        {"best": best["hist_best"], "mean": best["hist_mean"]}))
    st.subheader(f"Independent runs (n={len(ga['runs'])})")
    c1, c2, c3 = st.columns(3)
    c1.metric("Best", f"{s['best']:.4f}")
    c2.metric("Mean", f"{s['mean']:.4f}")
    c3.metric("Std", f"{s['std']:.4f}")
    st.line_chart(pd.DataFrame(
        {f"run {i+1}": r["hist_best"] for i, r in enumerate(ga["runs"])}))
    st.subheader("Fitur terpilih vs dieliminasi")
    chrom = ga["best_chromosome"]
    sel = [f for f, k in zip(ga["all_features"], chrom) if k]
    rem = [f for f, k in zip(ga["all_features"], chrom) if not k]
    c1, c2 = st.columns(2)
    c1.success(f"Terpilih ({len(sel)})\n\n" + ", ".join(sel))
    c2.error(f"Dieliminasi ({len(rem)})\n\n" + (", ".join(rem) or "-"))
    st.code("Chromosome: " + "".join(map(str, chrom)))
    freq = pd.Series(np.mean([r["chromosome"] for r in ga["runs"]], axis=0),
                     index=ga["all_features"], name="frekuensi terpilih antar-run")
    st.subheader("Stabilitas seleksi antar-run")
    st.bar_chart(freq)

# ======================= 4. PSO =======================
elif page == PAGES[3]:
    st.title("Particle Swarm Optimization - Hyperparameter")
    st.json(pso["config"])
    st.subheader("Search space")
    st.table(pd.DataFrame([{"Dimensi": "learning rate", "Rentang": "10^(-4) s.d. 10^(-1.5)"},
                           {"Dimensi": "hidden units", "Rentang": "8 s.d. 128"},
                           {"Dimensi": "dropout", "Rentang": "0.0 s.d. 0.5"},
                           {"Dimensi": "batch size", "Rentang": "2^4 s.d. 2^7 (16-128)"}]))
    key = st.radio("Skenario PSO", ["ga_features", "all_features"],
                   format_func=lambda k: "Pada fitur hasil GA (Model D)" if k == "ga_features" else "Pada semua fitur (Model C)", horizontal=True)
    blk = pso[key]
    s = blk["stats"]
    best = max(blk["runs"], key=lambda r: r["fitness"])
    st.subheader("Convergence curve (gbest per iterasi)")
    st.line_chart(pd.DataFrame(
        {f"run {i+1}": r["hist_best"] for i, r in enumerate(blk["runs"])}))
    c1, c2, c3 = st.columns(3)
    c1.metric("Best", f"{s['best']:.4f}")
    c2.metric("Mean", f"{s['mean']:.4f}")
    c3.metric("Std", f"{s['std']:.4f}")
    st.subheader("Global best & hyperparameter terbaik")
    st.json({"fitness (F1-macro val)": best["fitness"],
            "gbest_position": best["gbest_position"], "hyperparameter": best["hp"]})
    st.dataframe(pd.DataFrame(
        [{"run": i + 1, "fitness": r["fitness"], **r["hp"]} for i, r in enumerate(blk["runs"])]))

# ======================= 5. NN =======================
elif page == PAGES[4]:
    st.title("Neural Network (MLP)")
    r = res["D: GA + PSO + NN"]
    hp = r["hp"]
    st.markdown(f"**Arsitektur:** Input {r['n_features']} -> Dense({hp['hidden']}, ReLU, dropout {hp['dropout']:.2f}) -> "
                f"Dense({hp['hidden']}, ReLU, dropout {hp['dropout']:.2f}) -> Dense(7, Softmax). "
                f"Optimizer Adam (lr={hp['lr']:.5f}), batch {hp['batch_size']}, class-weighted cross-entropy, early stopping.")
    st.metric("Jumlah parameter", f"{r['n_params']:,}")
    h = r["history"]
    st.subheader("Training / validation curve")
    st.line_chart(pd.DataFrame(
        {"train_loss": h["train_loss"], "val_loss": h["val_loss"]}))
    st.line_chart(pd.DataFrame({"val_accuracy": h["val_acc"]}))
    st.subheader("Evaluasi pada test set (mean +- std, 5 seed)")
    m = r["metrics"]
    st.table(pd.DataFrame(
        {k: f"{v['mean']:.4f} +- {v['std']:.4f}" for k, v in m.items()}, index=["nilai"]).T)
    c1, c2 = st.columns([1, 1])
    c1.pyplot(confusion_fig(r["confusion_matrix"], "Confusion matrix (test)"))
    pc = pd.DataFrame(r["per_class"]).T.loc[C.CLASSES, [
        "precision", "recall", "f1-score", "support"]]
    c2.dataframe(pc.round(3))

# ======================= 6. FUZZY =======================
elif page == PAGES[5]:
    st.title("Fuzzy Inference System (Mamdani)")
    st.write("Input fuzzy: **severity** (rata-rata tertimbang tingkat keparahan kelas dengan bobot probabilitas NN) dan "
             "**confidence** (probabilitas tertinggi NN). Output: **priority** 0-100 -> Accept/Monitor/Rework/Reject. "
             "Metode: AND=min, implikasi=clip, agregasi=max, defuzzifikasi=centroid.")
    c1, c2, c3 = st.columns(3)
    c1.pyplot(mf_fig(FZ.SEV_MF, 0, 10, "Severity", "skor 0-10"))
    c2.pyplot(mf_fig(FZ.CONF_MF, 0, 1, "Confidence", "prob. tertinggi"))
    c3.pyplot(mf_fig(FZ.OUT_MF, 0, 100, "Priority (output)", "skor 0-100"))
    st.subheader("Severity per kelas (asumsi kelompok)")
    st.table(pd.DataFrame(meta["severity"], index=["severity"]))
    st.subheader("Rule base")
    st.table(pd.DataFrame(FZ.RULES, columns=["Severity", "Confidence", "Priority"]).assign(
        Rule=lambda d: "IF severity=" + d.Severity + " AND confidence=" + d.Confidence + " THEN " + d.Priority)[["Rule"]])
    st.subheader("Uji interaktif")
    sv = st.slider("Severity", 0.0, 10.0, 6.0, 0.1)
    cf = st.slider("Confidence", 0.0, 1.0, 0.8, 0.01)
    o = FZ.infer(sv, cf)
    fig, ax = plt.subplots(figsize=(7, 2.8))
    for k, v in FZ.OUT_MF.items():
        ax.plot(o["universe"], FZ.mf(v, o["universe"]),
                "--", alpha=.4, label=k)
    ax.fill_between(o["universe"], o["aggregated"],
                    alpha=.5, color="tab:orange")
    ax.axvline(o["priority"], color="r")
    ax.set_title(f"Agregasi & centroid = {o['priority']:.1f}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    st.pyplot(fig)
    st.success(f"Keputusan: **{o['label']}** (priority {o['priority']:.1f})")
    st.dataframe(pd.DataFrame(o["fired"]).query(
        "strength > 0")[["rule", "strength"]])
    if "fuzzy" in res["E: GA + PSO + NN + Fuzzy"]:
        fz = res["E: GA + PSO + NN + Fuzzy"]["fuzzy"]
        st.subheader("Keputusan fuzzy pada test set")
        st.bar_chart(pd.Series(fz["distribution"]))
        st.dataframe(pd.DataFrame(fz["crosstab"]).T)

# ======================= 7. PREDICTION =======================
elif page == PAGES[6]:
    st.title("Prediction / Simulation")
    model = load_model()
    sel = meta["selected_features"]
    sc = Scaler()
    sc.mean_, sc.scale_ = np.array(
        meta["scaler_mean"]), np.array(meta["scaler_scale"])
    st.caption(f"Model memakai {len(sel)} fitur hasil GA.")

    def show(proba):
        o = FZ.decide(proba)
        c1, c2 = st.columns(2)
        c1.bar_chart(pd.Series(proba, index=C.CLASSES, name="probabilitas"))
        c2.metric("Prediksi kelas", C.CLASSES[int(
            np.argmax(proba))], f"conf {proba.max():.2f}")
        c2.metric("Expected severity", f"{o['severity']:.2f} / 10")
        c2.metric("Fuzzy priority", f"{o['priority']:.1f}")
        {"Accept": c2.success, "Monitor": c2.info, "Rework": c2.warning,
            "Reject": c2.error}[o["label"]](f"Keputusan inspeksi: {o['label']}")
        with c2.expander("Rule yang aktif"):
            st.dataframe(pd.DataFrame(o["fired"]).query(
                "strength > 0")[["rule", "strength"]])

    tab1, tab2 = st.tabs(["Input manual", "Upload CSV"])
    with tab1:
        if st.button("Isi dengan sampel acak dari dataset"):
            row = df.sample(1).iloc[0]
            for f in sel:
                st.session_state[f"in_{f}"] = float(row[f])
            st.session_state["actual"] = row["Class"]
        if "actual" in st.session_state:
            st.caption(f"Kelas aktual sampel: {st.session_state['actual']}")
        cols = st.columns(3)
        vals = []
        for i, f in enumerate(sel):
            s = meta["feature_stats"][f]
            vals.append(cols[i % 3].number_input(f, value=float(st.session_state.get(f"in_{f}", s["median"])),
                                                 min_value=None, max_value=None, key=f"in_{f}",
                                                 help=f"min {s['min']:.3g} | max {s['max']:.3g}"))
        proba = model.predict_proba(sc.transform(np.array(vals)[None, :]))[0]
        show(proba)
    with tab2:
        up = st.file_uploader(
            f"CSV harus memuat kolom: {', '.join(sel)}", type="csv")
        if up is not None:
            d = pd.read_csv(up)
            miss = [f for f in sel if f not in d.columns]
            if miss:
                st.error(f"Kolom hilang: {miss}")
            else:
                P_ = model.predict_proba(sc.transform(d[sel].values))
                out = d.copy()
                out["pred_class"] = [C.CLASSES[i] for i in P_.argmax(1)]
                out["confidence"] = P_.max(1)
                dec = [FZ.decide(p) for p in P_]
                out["fuzzy_priority"] = [x["priority"] for x in dec]
                out["decision"] = [x["label"] for x in dec]
                st.dataframe(out)
                st.bar_chart(out["decision"].value_counts())
                st.download_button("Unduh hasil CSV", out.to_csv(
                    index=False), "hasil_prediksi.csv")
                i = st.number_input("Lihat detail baris ke-", 0, len(d) - 1, 0)
                show(P_[int(i)])

# ======================= 8. COMPARISON =======================
else:
    st.title("Model Comparison")
    cmp_ = pd.read_csv(C.ART / "comparison.csv")
    st.dataframe(cmp_.style.format(
        {c: "{:.4f}" for c in cmp_.columns if c not in ("Model", "n_fitur", "Params")}))
    st.subheader("F1-macro (test, mean +- std 5 seed)")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(cmp_["Model"], cmp_["F1(macro)"], yerr=cmp_[
           "F1(macro) std"], capsize=4, color="tab:blue")
    plt.xticks(rotation=20, ha="right")
    ax.set_ylabel("F1-macro")
    fig.tight_layout()
    st.pyplot(fig)
    st.subheader("Fitur & hyperparameter tiap model")
    st.dataframe(pd.DataFrame(
        [{"Model": k, "n_fitur": v["n_features"], **v["hp"]} for k, v in res.items()]))
    st.info("Model E memakai NN yang sama dengan D; fuzzy tidak mengubah akurasi, tetapi menambah lapisan keputusan "
            "yang interpretable (lihat halaman Fuzzy Inference). Interpretasi dan analisis sebab-akibat ditulis di laporan.")
