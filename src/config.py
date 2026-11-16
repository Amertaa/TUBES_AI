"""Konfigurasi terpusat. Ubah angka di sini untuk eksperimen yang lebih berat/ringan."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "steel_plates.csv"
ART = ROOT / "artifacts"

SEED = 42
CLASSES = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]

# Severity (0-10) tiap kelas defect: ASUMSI KELOMPOK (silakan didiskusikan/disesuaikan di laporan).
SEVERITY = {"Pastry": 4, "Z_Scratch": 5, "K_Scatch": 8, "Stains": 2,
            "Dirtiness": 1, "Bumps": 6, "Other_Faults": 5}

# Hyperparameter default untuk Baseline (Model A) dan Model B
DEFAULT_HP = {"lr": 1e-3, "hidden": 32, "dropout": 0.0, "batch_size": 32}

# ---- GA ----
GA = {"pop_size": 20, "generations": 15, "crossover_rate": 0.8, "mutation_rate": 0.03,
      "tournament_k": 3, "elitism": 2, "penalty": 0.02}   # penalty * (n_fitur/n_total)

# ---- PSO ----  (dimensi: log10(lr), hidden units, dropout, log2(batch_size))
PSO_SPACE = {"lr_log10": (-4.0, -1.5), "hidden": (8, 128), "dropout": (0.0, 0.5), "batch_log2": (4, 7)}
PSO = {"n_particles": 10, "iterations": 12, "w": 0.7, "c1": 1.5, "c2": 1.5}

N_RUNS = 5            # independent runs GA & PSO (best/mean/std)
N_FINAL_SEEDS = 5     # seed training akhir tiap model untuk mean±std test
FIT_EPOCHS = 60       # epoch maksimum saat evaluasi fitness (cepat)
FINAL_EPOCHS = 200    # epoch maksimum training akhir
PATIENCE = 15
