"""Jalankan seluruh eksperimen & simpan artefak ke artifacts/.
   python train.py                 -> penuh (download UCI otomatis)
   python train.py --quick         -> versi cepat untuk cek kode
   python train.py --synthetic     -> data sintetis (HANYA smoke test)
"""
import argparse, json, time
import numpy as np
import pandas as pd
from src import config as C
from src.data import load_dataset, three_way_split
from src import pipeline as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--jobs", type=int, default=1, help="paralel independent runs (-1 = semua core)")
    a = ap.parse_args()

    ga_cfg, pso_cfg = dict(C.GA), dict(C.PSO)
    n_runs, n_seeds, f_ep = C.N_RUNS, C.N_FINAL_SEEDS, C.FINAL_EPOCHS
    if a.quick:
        ga_cfg.update(pop_size=8, generations=4); pso_cfg.update(n_particles=5, iterations=4)
        n_runs, n_seeds, f_ep = 2, 2, 60
        C.FIT_EPOCHS = 30

    t0 = time.time()
    C.ART.mkdir(exist_ok=True)
    X, y = load_dataset(a.synthetic)
    S = three_way_split(X, y)
    all_cols = list(X.columns)
    print(f"Data: {X.shape}, kelas: {np.bincount(y).tolist()}  | train/val/test = "
          f"{len(S['y_sub'])}/{len(S['y_val'])}/{len(S['y_te'])}")

    # ---- GA (independent runs) ----
    print("[GA] seleksi fitur ..."); ga_runs = P.run_ga_multi(S, all_cols, ga_cfg, n_runs, a.jobs)
    ga_best = max(ga_runs, key=lambda r: r["fitness"])
    ga_cols = [c for c, k in zip(all_cols, ga_best["chromosome"]) if k]
    print(f"    best fitness={ga_best['fitness']:.4f}, fitur terpilih {len(ga_cols)}/{len(all_cols)}")

    # ---- PSO pada semua fitur (Model C) dan pada fitur GA (Model D) ----
    print("[PSO] semua fitur (Model C) ..."); pso_all = P.run_pso_multi(S, all_cols, pso_cfg, n_runs, a.jobs)
    print("[PSO] fitur hasil GA (Model D) ..."); pso_ga = P.run_pso_multi(S, ga_cols, pso_cfg, n_runs, a.jobs)
    hp_c = max(pso_all, key=lambda r: r["fitness"])["hp"]
    hp_d = max(pso_ga, key=lambda r: r["fitness"])["hp"]

    # ---- Evaluasi akhir pada TEST ----
    cfgs = {"A: Baseline NN": (all_cols, C.DEFAULT_HP), "B: GA + NN": (ga_cols, C.DEFAULT_HP),
            "C: PSO + NN": (all_cols, hp_c), "D: GA + PSO + NN": (ga_cols, hp_d)}
    results, best_models = {}, {}
    for name, (cols, hp) in cfgs.items():
        print(f"[EVAL] {name}")
        results[name], best_models[name] = P.evaluate_model(S, cols, hp, n_seeds, f_ep, C.PATIENCE)
        print(f"    F1-macro test = {results[name]['metrics']['f1_macro']['mean']:.4f}")

    # ---- Model E = D + Fuzzy ----
    bd = best_models["D: GA + PSO + NN"]
    results["E: GA + PSO + NN + Fuzzy"] = {**results["D: GA + PSO + NN"], "fuzzy": P.fuzzy_report(bd["proba"], S["y_te"])}
    print("[FUZZY]", results["E: GA + PSO + NN + Fuzzy"]["fuzzy"]["distribution"])

    # ---- Simpan artefak ----
    bd["m"].save(C.ART / "model_final.npz")
    Xtr = S["X_tr"][ga_cols]
    meta = {"classes": C.CLASSES, "all_features": all_cols, "selected_features": ga_cols,
            "scaler_mean": bd["sc"].mean_.tolist(), "scaler_scale": bd["sc"].scale_.tolist(),
            "hp_final": hp_d, "severity": C.SEVERITY,
            "feature_stats": {c: {"min": float(Xtr[c].min()), "max": float(Xtr[c].max()),
                                  "median": float(Xtr[c].median())} for c in ga_cols},
            "split": {"train": len(S["y_sub"]), "val": len(S["y_val"]), "test": len(S["y_te"])},
            "synthetic": bool(a.synthetic), "quick": bool(a.quick), "runtime_sec": round(time.time() - t0, 1)}
    json.dump(meta, open(C.ART / "meta.json", "w"), indent=1)
    json.dump({"config": {k: v for k, v in ga_cfg.items()}, "runs": ga_runs, "stats": P.stats(ga_runs),
               "all_features": all_cols, "selected": ga_cols,
               "best_chromosome": ga_best["chromosome"]}, open(C.ART / "ga.json", "w"), indent=1)
    json.dump({"config": pso_cfg, "space": C.PSO_SPACE,
               "all_features": {"runs": pso_all, "stats": P.stats(pso_all)},
               "ga_features": {"runs": pso_ga, "stats": P.stats(pso_ga)}}, open(C.ART / "pso.json", "w"), indent=1)
    json.dump(results, open(C.ART / "results.json", "w"), indent=1)
    np.savez(C.ART / "test_pred.npz", proba=bd["proba"], y_true=S["y_te"])
    rows = [{"Model": k, "n_fitur": v["n_features"], "Accuracy": v["metrics"]["accuracy"]["mean"],
             "Precision(macro)": v["metrics"]["precision_macro"]["mean"], "Recall(macro)": v["metrics"]["recall_macro"]["mean"],
             "F1(macro)": v["metrics"]["f1_macro"]["mean"], "F1(macro) std": v["metrics"]["f1_macro"]["std"],
             "Params": v["n_params"]} for k, v in results.items()]
    pd.DataFrame(rows).to_csv(C.ART / "comparison.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(f"Selesai dalam {time.time() - t0:.0f} detik. Artefak di {C.ART}")


if __name__ == "__main__":
    main()
