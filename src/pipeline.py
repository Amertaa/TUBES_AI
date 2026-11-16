"""Eksperimen A-E: Baseline, GA+NN, PSO+NN, GA+PSO+NN, GA+PSO+NN+Fuzzy."""
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, f1_score,
                             confusion_matrix, classification_report)
from . import config as C
from .data import Scaler
from .nn import MLP
from .ga import run_ga
from .pso import run_pso
from .fuzzy import decide


def _fit_eval(S, cols, hp, seed, epochs, patience, eval_on="val"):
    sc = Scaler().fit(S["X_sub"][cols])
    Xs, Xv = sc.transform(S["X_sub"][cols]), sc.transform(S["X_val"][cols])
    m = MLP(len(cols), len(C.CLASSES), hidden=hp["hidden"], lr=hp["lr"], dropout=hp["dropout"],
            batch_size=hp["batch_size"], epochs=epochs, patience=patience, seed=seed)
    m.fit(Xs, S["y_sub"], Xv, S["y_val"])
    return m, sc


def make_ga_fitness(S, all_cols, hp, penalty):
    def f(mask):
        cols = [c for c, k in zip(all_cols, mask) if k]
        m, sc = _fit_eval(S, cols, hp, 0, C.FIT_EPOCHS, 8)
        pred = m.predict(sc.transform(S["X_val"][cols]))
        return f1_score(S["y_val"], pred, average="macro", zero_division=0) - penalty * (len(cols) / len(all_cols))
    return f


def make_pso_fitness(S, cols):
    def f(hp):
        m, sc = _fit_eval(S, cols, hp, 0, C.FIT_EPOCHS, 8)
        pred = m.predict(sc.transform(S["X_val"][cols]))
        return f1_score(S["y_val"], pred, average="macro", zero_division=0)
    return f


def stats(runs):
    f = np.array([r["fitness"] for r in runs])
    return {"best": float(f.max()), "mean": float(f.mean()), "std": float(f.std()), "worst": float(f.min()),
            "all": f.tolist()}


def run_ga_multi(S, all_cols, cfg, n_runs, jobs):
    fit = make_ga_fitness(S, all_cols, C.DEFAULT_HP, cfg["penalty"])
    kw = {k: v for k, v in cfg.items() if k != "penalty"}
    runs = Parallel(n_jobs=jobs)(delayed(run_ga)(fit, len(all_cols), seed=100 + i, **kw) for i in range(n_runs))
    return runs


def run_pso_multi(S, cols, cfg, n_runs, jobs):
    fit = make_pso_fitness(S, cols)
    bounds = [C.PSO_SPACE["lr_log10"], C.PSO_SPACE["hidden"], C.PSO_SPACE["dropout"], C.PSO_SPACE["batch_log2"]]
    runs = Parallel(n_jobs=jobs)(delayed(run_pso)(fit, bounds, seed=200 + i, **cfg) for i in range(n_runs))
    return runs


def evaluate_model(S, cols, hp, n_seeds, epochs, patience):
    """Latih n_seeds kali, evaluasi pada TEST. Return ringkasan + model/scaler terbaik (by val F1)."""
    rows, best = [], None
    for s in range(n_seeds):
        m, sc = _fit_eval(S, cols, hp, s, epochs, patience)
        Xv, Xt = sc.transform(S["X_val"][cols]), sc.transform(S["X_te"][cols])
        vf1 = f1_score(S["y_val"], m.predict(Xv), average="macro", zero_division=0)
        pt = m.predict_proba(Xt); yp = pt.argmax(1)
        p, r, f, _ = precision_recall_fscore_support(S["y_te"], yp, average="macro", zero_division=0)
        rows.append({"accuracy": accuracy_score(S["y_te"], yp), "precision_macro": p, "recall_macro": r,
                     "f1_macro": f, "f1_weighted": f1_score(S["y_te"], yp, average="weighted", zero_division=0)})
        if best is None or vf1 > best["vf1"]:
            best = {"vf1": vf1, "m": m, "sc": sc, "proba": pt, "yp": yp}
    agg = {k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows]))} for k in rows[0]}
    rep = classification_report(S["y_te"], best["yp"], target_names=C.CLASSES, output_dict=True, zero_division=0)
    return {"features": list(cols), "n_features": len(cols), "hp": hp, "metrics": agg,
            "n_params": best["m"].n_params,
            "confusion_matrix": confusion_matrix(S["y_te"], best["yp"], labels=range(len(C.CLASSES))).tolist(),
            "per_class": rep, "history": best["m"].history}, best


def fuzzy_report(proba, y_true):
    """Terapkan fuzzy ke seluruh test set -> distribusi keputusan & crosstab kelas-vs-keputusan."""
    labels = ["Accept", "Monitor", "Rework", "Reject"]
    dec = [decide(p) for p in proba]
    names = [d["label"] for d in dec]
    cross = {c: {l: 0 for l in labels} for c in C.CLASSES}
    for t, n in zip(y_true, names):
        cross[C.CLASSES[t]][n] += 1
    return {"distribution": {l: names.count(l) for l in labels}, "crosstab": cross,
            "mean_priority": float(np.mean([d["priority"] for d in dec]))}
