"""Particle Swarm Optimization (kontinu) untuk hyperparameter NN."""
import numpy as np


def decode(x):
    """Vektor partikel -> hyperparameter. x = [log10(lr), hidden, dropout, log2(batch)]"""
    return {"lr": float(10 ** x[0]), "hidden": int(round(x[1])),
            "dropout": float(x[2]), "batch_size": int(2 ** int(round(x[3])))}


def run_pso(fitness_fn, bounds, n_particles=10, iterations=12, w=0.7, c1=1.5, c2=1.5, seed=0):
    """fitness_fn(hp dict) -> float (maksimasi). bounds: array (D,2)."""
    rng = np.random.default_rng(seed)
    lo, hi = np.array(bounds, float).T
    D = len(lo)
    vmax = 0.2 * (hi - lo)
    X = rng.uniform(lo, hi, (n_particles, D))
    V = rng.uniform(-vmax, vmax, (n_particles, D))
    pbest, pbest_f = X.copy(), np.array([fitness_fn(decode(x)) for x in X])
    gi = int(np.argmax(pbest_f)); gbest, gbest_f = pbest[gi].copy(), float(pbest_f[gi])
    hist_best, hist_mean, n_evals = [gbest_f], [float(pbest_f.mean())], n_particles
    for it in range(iterations):
        r1, r2 = rng.random((n_particles, D)), rng.random((n_particles, D))
        V = np.clip(w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X), -vmax, vmax)
        X = np.clip(X + V, lo, hi)
        f = np.array([fitness_fn(decode(x)) for x in X]); n_evals += n_particles
        imp = f > pbest_f
        pbest[imp], pbest_f[imp] = X[imp], f[imp]
        gi = int(np.argmax(pbest_f))
        if pbest_f[gi] > gbest_f:
            gbest_f, gbest = float(pbest_f[gi]), pbest[gi].copy()
        hist_best.append(gbest_f); hist_mean.append(float(f.mean()))
    return {"gbest_position": gbest.tolist(), "hp": decode(gbest), "fitness": gbest_f,
            "hist_best": hist_best, "hist_mean": hist_mean, "n_evals": n_evals, "seed": seed}
