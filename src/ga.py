"""Genetic Algorithm (kromosom biner) untuk seleksi fitur."""
import numpy as np


def run_ga(fitness_fn, n_genes, pop_size=20, generations=15, crossover_rate=0.8,
           mutation_rate=0.03, tournament_k=3, elitism=2, seed=0):
    """fitness_fn(mask: bool array) -> float (maksimasi). Return dict hasil + riwayat konvergensi."""
    rng = np.random.default_rng(seed)
    cache = {}

    def fit(c):
        key = c.tobytes()
        if key not in cache:
            cache[key] = fitness_fn(c.astype(bool))
        return cache[key]

    pop = (rng.random((pop_size, n_genes)) < 0.5).astype(np.int8)
    pop[pop.sum(1) == 0, rng.integers(n_genes)] = 1
    hist_best, hist_mean = [], []
    for g in range(generations):
        f = np.array([fit(c) for c in pop])
        hist_best.append(float(f.max())); hist_mean.append(float(f.mean()))
        order = np.argsort(-f)
        new = [pop[i].copy() for i in order[:elitism]]
        while len(new) < pop_size:
            def pick():
                cand = rng.integers(pop_size, size=tournament_k)
                return pop[cand[np.argmax(f[cand])]]
            p1, p2 = pick(), pick()
            if rng.random() < crossover_rate:                     # uniform crossover
                m = rng.random(n_genes) < 0.5
                c1, c2 = np.where(m, p1, p2), np.where(m, p2, p1)
            else:
                c1, c2 = p1.copy(), p2.copy()
            for c in (c1, c2):
                flip = rng.random(n_genes) < mutation_rate          # bit-flip mutation
                c[flip] = 1 - c[flip]
                if c.sum() == 0:
                    c[rng.integers(n_genes)] = 1
                new.append(c.astype(np.int8))
        pop = np.array(new[:pop_size])
    f = np.array([fit(c) for c in pop])
    hist_best.append(float(f.max())); hist_mean.append(float(f.mean()))
    b = int(np.argmax(f))
    return {"chromosome": pop[b].tolist(), "fitness": float(f[b]),
            "hist_best": hist_best, "hist_mean": hist_mean, "n_evals": len(cache), "seed": seed}
