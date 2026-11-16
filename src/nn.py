"""MLP klasifikasi multi-kelas dari nol (NumPy): ReLU, dropout, Adam, class-weight, early stopping.
Dibuat sendiri agar tiap anggota dapat menjelaskan forward/backward saat demo, dan ringan untuk deployment."""
import numpy as np


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class MLP:
    def __init__(self, n_in, n_out, hidden=32, n_layers=2, lr=1e-3, dropout=0.0, batch_size=32,
                 epochs=200, patience=15, class_weight=True, l2=1e-5, seed=0):
        self.n_in, self.n_out = n_in, n_out
        self.hidden, self.n_layers = int(hidden), n_layers
        self.lr, self.dropout, self.batch_size = lr, dropout, int(batch_size)
        self.epochs, self.patience, self.class_weight, self.l2, self.seed = epochs, patience, class_weight, l2, seed
        self.rng = np.random.default_rng(seed)
        sizes = [n_in] + [self.hidden] * n_layers + [n_out]
        self.W = [self.rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

    @property
    def n_params(self):
        return int(sum(w.size for w in self.W) + sum(b.size for b in self.b))

    # ---------- forward ----------
    def predict_proba(self, X):
        a = np.asarray(X, float)
        for i in range(len(self.W) - 1):
            a = np.maximum(a @ self.W[i] + self.b[i], 0)
        return _softmax(a @ self.W[-1] + self.b[-1])

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def _loss(self, X, y, w=None):
        p = self.predict_proba(X)
        l = -np.log(p[np.arange(len(y)), y] + 1e-12)
        return float(np.average(l, weights=w))

    # ---------- training ----------
    def fit(self, X, y, Xv=None, yv=None):
        X = np.asarray(X, float)
        n = len(X)
        cw = np.ones(self.n_out)
        if self.class_weight:
            cnt = np.bincount(y, minlength=self.n_out).astype(float)
            cw = n / (self.n_out * np.maximum(cnt, 1))
        sw = cw[y]
        m = [np.zeros_like(p) for p in self.W + self.b]
        v = [np.zeros_like(p) for p in self.W + self.b]
        t = 0
        best, best_w, wait = np.inf, None, 0
        L = len(self.W)
        for ep in range(self.epochs):
            perm = self.rng.permutation(n)
            tot = 0.0
            for s in range(0, n, self.batch_size):
                idx = perm[s:s + self.batch_size]
                xb, yb, wb = X[idx], y[idx], sw[idx]
                acts, masks = [xb], []
                a = xb
                for i in range(L - 1):
                    a = np.maximum(a @ self.W[i] + self.b[i], 0)
                    mk = (self.rng.random(a.shape) >= self.dropout) / (1 - self.dropout) if self.dropout > 0 else np.ones_like(a)
                    a = a * mk
                    masks.append(mk); acts.append(a)
                p = _softmax(a @ self.W[-1] + self.b[-1])
                tot += float(-(np.log(p[np.arange(len(yb)), yb] + 1e-12) * wb).sum())
                d = p.copy()
                d[np.arange(len(yb)), yb] -= 1
                d *= (wb / wb.sum())[:, None]
                gW, gb = [None] * L, [None] * L
                for i in reversed(range(L)):
                    gW[i] = acts[i].T @ d + self.l2 * self.W[i]
                    gb[i] = d.sum(0)
                    if i > 0:
                        d = (d @ self.W[i].T) * masks[i - 1] * (acts[i] > 0)
                t += 1
                params, grads = self.W + self.b, gW + gb
                for k, (p_, g) in enumerate(zip(params, grads)):   # Adam
                    m[k] = 0.9 * m[k] + 0.1 * g
                    v[k] = 0.999 * v[k] + 0.001 * g * g
                    p_ -= self.lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
            self.history["train_loss"].append(tot / sw.sum())
            if Xv is not None:
                vl = self._loss(Xv, yv)
                self.history["val_loss"].append(vl)
                self.history["val_acc"].append(float((self.predict(Xv) == yv).mean()))
                if vl < best - 1e-5:
                    best, wait = vl, 0
                    best_w = ([w.copy() for w in self.W], [b.copy() for b in self.b])
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
        if best_w is not None:
            self.W, self.b = best_w
        return self

    # ---------- simpan / muat ----------
    def save(self, path):
        d = {f"W{i}": w for i, w in enumerate(self.W)}
        d.update({f"b{i}": b for i, b in enumerate(self.b)})
        np.savez(path, **d)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        L = len([k for k in z.files if k.startswith("W")])
        obj = cls.__new__(cls)
        obj.W = [z[f"W{i}"] for i in range(L)]
        obj.b = [z[f"b{i}"] for i in range(L)]
        obj.history = {}
        return obj
