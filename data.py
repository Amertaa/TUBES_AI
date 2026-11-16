"""Loading, label building, split & scaling (tanpa data leakage)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import CLASSES, DATA_CSV, SEED


def make_synthetic(n=1941, seed=0):
    """HANYA untuk smoke test (bentuk mirip dataset asli). Jangan dipakai untuk laporan."""
    from sklearn.datasets import make_classification
    w = np.array([158, 391, 190, 55, 72, 402, 673], float); w /= w.sum()
    X, y = make_classification(n_samples=n, n_features=27, n_informative=10, n_redundant=6,
                               n_classes=7, weights=w, class_sep=1.0, random_state=seed)
    X = pd.DataFrame(X, columns=[f"F{i+1}" for i in range(27)])
    Y = pd.DataFrame(np.eye(7, dtype=int)[y], columns=CLASSES)
    return pd.concat([X, Y], axis=1)


def download_uci():
    """Unduh UCI ID 198 lalu simpan ke data/steel_plates.csv (hanya sekali)."""
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=198)
    X, Y = ds.data.features.copy(), ds.data.targets.copy()
    print("Kolom target dari UCI:", list(Y.columns))
    if Y.shape[1] != len(CLASSES):
        raise ValueError(f"Harapannya 7 kolom target, ditemukan {Y.shape[1]}: {list(Y.columns)}")
    Y.columns = CLASSES  # samakan nama/urutan kelas dengan config
    df = pd.concat([X, Y], axis=1)
    DATA_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(DATA_CSV, index=False)
    return df


def load_dataset(synthetic=False):
    """Return (X DataFrame, y int array). Label 7 kolom biner -> 1 kelas (argmax)."""
    if synthetic:
        df = make_synthetic()
        DATA_CSV.parent.mkdir(exist_ok=True)
        df.to_csv(DATA_CSV, index=False)
    elif DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV)
    else:
        df = download_uci()
    Y = df[CLASSES].values
    keep = Y.sum(1) == 1  # buang baris tanpa label / multi-label bila ada
    df = df[keep].reset_index(drop=True)
    X = df.drop(columns=CLASSES)
    y = df[CLASSES].values.argmax(1)
    return X, y


def three_way_split(X, y, seed=SEED):
    """Train-final (80%) / Test (20%) stratified; train dipecah lagi menjadi
    sub-train (75%) / validation (25%) -> dipakai oleh GA & PSO. Test tidak pernah disentuh optimasi."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    X_sub, X_val, y_sub, y_val = train_test_split(X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=seed)
    return dict(X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te,
                X_sub=X_sub, y_sub=y_sub, X_val=X_val, y_val=y_val)


class Scaler:
    """StandardScaler minimal (fit hanya pada data training)."""
    def fit(self, X):
        X = np.asarray(X, float)
        self.mean_ = X.mean(0)
        self.scale_ = X.std(0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        return (np.asarray(X, float) - self.mean_) / self.scale_