"""
utils.py
==========
Shared core for AIShield: dataset loading/synthesis, preprocessing, the five
individual models, and the hybrid fusion layer that combines them.

Both train_model.py (which BUILDS the models) and predict.py (which LOADS
and USES them) import from this file, so there is exactly one definition of
every class -- critical because joblib needs the same class definitions
available at load time as were used at save time.

------------------------------------------------------------------------
HYBRID DESIGN
------------------------------------------------------------------------
  1. SUPERVISED LAYER   : Random Forest + XGBoost + SVM + Neural Network,
                           soft-voted (weighted average of predict_proba)
                           -> supervised_score in [0, 1]
  2. UNSUPERVISED LAYER : Isolation Forest, trained ONLY on normal traffic
                           -> anomaly_score in [0, 1] (1 = very anomalous)
  3. FUSION             : final_score = ALPHA * supervised_score
                                        + (1 - ALPHA) * anomaly_score
                           verdict = ATTACK if final_score >= THRESHOLD
------------------------------------------------------------------------
"""
import os
import glob
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    _HAS_XGB = False


# =========================================================================
# CONFIG  (tune everything here -- no separate config file needed)
# =========================================================================
CONFIG = {
    "data": {
        "data_dir": "dataset",
        "preferred_dataset": "nsl_kdd",   # nsl_kdd | cic_ids | unsw_nb15 | synthetic
        "test_size": 0.25,
        "random_state": 42,
    },
    "feature_engineering": {
        "select_k_best": True,
        "k_features": 30,
        "scale_method": "standard",       # standard | minmax
    },
    "models": {
        "random_forest": dict(n_estimators=200, max_depth=20, n_jobs=-1,
                               class_weight="balanced", random_state=42),
        "xgboost": dict(n_estimators=300, max_depth=8, learning_rate=0.1,
                         subsample=0.9, random_state=42),
        "svm": dict(kernel="rbf", C=2.0, gamma="scale",
                    max_train_samples=15000, random_state=42),
        "neural_network": dict(hidden_layer_sizes=(128, 64, 32),
                                max_iter=300, alpha=0.0001, random_state=42),
        "isolation_forest": dict(n_estimators=200, contamination=0.1, random_state=42),
    },
    "hybrid_fusion": {
        "weights": {"random_forest": 1.0, "xgboost": 1.2, "svm": 0.8, "neural_network": 1.0},
        "alpha_supervised_vs_anomaly": 0.75,
        "decision_threshold": 0.5,
    },
    "paths": {
        "models_dir": "models",
    },
}

# =========================================================================
# FEATURE SCHEMA (NSL-KDD-style, 41 features) -- also used to render the web form
# =========================================================================
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

CATEGORICAL_OPTIONS = {
    "protocol_type": ["tcp", "udp", "icmp"],
    "service": ["http", "ftp", "ftp_data", "smtp", "telnet", "ssh", "dns", "private", "pop_3", "other"],
    "flag": ["SF", "S0", "REJ", "RSTR", "RSTO", "SH"],
}

FEATURE_GROUPS = {
    "Basic Connection Features": [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    ],
    "Content Features": [
        "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
        "num_compromised", "root_shell", "su_attempted", "num_root",
        "num_file_creations", "num_shells", "num_access_files",
        "num_outbound_cmds", "is_host_login", "is_guest_login",
    ],
    "Time-based Traffic Features": [
        "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
        "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    ],
    "Host-based Traffic Features": [
        "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    ],
}

ATTACK_CATEGORY_MAP = {
    "neptune": "dos", "smurf": "dos", "back": "dos", "teardrop": "dos", "land": "dos",
    "pod": "dos", "apache2": "dos", "udpstorm": "dos", "processtable": "dos", "mailbomb": "dos",
    "satan": "probe", "ipsweep": "probe", "nmap": "probe", "portsweep": "probe",
    "mscan": "probe", "saint": "probe",
    "guess_passwd": "r2l", "ftp_write": "r2l", "warezclient": "r2l", "warezmaster": "r2l",
    "imap": "r2l", "multihop": "r2l", "phf": "r2l", "spy": "r2l", "named": "r2l",
    "sendmail": "r2l", "snmpgetattack": "r2l", "snmpguess": "r2l", "xlock": "r2l",
    "xsnoop": "r2l", "worm": "r2l",
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r", "rootkit": "u2r",
    "httptunnel": "u2r", "ps": "u2r", "sqlattack": "u2r", "xterm": "u2r",
    "normal": "normal",
}


def map_category(label: str) -> str:
    label = str(label).strip().lower().rstrip(".")
    return ATTACK_CATEGORY_MAP.get(label, "attack_other" if label != "normal" else "normal")


# =========================================================================
# SYNTHETIC DATA GENERATOR (used when no real dataset is present)
# =========================================================================
ATTACK_LABELS = {
    "dos": ["neptune", "smurf", "back", "teardrop", "land"],
    "probe": ["satan", "ipsweep", "nmap", "portsweep"],
    "r2l": ["guess_passwd", "ftp_write", "warezclient", "imap"],
    "u2r": ["buffer_overflow", "rootkit", "loadmodule", "perl"],
}


def _rand_choice(rng, options, n):
    return rng.choice(options, size=n)


def _make_block(rng, n, category):
    df = pd.DataFrame(index=range(n), columns=NSL_KDD_COLUMNS)

    if category == "normal":
        df["duration"] = rng.exponential(2.0, n)
        df["protocol_type"] = _rand_choice(rng, CATEGORICAL_OPTIONS["protocol_type"], n)
        df["service"] = _rand_choice(rng, CATEGORICAL_OPTIONS["service"], n)
        df["flag"] = "SF"
        df["src_bytes"] = rng.lognormal(5, 1.5, n)
        df["dst_bytes"] = rng.lognormal(5, 1.5, n)
        df["logged_in"] = rng.choice([0, 1], n, p=[0.2, 0.8])
        df["count"] = rng.integers(1, 20, n)
        df["srv_count"] = rng.integers(1, 20, n)
        df["serror_rate"] = rng.uniform(0, 0.05, n)
        df["srv_serror_rate"] = rng.uniform(0, 0.05, n)
        df["rerror_rate"] = rng.uniform(0, 0.05, n)
        df["srv_rerror_rate"] = rng.uniform(0, 0.05, n)
        df["same_srv_rate"] = rng.uniform(0.8, 1.0, n)
        df["diff_srv_rate"] = rng.uniform(0, 0.1, n)
        df["dst_host_count"] = rng.integers(1, 255, n)
        df["dst_host_srv_count"] = rng.integers(1, 255, n)
        label = "normal"

    elif category == "dos":
        df["duration"] = rng.exponential(0.1, n)
        df["protocol_type"] = _rand_choice(rng, ["tcp", "icmp"], n)
        df["service"] = _rand_choice(rng, ["private", "http"], n)
        df["flag"] = _rand_choice(rng, ["S0", "REJ"], n)
        df["src_bytes"] = rng.uniform(0, 50, n)
        df["dst_bytes"] = 0
        df["logged_in"] = 0
        df["count"] = rng.integers(200, 511, n)
        df["srv_count"] = rng.integers(200, 511, n)
        df["serror_rate"] = rng.uniform(0.8, 1.0, n)
        df["srv_serror_rate"] = rng.uniform(0.8, 1.0, n)
        df["rerror_rate"] = rng.uniform(0, 0.2, n)
        df["srv_rerror_rate"] = rng.uniform(0, 0.2, n)
        df["same_srv_rate"] = rng.uniform(0.9, 1.0, n)
        df["diff_srv_rate"] = rng.uniform(0, 0.05, n)
        df["dst_host_count"] = rng.integers(200, 255, n)
        df["dst_host_srv_count"] = rng.integers(200, 255, n)
        label = rng.choice(ATTACK_LABELS["dos"], n)

    elif category == "probe":
        df["duration"] = rng.exponential(0.05, n)
        df["protocol_type"] = _rand_choice(rng, CATEGORICAL_OPTIONS["protocol_type"], n)
        df["service"] = _rand_choice(rng, CATEGORICAL_OPTIONS["service"], n)
        df["flag"] = _rand_choice(rng, ["S0", "REJ", "RSTR"], n)
        df["src_bytes"] = rng.uniform(0, 20, n)
        df["dst_bytes"] = rng.uniform(0, 20, n)
        df["logged_in"] = 0
        df["count"] = rng.integers(1, 50, n)
        df["srv_count"] = rng.integers(1, 10, n)
        df["serror_rate"] = rng.uniform(0.2, 0.6, n)
        df["srv_serror_rate"] = rng.uniform(0.2, 0.6, n)
        df["rerror_rate"] = rng.uniform(0.2, 0.6, n)
        df["srv_rerror_rate"] = rng.uniform(0.2, 0.6, n)
        df["same_srv_rate"] = rng.uniform(0, 0.3, n)
        df["diff_srv_rate"] = rng.uniform(0.6, 1.0, n)
        df["dst_host_count"] = rng.integers(1, 255, n)
        df["dst_host_srv_count"] = rng.integers(1, 30, n)
        label = rng.choice(ATTACK_LABELS["probe"], n)

    elif category == "r2l":
        df["duration"] = rng.exponential(3.0, n)
        df["protocol_type"] = "tcp"
        df["service"] = _rand_choice(rng, ["ftp", "telnet", "smtp", "pop_3"], n)
        df["flag"] = _rand_choice(rng, ["SF", "RSTO"], n)
        df["src_bytes"] = rng.lognormal(4, 1.0, n)
        df["dst_bytes"] = rng.lognormal(3, 1.0, n)
        df["num_failed_logins"] = rng.integers(1, 5, n)
        df["logged_in"] = rng.choice([0, 1], n, p=[0.7, 0.3])
        df["is_guest_login"] = rng.choice([0, 1], n, p=[0.6, 0.4])
        df["count"] = rng.integers(1, 10, n)
        df["srv_count"] = rng.integers(1, 10, n)
        df["same_srv_rate"] = rng.uniform(0.5, 1.0, n)
        df["dst_host_count"] = rng.integers(1, 50, n)
        df["dst_host_srv_count"] = rng.integers(1, 50, n)
        label = rng.choice(ATTACK_LABELS["r2l"], n)

    else:  # u2r
        df["duration"] = rng.exponential(10.0, n)
        df["protocol_type"] = "tcp"
        df["service"] = _rand_choice(rng, ["telnet", "ftp", "other"], n)
        df["flag"] = "SF"
        df["src_bytes"] = rng.lognormal(6, 1.5, n)
        df["dst_bytes"] = rng.lognormal(5, 1.5, n)
        df["logged_in"] = 1
        df["root_shell"] = rng.choice([0, 1], n, p=[0.3, 0.7])
        df["su_attempted"] = rng.choice([0, 1, 2], n, p=[0.4, 0.4, 0.2])
        df["num_root"] = rng.integers(1, 10, n)
        df["num_file_creations"] = rng.integers(0, 5, n)
        df["num_shells"] = rng.integers(0, 3, n)
        df["count"] = rng.integers(1, 5, n)
        df["srv_count"] = rng.integers(1, 5, n)
        df["dst_host_count"] = rng.integers(1, 20, n)
        df["dst_host_srv_count"] = rng.integers(1, 20, n)
        label = rng.choice(ATTACK_LABELS["u2r"], n)

    for col in NSL_KDD_COLUMNS:
        if df[col].isna().all():
            df[col] = 0
    df = df.fillna(0)

    for rate_col in [
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    ]:
        if (df[rate_col] == 0).all():
            df[rate_col] = rng.uniform(0, 1, n)

    df["label"] = label
    return df


def generate_synthetic(n_samples=40000, attack_ratio=0.35, random_state=42):
    """Builds a synthetic, NSL-KDD-shaped dataset with realistic DoS/Probe/R2L/U2R clusters."""
    rng = np.random.default_rng(random_state)
    n_attack = int(n_samples * attack_ratio)
    n_normal = n_samples - n_attack

    cat_weights = {"dos": 0.55, "probe": 0.25, "r2l": 0.15, "u2r": 0.05}
    blocks = [_make_block(rng, n_normal, "normal")]
    for cat, w in cat_weights.items():
        n_cat = max(1, int(n_attack * w))
        blocks.append(_make_block(rng, n_cat, cat))

    df = pd.concat(blocks, ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    non_numeric = {"protocol_type", "service", "flag", "label"}
    for col in NSL_KDD_COLUMNS:
        if col not in non_numeric:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def sample_record(kind="normal", random_state=None):
    """Returns a single realistic sample row (as a dict) -- used by the 'Load Sample' buttons."""
    rng = np.random.default_rng(random_state)
    df = _make_block(rng, 1, kind)
    row = df.iloc[0].drop("label").to_dict()
    return row


# =========================================================================
# DATASET LOADING (NSL-KDD / CIC-IDS / UNSW-NB15 / synthetic fallback)
# =========================================================================
def _load_nsl_kdd(data_dir):
    train_path = os.path.join(data_dir, "KDDTrain+.txt")
    test_path = os.path.join(data_dir, "KDDTest+.txt")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        return None
    cols = NSL_KDD_COLUMNS + ["label", "difficulty"]
    df = pd.concat([pd.read_csv(train_path, names=cols), pd.read_csv(test_path, names=cols)],
                    ignore_index=True)
    df = df.drop(columns=["difficulty"], errors="ignore")
    y_category = df["label"].apply(map_category)
    y_binary = (y_category != "normal").astype(int)
    return df.drop(columns=["label"]), y_binary, y_category


def _load_synthetic(data_dir):
    train_path = os.path.join(data_dir, "synthetic_train.csv")
    test_path = os.path.join(data_dir, "synthetic_test.csv")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        return None
    df = pd.concat([pd.read_csv(train_path), pd.read_csv(test_path)], ignore_index=True)
    y_category = df["label"].apply(map_category)
    y_binary = (y_category != "normal").astype(int)
    return df.drop(columns=["label"]), y_binary, y_category


def _load_unsw_nb15(data_dir):
    train_path = os.path.join(data_dir, "UNSW_NB15_training-set.csv")
    test_path = os.path.join(data_dir, "UNSW_NB15_testing-set.csv")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        return None
    df = pd.concat([pd.read_csv(train_path), pd.read_csv(test_path)], ignore_index=True)
    y_binary = df["label"].astype(int)
    y_category = df.get("attack_cat", pd.Series(["normal"] * len(df)))
    y_category = y_category.fillna("normal").apply(lambda v: str(v).strip().lower() or "normal")
    X = df.drop(columns=[c for c in ["label", "attack_cat", "id"] if c in df.columns])
    return X, y_binary, y_category


def _load_cic_ids(data_dir):
    csv_files = [f for f in glob.glob(os.path.join(data_dir, "*.csv"))
                 if "synthetic" not in os.path.basename(f).lower()]
    matching = []
    for f in csv_files:
        try:
            head = pd.read_csv(f, nrows=1)
        except Exception:
            continue
        if "label" in [c.strip().lower() for c in head.columns]:
            matching.append(f)
    if not matching:
        return None
    df = pd.concat([pd.read_csv(f) for f in matching], ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    label_col = [c for c in df.columns if c.lower() == "label"][0]
    y_category = df[label_col].apply(
        lambda v: "normal" if str(v).strip().lower() in ("benign", "normal") else str(v).strip().lower()
    )
    y_binary = (y_category != "normal").astype(int)
    X = df.drop(columns=[label_col]).select_dtypes(include=["number", "object"])
    return X, y_binary, y_category


_LOADERS = {"nsl_kdd": _load_nsl_kdd, "unsw_nb15": _load_unsw_nb15,
            "cic_ids": _load_cic_ids, "synthetic": _load_synthetic}


def load_dataset(data_dir="dataset", preferred_dataset="nsl_kdd"):
    """Returns (X, y_binary, y_category, source_name). Falls back to synthetic if nothing found."""
    order = [preferred_dataset] + [k for k in _LOADERS if k != preferred_dataset]
    for name in order:
        result = _LOADERS[name](data_dir)
        if result is not None:
            X, y_binary, y_category = result
            print(f"[utils] Loaded '{name}' dataset: {len(X)} rows "
                  f"({int(y_binary.sum())} attack / {int((y_binary == 0).sum())} normal).")
            return X, y_binary, y_category, name

    print(f"[utils] No dataset found in '{data_dir}'. Generating synthetic data...")
    df = generate_synthetic()
    os.makedirs(data_dir, exist_ok=True)
    split = int(len(df) * 0.75)
    df.iloc[:split].to_csv(os.path.join(data_dir, "synthetic_train.csv"), index=False)
    df.iloc[split:].to_csv(os.path.join(data_dir, "synthetic_test.csv"), index=False)
    X, y_binary, y_category = _load_synthetic(data_dir)
    return X, y_binary, y_category, "synthetic"


def split_dataset(X, y_binary, y_category, test_size=0.25, random_state=42):
    return train_test_split(X, y_binary, y_category, test_size=test_size,
                             random_state=random_state, stratify=y_binary)


# =========================================================================
# PREPROCESSING
# =========================================================================
class Preprocessor:
    """Cleans, encodes and scales raw traffic features. Fit once on training
    data; the exact same fitted transform is reused at prediction time."""

    def __init__(self, scale_method="standard"):
        self.scale_method = scale_method
        self.label_encoders = {}
        self.scaler = None
        self.categorical_cols = []
        self.numeric_cols = []
        self.fitted_columns = None

    def _clean(self, X: pd.DataFrame, is_fit: bool) -> pd.DataFrame:
        X = X.copy()
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in X.columns:
            is_categorical = (X[col].dtype == object) if is_fit else (col in self.categorical_cols)
            if is_categorical:
                X[col] = X[col].fillna("unknown").astype(str).str.strip()
            else:
                X[col] = pd.to_numeric(X[col], errors="coerce")
                X[col] = X[col].fillna(X[col].median() if X[col].notna().any() else 0)
        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._clean(X, is_fit=True)
        self.categorical_cols = [c for c in X.columns if X[c].dtype == object]
        self.numeric_cols = [c for c in X.columns if c not in self.categorical_cols]

        for col in self.categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le

        Scaler = StandardScaler if self.scale_method == "standard" else MinMaxScaler
        self.scaler = Scaler()
        if self.numeric_cols:
            X[self.numeric_cols] = self.scaler.fit_transform(X[self.numeric_cols])

        self.fitted_columns = list(X.columns)
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._clean(X, is_fit=False)
        for col in self.fitted_columns:
            if col not in X.columns:
                X[col] = "unknown" if col in self.categorical_cols else 0
        X = X[self.fitted_columns]

        for col in self.categorical_cols:
            le = self.label_encoders[col]
            known = set(le.classes_)
            X[col] = X[col].apply(lambda v: v if v in known else le.classes_[0])
            X[col] = le.transform(X[col])

        if self.numeric_cols:
            X[self.numeric_cols] = self.scaler.transform(X[self.numeric_cols])
        return X


class FeatureSelector:
    def __init__(self, k=30, enabled=True):
        self.k = k
        self.enabled = enabled
        self.selector = None
        self.selected_columns = None

    def fit_transform(self, X: pd.DataFrame, y) -> pd.DataFrame:
        if not self.enabled or self.k >= X.shape[1]:
            self.selected_columns = list(X.columns)
            return X
        self.selector = SelectKBest(score_func=mutual_info_classif, k=self.k)
        arr = self.selector.fit_transform(X, y)
        mask = self.selector.get_support()
        self.selected_columns = list(X.columns[mask])
        return pd.DataFrame(arr, columns=self.selected_columns, index=X.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled or self.selector is None:
            return X[self.selected_columns] if self.selected_columns else X
        return X[self.selected_columns]


# =========================================================================
# INDIVIDUAL MODEL WRAPPERS
# =========================================================================
class RandomForestModel:
    name = "random_forest"

    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X):
        return self.model.predict(X)


class XGBoostModel:
    name = "xgboost"

    def __init__(self, n_estimators=300, max_depth=8, learning_rate=0.1,
                 subsample=0.9, random_state=42, **kwargs):
        if _HAS_XGB:
            self.model = XGBClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate, subsample=subsample,
                random_state=random_state, eval_metric="logloss", n_jobs=-1,
            )
        else:
            print("[utils] 'xgboost' not installed -- falling back to "
                  "GradientBoostingClassifier. Run `pip install xgboost` for the real thing.")
            self.model = GradientBoostingClassifier(
                n_estimators=min(n_estimators, 150), max_depth=min(max_depth, 5),
                learning_rate=learning_rate, subsample=subsample, random_state=random_state,
            )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X):
        return self.model.predict(X)


class SVMModel:
    """Subsamples large training sets (stratified) since SVMs don't scale to 100k+ rows."""
    name = "svm"

    def __init__(self, kernel="rbf", C=2.0, gamma="scale",
                 max_train_samples=15000, random_state=42, **kwargs):
        self.model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True,
                          random_state=random_state, class_weight="balanced")
        self.max_train_samples = max_train_samples
        self.random_state = random_state

    def _subsample(self, X, y):
        if len(X) <= self.max_train_samples:
            return X, y
        rng = np.random.default_rng(self.random_state)
        y_arr = y.values if hasattr(y, "values") else np.asarray(y)
        idx_0, idx_1 = np.where(y_arr == 0)[0], np.where(y_arr == 1)[0]
        n_each = self.max_train_samples // 2
        take_0 = rng.choice(idx_0, size=min(n_each, len(idx_0)), replace=False)
        take_1 = rng.choice(idx_1, size=min(n_each, len(idx_1)), replace=False)
        idx = np.concatenate([take_0, take_1])
        rng.shuffle(idx)
        if hasattr(X, "iloc"):
            return X.iloc[idx], y.iloc[idx] if hasattr(y, "iloc") else y[idx]
        return X[idx], y[idx]

    def fit(self, X, y):
        X_sub, y_sub = self._subsample(X, y)
        self.model.fit(X_sub, y_sub)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X):
        return self.model.predict(X)


class NeuralNetworkModel:
    name = "neural_network"

    def __init__(self, hidden_layer_sizes=(128, 64, 32), max_iter=300,
                 alpha=0.0001, random_state=42, **kwargs):
        self.model = MLPClassifier(hidden_layer_sizes=tuple(hidden_layer_sizes),
                                    max_iter=max_iter, alpha=alpha,
                                    random_state=random_state, early_stopping=True)

    def fit(self, X, y):
        X_arr = X.values if hasattr(X, "values") else X
        y_arr = y.values if hasattr(y, "values") else y
        self.model.fit(X_arr, y_arr)
        return self

    def predict_proba(self, X):
        X_arr = X.values if hasattr(X, "values") else X
        return self.model.predict_proba(X_arr)[:, 1]

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class IsolationForestModel:
    """Trained ONLY on normal traffic -- flags statistically weird traffic even
    if no supervised model has ever seen that specific attack pattern."""
    name = "isolation_forest"

    def __init__(self, n_estimators=200, contamination=0.1, random_state=42, **kwargs):
        self.model = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                                      random_state=random_state, n_jobs=-1)
        self._score_min = None
        self._score_max = None

    def fit(self, X_normal_only):
        self.model.fit(X_normal_only)
        raw = self.model.score_samples(X_normal_only)
        self._score_min, self._score_max = raw.min(), raw.max()
        return self

    def anomaly_score(self, X):
        raw = self.model.score_samples(X)
        span = max(self._score_max - self._score_min, 1e-9)
        normal_likeness = np.clip((raw - self._score_min) / span, 0, 1)
        return 1.0 - normal_likeness

    def predict(self, X):
        return (self.model.predict(X) == -1).astype(int)


# =========================================================================
# HYBRID ENSEMBLE -- the fusion layer
# =========================================================================
class HybridEnsemble:
    def __init__(self, config: dict = CONFIG):
        m_cfg = config["models"]
        self.weights = config["hybrid_fusion"]["weights"]
        self.alpha = config["hybrid_fusion"]["alpha_supervised_vs_anomaly"]
        self.threshold = config["hybrid_fusion"]["decision_threshold"]

        self.supervised_models = {
            "random_forest": RandomForestModel(**m_cfg["random_forest"]),
            "xgboost": XGBoostModel(**m_cfg["xgboost"]),
            "svm": SVMModel(**m_cfg["svm"]),
            "neural_network": NeuralNetworkModel(**m_cfg["neural_network"]),
        }
        self.anomaly_model = IsolationForestModel(**m_cfg["isolation_forest"])

    def fit(self, X_train, y_train, verbose=True):
        for name, model in self.supervised_models.items():
            if verbose:
                print(f"[HybridEnsemble] Training {name}...")
            model.fit(X_train, y_train)

        X_train_ri = X_train.reset_index(drop=True) if hasattr(X_train, "reset_index") else X_train
        y_train_arr = y_train.values if hasattr(y_train, "values") else np.asarray(y_train)
        normal_mask = (y_train_arr == 0)
        X_normal = X_train_ri[normal_mask]
        if verbose:
            print(f"[HybridEnsemble] Training isolation_forest on {len(X_normal)} normal-only samples...")
        self.anomaly_model.fit(X_normal)
        return self

    def _supervised_scores(self, X):
        scores, weighted_sum, total_weight = {}, np.zeros(len(X)), 0.0
        for name, model in self.supervised_models.items():
            proba = model.predict_proba(X)
            scores[name] = proba
            w = self.weights.get(name, 1.0)
            weighted_sum += w * proba
            total_weight += w
        return scores, weighted_sum / max(total_weight, 1e-9)

    def score(self, X):
        individual, supervised_score = self._supervised_scores(X)
        anomaly_score = self.anomaly_model.anomaly_score(X)
        final_score = self.alpha * supervised_score + (1 - self.alpha) * anomaly_score
        verdict = (final_score >= self.threshold).astype(int)
        return {
            "individual_scores": individual,
            "supervised_score": supervised_score,
            "anomaly_score": anomaly_score,
            "final_score": final_score,
            "verdict": verdict,
        }

    def predict(self, X):
        return self.score(X)["verdict"]

    def predict_proba(self, X):
        return self.score(X)["final_score"]

    def save(self, models_dir):
        os.makedirs(models_dir, exist_ok=True)
        for name, model in self.supervised_models.items():
            joblib.dump(model, os.path.join(models_dir, f"{name}.joblib"))
        joblib.dump(self.anomaly_model, os.path.join(models_dir, "isolation_forest.joblib"))

    def load(self, models_dir):
        for name in self.supervised_models:
            self.supervised_models[name] = joblib.load(os.path.join(models_dir, f"{name}.joblib"))
        self.anomaly_model = joblib.load(os.path.join(models_dir, "isolation_forest.joblib"))
        return self


# =========================================================================
# ARTIFACT SAVE / LOAD HELPERS
# =========================================================================
def save_pipeline(models_dir, ensemble: HybridEnsemble, preprocessor: Preprocessor,
                   selector: FeatureSelector, extra: dict = None):
    os.makedirs(models_dir, exist_ok=True)
    ensemble.save(models_dir)
    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor.joblib"))
    joblib.dump(selector, os.path.join(models_dir, "feature_selector.joblib"))
    if extra:
        joblib.dump(extra, os.path.join(models_dir, "meta.joblib"))


def load_pipeline(models_dir="models", config: dict = CONFIG):
    required = ["preprocessor.joblib", "feature_selector.joblib",
                "random_forest.joblib", "xgboost.joblib", "svm.joblib",
                "neural_network.joblib", "isolation_forest.joblib"]
    missing = [f for f in required if not os.path.exists(os.path.join(models_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"Model artifacts missing from '{models_dir}/': {missing}. "
            "Run `python train_model.py` first."
        )
    ensemble = HybridEnsemble(config).load(models_dir)
    preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.joblib"))
    selector = joblib.load(os.path.join(models_dir, "feature_selector.joblib"))
    return ensemble, preprocessor, selector


def predict_dataframe(df_raw: pd.DataFrame, ensemble, preprocessor, selector) -> pd.DataFrame:
    """Runs the full raw-features -> verdict pipeline on a DataFrame of one or more records."""
    X_t = preprocessor.transform(df_raw)
    X_sel = selector.transform(X_t)
    result = ensemble.score(X_sel)

    out = pd.DataFrame({
        "final_score": result["final_score"],
        "verdict": ["ATTACK" if v == 1 else "NORMAL" for v in result["verdict"]],
        "supervised_score": result["supervised_score"],
        "anomaly_score": result["anomaly_score"],
    })
    for name, proba in result["individual_scores"].items():
        out[f"{name}_score"] = proba
        out[f"{name}_vote"] = ["ATTACK" if p >= 0.5 else "normal" for p in proba]
    return out
