from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate

from . import config as C


@dataclass
class Result:
    name: str
    folds: pd.DataFrame
    pred: np.ndarray
    proba: np.ndarray
    estimators: list = field(repr=False)
    test_idx: list = field(repr=False)
    fit_time: np.ndarray = field(repr=False)


def has_proba(estimator) -> bool:
    return hasattr(estimator, "predict_proba")


def scoring_for(estimator) -> dict:
    if has_proba(estimator):
        return C.SCORING
    return {k: v for k, v in C.SCORING.items() if k != "roc_auc"}


def evaluate(name, estimator, X, y, cv) -> Result:
    scoring = scoring_for(estimator)
    r = cross_validate(
        estimator, X, y, cv=cv, scoring=scoring, return_estimator=True, return_indices=True
    )
    pred = np.empty(len(y), dtype=int)
    proba = np.full(len(y), np.nan)
    for est, idx in zip(r["estimator"], r["indices"]["test"]):
        pred[idx] = est.predict(X.iloc[idx])
        if has_proba(est):
            proba[idx] = est.predict_proba(X.iloc[idx])[:, 1]
    folds = pd.DataFrame({m: r[f"test_{m}"] for m in scoring}).reindex(columns=list(C.SCORING))
    return Result(name, folds, pred, proba, r["estimator"], r["indices"]["test"], r["fit_time"])


def summary(results: dict, families: dict | None = None) -> pd.DataFrame:
    rows = {}
    for name, res in results.items():
        row = {"família": (families or {}).get(name, "")}
        for m in C.SCORING:
            row[f"{m}_media"] = res.folds[m].mean()
            row[f"{m}_dp"] = res.folds[m].std(ddof=1)
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("modelo")


def formatted(table: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    out = pd.DataFrame(index=table.index)
    if "família" in table:
        out["família"] = table["família"]
    for m in C.SCORING:
        mean, sd = table[f"{m}_media"], table[f"{m}_dp"]
        out[m] = [
            "—" if np.isnan(a) else f"{a:.{digits}f} ± {b:.{digits}f}" for a, b in zip(mean, sd)
        ]
    return out


def paired_comparison(a: Result, b: Result, metric: str = C.PRIMARY_METRIC, alpha: float = C.ALPHA) -> dict:
    d = a.folds[metric].to_numpy() - b.folds[metric].to_numpy()
    k = len(d)
    sd = d.std(ddof=1)
    corrected = np.sqrt((1 / k + 1 / (k - 1)) * sd**2)
    t = d.mean() / corrected if corrected > 0 else np.nan
    p = 2 * stats.t.sf(abs(t), k - 1) if corrected > 0 else np.nan
    return {
        "A": a.name,
        "B": b.name,
        "dif_media": d.mean(),
        "dif_dp": sd,
        "dobras_A_melhor": int((d > 0).sum()),
        "dobras_empate": int((d == 0).sum()),
        "t_corrigido": t,
        "p_valor": p,
        "A_supera_B": bool(p < alpha and d.mean() > sd),
        "B_supera_A": bool(p < alpha and -d.mean() > sd),
    }


def comparisons(pairs, results: dict, metric: str = C.PRIMARY_METRIC) -> pd.DataFrame:
    return pd.DataFrame([paired_comparison(results[a], results[b], metric) for a, b in pairs])


def oracle(preds, y) -> np.ndarray:
    return (np.atleast_2d(np.asarray(preds)) == np.asarray(y)).astype(float)


def pairwise_counts(correct: np.ndarray):
    c = np.asarray(correct, dtype=float)
    w = 1 - c
    n11, n00, n10 = c @ c.T, w @ w.T, c @ w.T
    return n11, n00, n10, n10.T


def disagreement_matrix(correct: np.ndarray) -> np.ndarray:
    _, _, n10, n01 = pairwise_counts(correct)
    return (n10 + n01) / correct.shape[1]


def q_matrix(correct: np.ndarray) -> np.ndarray:
    n11, n00, n10, n01 = pairwise_counts(correct)
    num = n11 * n00 - n01 * n10
    den = n11 * n00 + n01 * n10
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def disagreement(pred_a, pred_b, y) -> float:
    return disagreement_matrix(oracle([pred_a, pred_b], y))[0, 1]


def yule_q(pred_a, pred_b, y) -> float:
    return q_matrix(oracle([pred_a, pred_b], y))[0, 1]


def pairwise_diversity(correct: np.ndarray) -> dict:
    iu = np.triu_indices(len(correct), 1)
    return {
        "discordancia": float(disagreement_matrix(correct)[iu].mean()),
        "q_yule": float(np.nanmean(q_matrix(correct)[iu])),
    }


def diversity_frames(preds: dict, y) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = list(preds)
    correct = oracle([preds[n] for n in names], y)
    dis = pd.DataFrame(disagreement_matrix(correct), index=names, columns=names)
    q = pd.DataFrame(q_matrix(correct), index=names, columns=names)
    return dis, q


def unwrap(estimator):
    return estimator.best_estimator_ if hasattr(estimator, "best_estimator_") else estimator


def internal_predictions(fitted, X) -> np.ndarray | None:
    model = unwrap(fitted)
    if hasattr(model, "named_estimators_"):
        return np.array([m.predict(X) for m in model.estimators_])
    prep, ens = model[:-1], model[-1]
    if not hasattr(ens, "estimators_") or np.ndim(ens.estimators_) != 1:
        return None
    Xt = np.asarray(prep.transform(X))
    if hasattr(ens, "estimators_features_"):
        return np.array([e.predict(Xt[:, f]) for e, f in zip(ens.estimators_, ens.estimators_features_)])
    return np.array([e.predict(Xt) for e in ens.estimators_])


def internal_diversity(result: Result, X, y) -> dict | None:
    per_fold = []
    for est, idx in zip(result.estimators, result.test_idx):
        preds = internal_predictions(est, X.iloc[idx])
        if preds is None:
            return None
        per_fold.append(pairwise_diversity(oracle(preds, y.iloc[idx])))
    return pd.DataFrame(per_fold).mean().to_dict()


def per_class(result: Result, y) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, result.pred, labels=[0, 1]).ravel()
    return {
        "VN": tn,
        "FP": fp,
        "FN": fn,
        "VP": tp,
        f"recall_{C.CLASS_NAMES[0]}": tn / (tn + fp),
        f"recall_{C.CLASS_NAMES[1]}": tp / (tp + fn),
        "precisão_doença": tp / (tp + fp) if tp + fp else np.nan,
    }


def timing(estimator, X, y, cv) -> pd.DataFrame:
    rows = []
    for train_idx, val_idx in cv.split(X, y):
        model = clone(estimator)
        t0 = time.perf_counter()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        t1 = time.perf_counter()
        model.predict(X.iloc[val_idx])
        t2 = time.perf_counter()
        rows.append({"treino_s": t1 - t0, "predicao_s": t2 - t1, "n_predito": len(val_idx)})
    return pd.DataFrame(rows)


def scores(y_true, pred, proba=None) -> dict:
    out = {
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "f1_macro": f1_score(y_true, pred, average="macro"),
        "recall_doenca": recall_score(y_true, pred),
        "roc_auc": np.nan,
    }
    if proba is not None:
        out["roc_auc"] = roc_auc_score(y_true, proba)
    return out


def bootstrap(y_true, pred, metric=balanced_accuracy_score, n: int = C.N_BOOTSTRAP, seed: int = C.SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y_true, pred = np.asarray(y_true), np.asarray(pred)
    idx = rng.integers(0, len(y_true), size=(n, len(y_true)))
    return np.array([metric(y_true[i], pred[i]) for i in idx])
