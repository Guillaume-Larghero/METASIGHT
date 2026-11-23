"""Patient-level evaluation metrics with bootstrap 95% confidence intervals.

All metrics are computed on patients, not slides: slide probabilities are averaged
within each patient first (``aggregate_to_patient``).

* AUROC: Model 1 (binary) uses P(metastasis); Model 2 (3-class) is the macro
  one-vs-rest AUROC over the classes present.
* AUPRC: binary only, on P(metastasis).
* Brier: Model 1 = mean (y - p)^2; Model 2 = mean over the classes present of the
  one-vs-rest Brier.
* With cross-validation folds, AUROC and AUPRC are the mean of the per-fold values
  and Brier is pooled over all patients; the bootstrap resamples patients within
  each fold. Without folds (external cohorts) every metric is pooled.
* 95% CI = percentile bootstrap over patients (default 1000 resamples, seed 42).
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

N_BOOT = 1000
BOOT_SEED = 42


def aggregate_to_patient(pred_df, prob_cols, patient_col='patient_id'):
    """
    Average slide-level predictions within each patient.

    Args:
        pred_df: Slide-level predictions with `patient_col`, `true_label`, the
            probability columns and optionally `fold_id` / `cancer_type`
        prob_cols: Probability columns to average
        patient_col: Patient identifier column

    Returns:
        DataFrame with one row per patient and an `n_slides` column
    """
    n_labels = pred_df.groupby(patient_col)['true_label'].nunique()
    if (n_labels > 1).any():
        bad = n_labels[n_labels > 1].index.tolist()[:5]
        raise ValueError(f"Patients with conflicting labels across slides: {bad}")

    agg = {c: 'mean' for c in prob_cols}
    agg['true_label'] = 'first'
    for col in ('fold_id', 'cancer_type'):
        if col in pred_df.columns:
            agg[col] = 'first'
    out = pred_df.groupby(patient_col, sort=False).agg(agg)
    out['n_slides'] = pred_df.groupby(patient_col, sort=False).size()
    return out.reset_index()


def _rank_auc(y_bin, score):
    """Mann-Whitney AUROC with average ranks for ties (equals sklearn roc_auc_score)."""
    y_bin = np.asarray(y_bin, dtype=bool)
    n1 = int(y_bin.sum())
    n0 = len(y_bin) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((rankdata(score)[y_bin].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auroc_score(y, P):
    """Binary AUROC if P has one column, else macro one-vs-rest over classes present."""
    y = np.asarray(y).astype(int)
    P = np.asarray(P, dtype=float).reshape(len(y), -1)
    if P.shape[1] == 1:
        return _rank_auc(y == 1, P[:, 0])
    vals = [_rank_auc(y == c, P[:, c]) for c in range(P.shape[1])
            if 0 < (y == c).sum() < len(y)]
    return float(np.mean(vals)) if vals else np.nan


def auprc_score(y, P):
    """Average precision on P(metastasis); binary only."""
    y = np.asarray(y).astype(int)
    P = np.asarray(P, dtype=float).reshape(len(y), -1)
    if P.shape[1] != 1 or np.unique(y).size < 2:
        return np.nan
    return float(average_precision_score(y, P[:, 0]))


def brier_score(y, P):
    """Binary Brier if P has one column, else macro one-vs-rest Brier over classes present."""
    y = np.asarray(y).astype(int)
    P = np.asarray(P, dtype=float).reshape(len(y), -1)
    if len(y) == 0:
        return np.nan
    if P.shape[1] == 1:
        return float(np.mean((y - P[:, 0]) ** 2))
    parts = [np.mean(((y == c).astype(float) - P[:, c]) ** 2)
             for c in range(P.shape[1]) if (y == c).sum() > 0]
    return float(np.mean(parts)) if parts else np.nan


_METRICS = {'auroc': auroc_score, 'auprc': auprc_score, 'brier': brier_score}
_FOLD_MEAN = {'auroc', 'auprc'}


def _point(y, P, fold_idx, metrics):
    """Metric values on one (possibly resampled) set of patients."""
    out = {}
    for name in metrics:
        fn = _METRICS[name]
        if fold_idx is None or name not in _FOLD_MEAN:
            idx = np.concatenate(fold_idx) if fold_idx is not None else slice(None)
            out[name] = fn(y[idx], P[idx])
        else:
            vals = [fn(y[i], P[i]) for i in fold_idx]
            vals = [v for v in vals if not np.isnan(v)]
            out[name] = float(np.mean(vals)) if vals else np.nan
    return out


def patient_level_metrics(patient_df, prob_cols, metrics=('auroc', 'brier'),
                          n_boot=N_BOOT, seed=BOOT_SEED, use_folds=True):
    """
    Point estimates and percentile-bootstrap 95% CIs on patient-level predictions.

    Args:
        patient_df: Output of `aggregate_to_patient`
        prob_cols: [P(metastasis)] for Model 1, or the 3 class columns for Model 2
        metrics: Any of 'auroc', 'auprc', 'brier'
        n_boot: Bootstrap resamples (0 disables the CI)
        seed: Bootstrap seed
        use_folds: Average AUROC/AUPRC over `fold_id` and resample within folds

    Returns:
        Dict with n_patients, n_slides, n per class, and for each metric
        `<metric>`, `<metric>_ci_lo`, `<metric>_ci_hi`
    """
    y = patient_df['true_label'].astype(int).values
    P = patient_df[list(prob_cols)].values.astype(float)
    keep = np.isfinite(P).all(axis=1)
    y, P = y[keep], P[keep]

    fold_idx = None
    if use_folds and 'fold_id' in patient_df.columns:
        folds = patient_df['fold_id'].values[keep]
        if np.unique(folds).size > 1:
            fold_idx = [np.where(folds == f)[0] for f in np.unique(folds)]

    n_classes = 2 if P.shape[1] == 1 else P.shape[1]
    out = {'n_patients': int(len(y))}
    if 'n_slides' in patient_df.columns:
        out['n_slides'] = int(patient_df['n_slides'].values[keep].sum())
    for c in range(n_classes):
        out[f'n_class_{c}'] = int((y == c).sum())

    point = _point(y, P, fold_idx, metrics)
    boots = {m: [] for m in metrics}
    if n_boot and len(y) > 0:
        rng = np.random.RandomState(seed)
        for _ in range(n_boot):
            if fold_idx is None:
                idx = rng.randint(0, len(y), len(y))
                vals = _point(y[idx], P[idx], None, metrics)
            else:
                draw = [f[rng.randint(0, len(f), len(f))] for f in fold_idx]
                vals = _point(y, P, draw, metrics)
            for m, v in vals.items():
                if not np.isnan(v):
                    boots[m].append(v)

    for m in metrics:
        out[m] = point[m]
        if boots[m] and not np.isnan(point[m]):
            lo, hi = np.percentile(boots[m], [2.5, 97.5])
            out[f'{m}_ci_lo'], out[f'{m}_ci_hi'] = float(lo), float(hi)
        else:
            out[f'{m}_ci_lo'] = out[f'{m}_ci_hi'] = np.nan
    return out


def evaluate_per_cancer(patient_df, prob_cols, metrics=('auroc', 'brier'), min_patients=30,
                        n_boot=N_BOOT, seed=BOOT_SEED):
    """
    Patient-level metrics with 95% CIs for each cancer type.

    Cancer types with fewer than `min_patients` patients or a single outcome class
    are skipped. Folds are ignored within a cancer type (metrics pooled over its
    patients), since a single cancer type can leave folds with one class.

    Returns:
        DataFrame with one row per scored cancer type
    """
    rows = []
    for cancer, g in patient_df.groupby('cancer_type'):
        if len(g) < min_patients or g['true_label'].nunique() < 2:
            continue
        res = patient_level_metrics(g, prob_cols, metrics, n_boot, seed, use_folds=False)
        rows.append({'cancer': cancer, **res})
    return pd.DataFrame(rows)


def format_ci(res, metric, digits=4):
    """'0.8012 (95% CI 0.7810-0.8205)' from a `patient_level_metrics` result."""
    v, lo, hi = res.get(metric), res.get(f'{metric}_ci_lo'), res.get(f'{metric}_ci_hi')
    if v is None or np.isnan(v):
        return 'N/A'
    if lo is None or np.isnan(lo):
        return f"{v:.{digits}f}"
    return f"{v:.{digits}f} (95% CI {lo:.{digits}f}-{hi:.{digits}f})"
