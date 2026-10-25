"""AUC の差の検定: fold 別 DeLong 検定 + Holm 補正。

同じ行に対する複数モデルの AUC を比べる標準的な方法は DeLong 検定（DeLong et al., 1988）で、
複数の比較には Holm 補正をかける。ここでの指標は「fold 別 AUC の平均」なので、DeLong の
分散を fold ごとに出し、平均の差の分散にまとめる。時系列CV の各 fold のテスト期間は互いに
重ならないため、fold 間は独立とみなして分散を足し合わせられる。

全 fold をまとめて 1 本の ROC にしないのは、学習データが増えるほど確率が低く出る
（fold ごとの確率の水準がずれる）ため、まとめた AUC は fold の違いを混ぜてしまうから。
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _structural_components(scores, y):
    """DeLong の構造成分。scores は (モデル数, 件数)。

    V10[r, i]: 不良 i の得点が良品の得点を上回る割合（同点は 0.5）
    V01[r, j]: 良品 j の得点を不良の得点が上回る割合
    """
    pos, neg = scores[:, y == 1], scores[:, y == 0]
    cmp = (pos[:, :, None] > neg[:, None, :]) + 0.5 * (pos[:, :, None] == neg[:, None, :])
    return cmp.mean(axis=2), cmp.mean(axis=1)


def fold_delong(scores, y):
    """1 fold 分の AUC（モデルごと）と、その共分散行列。"""
    v10, v01 = _structural_components(scores, y)
    auc = v10.mean(axis=1)
    cov = np.atleast_2d(np.cov(v10)) / v10.shape[1] + np.atleast_2d(np.cov(v01)) / v01.shape[1]
    return auc, cov


def compare_to_baseline(proba_by_name, baseline, y, fold):
    """各モデルと基準の fold 別 AUC 平均の差を、fold 別 DeLong で検定する。

    Returns: {name: dict(diff, se, ci_low, ci_high, p)}（基準自身は含まない）
    """
    names = list(proba_by_name)
    b = names.index(baseline)
    folds = np.unique(fold)
    diff_sum = np.zeros(len(names))
    var_sum = np.zeros(len(names))
    for k in folds:
        m = fold == k
        auc, cov = fold_delong(np.vstack([proba_by_name[n][m] for n in names]), y[m])
        diff_sum += auc - auc[b]
        var_sum += np.diag(cov) + cov[b, b] - 2 * cov[:, b]
    out = {}
    for i, name in enumerate(names):
        if i == b:
            continue
        diff = diff_sum[i] / len(folds)
        se = np.sqrt(var_sum[i]) / len(folds)
        out[name] = {
            "diff": diff, "se": se,
            "ci_low": diff - norm.ppf(0.975) * se, "ci_high": diff + norm.ppf(0.975) * se,
            "p": float(2 * norm.sf(abs(diff) / se)),
        }
    return out


def holm(pvalues):
    """Holm 補正後の p 値（入力と同じ順序）。"""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj
