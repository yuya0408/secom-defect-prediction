"""見出しの数字の確かめ: 平均 AUC の区間と、固定しきい値での検査割合・再現率。

- 平均 AUC の 95% 区間と 0.5 に対する p は、fold ごとの DeLong の分散を足して求める。
  fold のテスト期間は重ならないので独立とみなすが、学習データは入れ子に重なり、学習のばらつきは
  数えていない（区間は狭めに出る）。
- README の再現率 66.2% は、各 fold の中で不良確率に順位を付けて上位 50% を選んだ値である。
  API が使う固定の既定しきい値（metadata の ``default_threshold``）を各 fold にそのまま当てると、
  確率の水準が fold ごとに違うため、検査に回る割合は 50% に保たれない。その値を並べる。

使い方（リポジトリのルートで）::

    python -m scripts.check_headline

出力:
    reports/headline_checks.md
"""

from __future__ import annotations

import math
import os

import joblib
import numpy as np
from scipy.stats import norm

from scripts.auc_tests import fold_delong
from src.train import MODELS_DIR, load_data, time_series_oof

OUT = os.path.join("reports", "headline_checks.md")


def mean_auc_interval(proba, y, fold):
    aucs, variances = [], []
    for k in np.unique(fold):
        m = fold == k
        auc, cov = fold_delong(proba[m][None, :], y[m])
        aucs.append(float(auc[0]))
        variances.append(float(cov[0, 0]))
    mean = float(np.mean(aucs))
    se = math.sqrt(sum(variances)) / len(aucs)
    z = (mean - 0.5) / se
    return mean, se, (mean - 1.96 * se, mean + 1.96 * se), 2 * norm.sf(abs(z))


def fixed_threshold_rows(proba, y, fold, threshold):
    rows = []
    for k in np.unique(fold):
        m = fold == k
        flagged = proba[m] >= threshold
        rows.append((f"fold {k + 1}", flagged.mean(), int((flagged & (y[m] == 1)).sum()), int(y[m].sum())))
    flagged = proba >= threshold
    rows.append(("全体", flagged.mean(), int((flagged & (y == 1)).sum()), int(y.sum())))
    return rows


def main():
    proba, y, fold = time_series_oof(*load_data())
    threshold = joblib.load(os.path.join(MODELS_DIR, "metadata.pkl"))["default_threshold"]
    mean, se, (lo, hi), p = mean_auc_interval(proba, y, fold)
    lines = [
        "# 見出しの数字の確かめ（厳密な時系列CV）",
        "",
        "`python -m scripts.check_headline` で再生成。前提はスクリプト冒頭の docstring にある。",
        "",
        f"評価対象: out-of-fold 予測 {len(y)} 件（不良 {int(y.sum())} 件）。",
        "",
        "## 平均 AUC の区間",
        "",
        "| 平均 AUC | 標準誤差 | 95% 区間 | 0.5 に対する p（両側） |",
        "|---|---|---|---|",
        f"| {mean:.4f} | {se:.4f} | [{lo:.3f}, {hi:.3f}] | {p:.4f} |",
        "",
        "fold を独立とみなした DeLong の区間で、学習のばらつきは含まない。",
        "",
        f"## 固定しきい値 {threshold:.4f} を各 fold に当てた場合",
        "",
        "| fold | 検査に回る割合 | 再現率 |",
        "|---|---|---|",
    ]
    for name, rate, tp, pos in fixed_threshold_rows(proba, y, fold, threshold):
        lines.append(f"| {name} | {rate:.1%} | {tp / pos:.1%}（{tp}/{pos}） |")
    lines += [
        "",
        "README の再現率 66.2% は各 fold の中で順位を付けて上位 50% を選んだ値で、この表とは違う。",
        "",
    ]
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"保存: {OUT}")


if __name__ == "__main__":
    main()
