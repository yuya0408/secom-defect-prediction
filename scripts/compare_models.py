"""モデルの比較: Random Forest とロジスティック回帰・HistGradientBoosting。

RF は比較で選んだモデルではなく、「高次元・少サンプルの表形式データで、チューニングを
ほとんどせずに崩れにくい出発点」として事前の理由で置いたもの。ここで初めて、学習
（src/train.py）と同じ厳密な時系列CV（fold 内に閉じた前処理・時刻順の 5 分割）で比べる。

結果を見る前に決めた規則:
- 相手のハイパラは既定値に ``class_weight="balanced"`` だけを足す（RF もチューニングしていない）。
  ロジスティック回帰は係数に罰則がかかるため標準化を入れる。
- 指標は fold 別 AUC の平均。差（相手 − RF）の 95% 区間は、fold の中で不良・良品を
  それぞれ復元抽出する対応のあるブートストラップで求める。
- RF を替えるのは、区間が 0 をまたがず相手の方が良いときだけ。替えるとAPI・監視の基準・
  文書をやり直す費用がかかるため、見分けられない差では替えない。
- 不良は 65 件しかなく検出力は低い。「差なし」は「同等」ではなく「見分けられない」と読む。

使い方（リポジトリのルートで）::

    python -m scripts.compare_models

出力:
    reports/model_comparison.md
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.train import ROOT, build_model, load_data, time_series_oof

N_BOOT = 2000
SEED = 0
REPORT_PATH = os.path.join(ROOT, "reports", "model_comparison.md")

CANDIDATES = {
    "LogisticRegression": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000),
    ),
    "HistGradientBoosting": lambda: HistGradientBoostingClassifier(
        class_weight="balanced", random_state=42,
    ),
}


def mean_fold_auc(proba, y, fold):
    return float(np.mean([roc_auc_score(y[fold == k], proba[fold == k]) for k in np.unique(fold)]))


def paired_bootstrap_diff(proba_a, proba_b, y, fold, n_boot=N_BOOT, seed=SEED):
    """fold 別 AUC 平均の差（b − a）の分布。同じ行を両モデルで引くので対応がある。

    fold の中で不良・良品を別々に復元抽出する。不良が数件しかない fold で、不良 0 件の
    標本ができて AUC が計算できなくなるのを避けるため。
    """
    rng = np.random.default_rng(seed)
    strata = [np.where((fold == k) & (y == c))[0] for k in np.unique(fold) for c in (0, 1)]
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
        diffs[b] = (mean_fold_auc(proba_b[idx], y[idx], fold[idx])
                    - mean_fold_auc(proba_a[idx], y[idx], fold[idx]))
    return diffs


def main():
    df, y, ts = load_data()
    oof = {"RandomForest": time_series_oof(df, y, ts, build_model)}
    for name, make in CANDIDATES.items():
        print(f"評価中: {name}")
        oof[name] = time_series_oof(df, y, ts, make)

    p_rf, y_oof, fold = oof["RandomForest"]
    lines = [
        "# モデルの比較（厳密な時系列CV）",
        "",
        "`python -m scripts.compare_models` で再生成。規則はスクリプト冒頭の docstring に、結果を見る前に記した。",
        "",
        f"評価対象: out-of-fold 予測 {len(y_oof)} 件（不良 {int(y_oof.sum())} 件）。"
        f"差の区間は fold 内で層別した対応のあるブートストラップ（{N_BOOT} 回）。",
        "",
        "| モデル | AUC（fold 平均） | fold 別 | RF との差 | 差の 95% 区間 | 判定 |",
        "|---|---|---|---|---|---|",
    ]
    for name, (p, y_k, fold_k) in oof.items():
        assert np.array_equal(y_k, y_oof) and np.array_equal(fold_k, fold)
        by_fold = [roc_auc_score(y_oof[fold == k], p[fold == k]) for k in np.unique(fold)]
        by_fold_s = " / ".join(f"{a:.3f}" for a in by_fold)
        if name == "RandomForest":
            lines.append(f"| {name}（現行） | {np.mean(by_fold):.4f} | {by_fold_s} | — | — | 基準 |")
            continue
        diffs = paired_bootstrap_diff(p_rf, p, y_oof, fold)
        lo, hi = np.quantile(diffs, [0.025, 0.975])
        diff = np.mean(by_fold) - mean_fold_auc(p_rf, y_oof, fold)
        verdict = "RF を替える" if lo > 0 else "替えない（RF より良いとは見分けられない）"
        lines.append(f"| {name} | {np.mean(by_fold):.4f} | {by_fold_s} | {diff:+.4f} | "
                     f"[{lo:+.4f}, {hi:+.4f}] | {verdict} |")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n保存: {os.path.relpath(REPORT_PATH, ROOT)}")


if __name__ == "__main__":
    main()
