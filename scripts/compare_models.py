"""モデルの比較: Random Forest とロジスティック回帰・HistGradientBoosting。

RF は比較で選んだモデルではなく、「高次元・少サンプルの表形式データで、チューニングを
ほとんどせずに崩れにくい出発点」として事前の理由で置いたもの。ここで初めて、学習
（src/train.py）と同じ厳密な時系列CV（fold 内に閉じた前処理・時刻順の 5 分割）で比べる。

結果を見る前に決めた規則:
- 相手のハイパラは既定値に ``class_weight="balanced"`` だけを足す（RF もチューニングしていない）。
  ロジスティック回帰は係数に罰則がかかるため標準化を入れる。
- 指標は fold 別 AUC の平均。差（相手 − RF）は fold 別 DeLong 検定（scripts/auc_tests.py）で
  検定し、2 つの比較に Holm 補正をかける。
- RF を替えるのは、Holm 補正後の p < 0.05 で相手の方が良いときだけ。替えると API・監視の
  基準・文書をやり直す費用がかかるため、見分けられない差では替えない。
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

from scripts.auc_tests import compare_to_baseline, holm
from src.train import ROOT, build_model, load_data, time_series_oof

ALPHA = 0.05
REPORT_PATH = os.path.join(ROOT, "reports", "model_comparison.md")

CANDIDATES = {
    "LogisticRegression": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000),
    ),
    "HistGradientBoosting": lambda: HistGradientBoostingClassifier(
        class_weight="balanced", random_state=42,
    ),
}


def result_table(proba_by_name, baseline, y, fold, switch_label):
    """基準との差の表（Markdown の行）。compare_configs からも使う。"""
    tests = compare_to_baseline(proba_by_name, baseline, y, fold)
    p_holm = dict(zip(tests, holm([t["p"] for t in tests.values()])))
    lines = [
        "| 構成 | AUC（fold 平均） | fold 別 | 基準との差 | 差の 95% 区間 | p | p（Holm） | 判定 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, p in proba_by_name.items():
        by_fold = [roc_auc_score(y[fold == k], p[fold == k]) for k in np.unique(fold)]
        head = f"| {name} | {np.mean(by_fold):.4f} | {' / '.join(f'{a:.3f}' for a in by_fold)} |"
        if name == baseline:
            lines.append(head + " — | — | — | — | 基準 |")
            continue
        t = tests[name]
        if p_holm[name] >= ALPHA:
            verdict = "見分けられない"
        elif t["diff"] > 0:
            verdict = switch_label
        else:
            verdict = "基準より悪い"
        lines.append(head + f" {t['diff']:+.4f} | [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}] |"
                     f" {t['p']:.3f} | {p_holm[name]:.3f} | {verdict} |")
    return lines


def conclusion_line(table_lines):
    """表の判定列から、レポート冒頭に置く結論の 1 行を作る。"""
    rows = [l for l in table_lines[2:] if not l.endswith("基準 |")]
    n_undistinguished = sum(l.endswith("見分けられない |") for l in rows)
    if n_undistinguished == len(rows):
        return f"**結論: {len(rows)} つの比較のどれも、基準と見分けられない（Holm 補正後 p ≥ {ALPHA}）。「同等」という意味ではない。**"
    return f"**結論: {len(rows)} つの比較のうち {len(rows) - n_undistinguished} つで基準との差を検出した（表の判定列）。**"


def main():
    df, y, ts = load_data()
    oof = {"RandomForest": time_series_oof(df, y, ts, build_model)}
    for name, make in CANDIDATES.items():
        print(f"評価中: {name}")
        oof[name] = time_series_oof(df, y, ts, make)

    _, y_oof, fold = oof["RandomForest"]
    for _, y_k, fold_k in oof.values():
        assert np.array_equal(y_k, y_oof) and np.array_equal(fold_k, fold)
    table = result_table({n: o[0] for n, o in oof.items()}, "RandomForest", y_oof, fold,
                         switch_label="RF を替える")
    lines = [
        "# モデルの比較（厳密な時系列CV）",
        "",
        conclusion_line(table),
        "",
        "`python -m scripts.compare_models` で再生成。規則はスクリプト冒頭の docstring に、結果を見る前に記した。",
        "",
        f"評価対象: out-of-fold 予測 {len(y_oof)} 件（不良 {int(y_oof.sum())} 件）。"
        "差の検定は fold 別 DeLong 検定、多重比較は Holm 補正。差の 95% 区間は補正前。",
        "",
    ] + table

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n保存: {os.path.relpath(REPORT_PATH, ROOT)}")


if __name__ == "__main__":
    main()
