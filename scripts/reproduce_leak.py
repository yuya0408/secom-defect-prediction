"""データリークで AUC がどれだけ高く見えるかを、同じ構成で比べて確かめる。

当初の分析で AUC 0.72 が出たコードはリポジトリに残っていない。そこで、記事に書いた手順
（全データで前処理と特徴量重要度を計算してから、時系列CVで評価する）を作り直し、同じ構成を
fold 内に閉じた場合（scripts/compare_configs.py の E-50 / E-100）と比べる。
作り直したものなので、当初の 0.7180 と小数点以下まで同じになるとは限らない。

使い方（リポジトリのルートで）::

    python -m scripts.reproduce_leak

出力:
    reports/leak_check.md
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from scripts.compare_configs import config_oof, make_time_features
from src.preprocessing import SecomPreprocessor
from src.train import N_SPLITS, ROOT, build_model, load_data

REPORT_PATH = os.path.join(ROOT, "reports", "leak_check.md")
TOP_FEATURES = (50, 100)


def leaked_auc(X, y, n_top):
    """全データの重要度で列を選んでから時系列CVで評価する（これがリーク）。"""
    imp = RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight="balanced",
        random_state=42, n_jobs=-1,
    ).fit(X, y).feature_importances_
    top = np.argsort(imp)[::-1][:n_top]
    aucs = [
        roc_auc_score(y[te], build_model().fit(X[tr][:, top], y[tr]).predict_proba(X[te][:, top])[:, 1])
        for tr, te in TimeSeriesSplit(N_SPLITS).split(X)
    ]
    return float(np.mean(aucs))


def strict_auc(df, y, datetimes, n_top):
    proba, y_oof, fold = config_oof(df, y, datetimes, True, True, n_top)
    return float(np.mean([roc_auc_score(y_oof[fold == k], proba[fold == k]) for k in np.unique(fold)]))


def main():
    df, y, ts = load_data()
    datetimes = pd.Series(pd.to_datetime(ts))
    interval_median = (datetimes.diff().dt.total_seconds() / 3600).median()
    X = np.hstack([SecomPreprocessor().fit_transform(df), make_time_features(datetimes, y, interval_median)])

    rows = [(n, leaked_auc(X, y, n), strict_auc(df, y, datetimes, n)) for n in TOP_FEATURES]
    gap = max(leak - strict for _, leak, strict in rows)
    lines = [
        "# データリークで AUC がどれだけ高く見えるか",
        "",
        f"**結論: 同じ構成でも、全データで特徴量を選ぶと AUC は最大 {gap:.3f} 高く見える。**",
        "",
        "`python -m scripts.reproduce_leak` で再生成。当初のコードは残っていないので、記事に書いた手順を作り直した値である。",
        "",
        "| 特徴量選択 | リークあり（全データで選択） | リークなし（fold 内で選択） | 差 |",
        "|---|---|---|---|",
    ] + [f"| 上位 {n} 列 | {leak:.4f} | {strict:.4f} | {leak - strict:+.4f} |" for n, leak, strict in rows]
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n保存: {os.path.relpath(REPORT_PATH, ROOT)}")


if __name__ == "__main__":
    main()
