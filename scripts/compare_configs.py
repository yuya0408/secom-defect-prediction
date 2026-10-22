"""前処理・特徴量の構成の比較（モデルは RF に固定）。

考察編 §4 とノートブック（notebooks/secom_strict_analysis.ipynb の Step 5）の 6 構成を、
同じ厳密な時系列CVで評価し直し、現行の C（きちんとした前処理のみ）との差を検定する。
ノートブックは平均 AUC の並びだけで「C が最良」としていたが、差の検定はしていなかった。

構成の定義はノートブックの ``evaluate_strict`` / ``make_time_features`` をそのまま移した:
- A: 雑な前処理（定数列の削除 + 中央値補完）のみ
- B: A + 時系列特徴量
- C: きちんとした前処理（SecomPreprocessor）のみ ← 現行・基準
- D: C + 時系列特徴量（時刻・生産順・直前 1/3/5/10 件の不良数・前の品からの間隔）
- E-50 / E-100: D + 特徴量選択（訓練データの RF 重要度の上位 50 / 100 列）

結果を見る前に決めた規則:
- 差（構成 − C）は fold 別 DeLong 検定（scripts/auc_tests.py）、5 つの比較に Holm 補正。
- Holm 補正後の p < 0.05 でなければ「見分けられない」とし、「C が最良」とは書かない。

使い方（リポジトリのルートで）::

    python -m scripts.compare_configs

出力:
    reports/config_comparison.md
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

from scripts.compare_models import result_table
from src.preprocessing import SecomPreprocessor
from src.train import N_SPLITS, ROOT, build_model, load_data

REPORT_PATH = os.path.join(ROOT, "reports", "config_comparison.md")

CONFIGS = {
    "A: 雑な前処理": dict(use_preprocessor=False, use_time_features=False),
    "B: 雑な前処理 + 時系列": dict(use_preprocessor=False, use_time_features=True),
    "C: きちんと前処理（現行）": dict(use_preprocessor=True, use_time_features=False),
    "D: C + 時系列": dict(use_preprocessor=True, use_time_features=True),
    "E-50: D + 特徴量選択 50": dict(use_preprocessor=True, use_time_features=True, n_top_features=50),
    "E-100: D + 特徴量選択 100": dict(use_preprocessor=True, use_time_features=True, n_top_features=100),
}
BASELINE = "C: きちんと前処理（現行）"


def make_time_features(datetimes, y_full, interval_median):
    """時系列特徴量。``shift(1)`` で過去の品の不良だけを参照する（ノートブックと同じ）。"""
    feats = pd.DataFrame({
        "production_order": range(len(datetimes)),
        "hour": datetimes.dt.hour,
        "day_of_week": datetimes.dt.dayofweek,
        "month": datetimes.dt.month,
        "days_from_start": (datetimes - datetimes.min()).dt.total_seconds() / 86400,
    })
    is_fail = pd.Series(y_full)
    for window in [1, 3, 5, 10]:
        feats[f"prev_{window}_fail_count"] = (
            is_fail.shift(1).rolling(window=window, min_periods=1).sum().fillna(0)
        )
    intervals = datetimes.diff().dt.total_seconds() / 3600
    feats["interval_hours"] = intervals.fillna(interval_median)
    return feats.values


def config_oof(df, y, datetimes, use_preprocessor, use_time_features, n_top_features=None):
    """1 構成分の out-of-fold 予測。前処理・特徴量選択・モデルはすべて fold 内で学習する。"""
    proba, fold = [], []
    for k, (tr, te) in enumerate(TimeSeriesSplit(N_SPLITS).split(df)):
        if use_preprocessor:
            pre = SecomPreprocessor()
            X_tr, X_te = pre.fit_transform(df.iloc[tr]), pre.transform(df.iloc[te])
        else:
            const_cols = df.iloc[tr].columns[df.iloc[tr].nunique() <= 1]
            keep = [c for c in df.columns if c not in set(const_cols)]
            med = df[keep].iloc[tr].median()
            X_tr, X_te = df[keep].iloc[tr].fillna(med).values, df[keep].iloc[te].fillna(med).values
        if use_time_features:
            interval_median = (datetimes.iloc[tr].diff().dt.total_seconds() / 3600).median()
            time_X = make_time_features(datetimes, y, interval_median)
            X_tr, X_te = np.hstack([X_tr, time_X[tr]]), np.hstack([X_te, time_X[te]])
        if n_top_features is not None and n_top_features < X_tr.shape[1]:
            imp = RandomForestClassifier(
                n_estimators=300, max_depth=10, class_weight="balanced",
                random_state=42, n_jobs=-1,
            ).fit(X_tr, y[tr]).feature_importances_
            top = np.argsort(imp)[::-1][:n_top_features]
            X_tr, X_te = X_tr[:, top], X_te[:, top]
        proba.append(build_model().fit(X_tr, y[tr]).predict_proba(X_te)[:, 1])
        fold.append(np.full(len(te), k))
    test_idx = np.concatenate([te for _, te in TimeSeriesSplit(N_SPLITS).split(df)])
    return np.concatenate(proba), y[test_idx], np.concatenate(fold)


def main():
    df, y, ts = load_data()
    # 時系列CV はファイルの行順で分割する。ファイルが時刻順であることを確かめておく。
    assert (np.diff(ts.astype("int64")) >= 0).all(), "secom_labels.data が時刻順ではない"
    datetimes = pd.Series(pd.to_datetime(ts))

    proba = {}
    for name, kwargs in CONFIGS.items():
        print(f"評価中: {name}")
        proba[name], y_oof, fold = config_oof(df, y, datetimes, **kwargs)

    lines = [
        "# 前処理・特徴量の構成の比較（厳密な時系列CV、モデルは RF）",
        "",
        "`python -m scripts.compare_configs` で再生成。規則はスクリプト冒頭の docstring に、結果を見る前に記した。",
        "",
        f"評価対象: out-of-fold 予測 {len(y_oof)} 件（不良 {int(y_oof.sum())} 件）。"
        "基準 C との差を fold 別 DeLong 検定で検定し、5 つの比較に Holm 補正。",
        "",
    ] + result_table(proba, BASELINE, y_oof, fold, switch_label="C より良い")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n保存: {os.path.relpath(REPORT_PATH, ROOT)}")


if __name__ == "__main__":
    main()
