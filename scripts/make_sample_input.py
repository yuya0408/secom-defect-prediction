"""テストデータから API 用のサンプル入力 JSON を生成する。

590 個のセンサー値を手で書くのは現実的でないため、実データの行をそのまま
``sample_input.json`` に書き出す。欠損値は JSON の ``null`` として保持されるので、
API の欠損補完ロジックの動作確認にもなる。

既定では「実際に不良だった行」を 1 件選んで出力する（デモとして分かりやすいため）。

使い方（リポジトリのルートで）::

    python -m scripts.make_sample_input
"""

from __future__ import annotations

import json
import math
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "secom.data")
LABELS_PATH = os.path.join(ROOT, "data", "secom_labels.data")
OUT_PATH = os.path.join(ROOT, "sample_input.json")


def main():
    df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, na_values="NaN")
    labels = pd.read_csv(
        LABELS_PATH, sep=r'\s(?=")', header=None,
        names=["label", "datetime_str"], engine="python",
    )
    y = (labels["label"] == 1).astype(int).values

    # 実際に不良だった最初の行を採用（なければ先頭行）
    fail_idx = next((i for i, v in enumerate(y) if v == 1), 0)
    row = df.iloc[fail_idx].tolist()

    # NaN -> None（json.dump が null として書き出す）
    features = [None if (isinstance(v, float) and math.isnan(v)) else float(v) for v in row]

    payload = {"features": features}
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"行 {fail_idx}（実ラベル: {'不良' if y[fail_idx] == 1 else '合格'}）を出力")
    print(f"特徴量数: {len(features)}  欠損(null): {sum(v is None for v in features)}")
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
