"""最終モデルの学習と保存。

分析ノートブック（notebooks/secom_strict_analysis.ipynb）の結論である
「きちんとした前処理 + Random Forest」を、全データで学習して ``models/`` に保存する。

ポイントは ``SecomPreprocessor`` を ``src.preprocessing`` から import していること。
こうして保存した pickle は ``src.preprocessing.SecomPreprocessor`` というパスを記録するため、
同じく ``src.preprocessing`` を import する推論サーバー（src/api.py）から復元できる。

使い方（リポジトリのルートで実行）::

    python -m src.train
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import SecomPreprocessor

# --- パス設定 -----------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "secom.data")
LABELS_PATH = os.path.join(ROOT, "data", "secom_labels.data")
MODELS_DIR = os.path.join(ROOT, "models")

# 厳密評価（Nested CV）で得た真の実力値。ノートブックの分析結果をそのまま記録する。
EVAL_AUC_STRICT = 0.6074
EVAL_PR_AUC_STRICT = 0.1346

# モデルの版。学習構成（前処理・アルゴリズム・ハイパラ）を変えたら手で上げる。
# 推論レスポンスとログに刻むことで、「どの予測がどのモデルから出たか」を後から辿れる。
MODEL_VERSION = "1.0.0"


def load_data():
    """SECOM の特徴量とラベルを読み込む。"""
    df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, na_values="NaN")

    labels_raw = pd.read_csv(
        LABELS_PATH, sep=r'\s(?=")', header=None,
        names=["label", "datetime_str"], engine="python",
    )
    y = (labels_raw["label"] == 1).astype(int).values
    return df, y


def main():
    df, y = load_data()
    print(f"データ読み込み: {df.shape}  不良率 {y.mean() * 100:.2f}%")

    # 全データで前処理とモデルを学習（本番デプロイ用の最終モデル）
    preprocessor = SecomPreprocessor()
    X = preprocessor.fit_transform(df)
    print("前処理サマリー:", preprocessor.fit_summary_)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    model.fit(X, y)

    metadata = {
        "version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_type": "RandomForestClassifier",
        "n_estimators": 200,
        "max_depth": 10,
        "class_weight": "balanced",
        "random_state": 42,
        "input_features": int(df.shape[1]),          # 生の特徴量数（API 入力の長さ）
        "preprocessed_features": int(X.shape[1]),     # 前処理後の特徴量数
        "training_samples": int(len(df)),
        "training_fail_rate": float(y.mean()),
        "evaluation_AUC_strict": EVAL_AUC_STRICT,
        "evaluation_PR_AUC_strict": EVAL_PR_AUC_STRICT,
        "default_threshold": 0.5,
        "description": "SECOM 不良予測の最終モデル（厳密評価版）",
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.pkl"))
    joblib.dump(model, os.path.join(MODELS_DIR, "model.pkl"))
    joblib.dump(metadata, os.path.join(MODELS_DIR, "metadata.pkl"))

    print(f"=== 保存完了（model version {metadata['version']} / {metadata['trained_at']}） ===")
    for name in ("preprocessor.pkl", "model.pkl", "metadata.pkl"):
        path = os.path.join(MODELS_DIR, name)
        print(f"  models/{name} ({os.path.getsize(path) / 1024:.1f} KB)")
    print(f"入力特徴量数: {metadata['input_features']}  ->  前処理後: {metadata['preprocessed_features']}")


if __name__ == "__main__":
    main()
