"""最終モデルの学習と保存。

分析ノートブック（notebooks/secom_strict_analysis.ipynb）の結論である
「きちんとした前処理 + Random Forest」を、全データで学習して ``models/`` に保存する。
あわせて、リークを排除した時系列CVを学習のたびに回し直し、評価値と既定しきい値を
その out-of-fold 予測から決めて ``metadata.pkl`` に記録する。

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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from src.preprocessing import SecomPreprocessor

# --- パス設定 -----------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "secom.data")
LABELS_PATH = os.path.join(ROOT, "data", "secom_labels.data")
MODELS_DIR = os.path.join(ROOT, "models")

# 厳密評価（リーク排除した時系列CV）の分割数。ノートブックの分析と同じ 5 分割。
N_SPLITS = 5

# 既定しきい値の決め方: 「不良確率の上位 INSPECT_FRACTION を検査に回す」。
# 業務上の仮定であり、データから一意に決まる値ではない。
# - 確率の値ではなく割合で決めるのは、学習データが増えるほど確率が低く出て
#   （fold ごとの平均 0.15 → 0.07）、値で固定したしきい値は再現率が大きく振れるため。
# - 50% は「流出1件の損失が検査1件の約20倍」の想定で費用最小に近く、fold 間の再現率の
#   ばらつきも最小だった。流出の損失がそれより大きい（約50倍以上）現場では全数検査の方が
#   安く、このモデルは検査の優先順位づけにしか使えない。
INSPECT_FRACTION = 0.5

# 評価表に載せる検査割合の刻み。
INSPECT_FRACTIONS_REPORTED = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

# モデルの版。学習構成（前処理・アルゴリズム・ハイパラ）を変えたら手で上げる。
# 推論レスポンスとログに刻むことで、「どの予測がどのモデルから出たか」を後から辿れる。
MODEL_VERSION = "1.1.0"


def load_data():
    """SECOM の特徴量・ラベル・検査タイムスタンプを読み込む（行の並びはファイルのまま）。"""
    df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, na_values="NaN")

    labels_raw = pd.read_csv(
        LABELS_PATH, sep=r'\s(?=")', header=None,
        names=["label", "datetime_str"], engine="python",
    )
    y = (labels_raw["label"] == 1).astype(int).values
    ts = pd.to_datetime(labels_raw["datetime_str"].str.strip('"'), format="%d/%m/%Y %H:%M:%S")
    return df, y, ts.values


def build_model():
    """最終モデルと評価で同じ設定を使うためのファクトリ。"""
    return RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )


def time_series_oof(df, y, ts):
    """時刻順に並べた時系列CVで、fold 内に閉じた前処理 + モデルの out-of-fold 予測を作る。

    最初の 1/(N_SPLITS+1) は学習専用で予測を持たないため、返すのはテスト側の行だけ。
    """
    order = np.argsort(ts, kind="stable")
    X_all, y_all = df.iloc[order].reset_index(drop=True), y[order]
    proba, fold, test_idx = [], [], []
    for k, (tr, te) in enumerate(TimeSeriesSplit(N_SPLITS).split(X_all)):
        pre = SecomPreprocessor()
        model = build_model().fit(pre.fit_transform(X_all.iloc[tr]), y_all[tr])
        proba.append(model.predict_proba(pre.transform(X_all.iloc[te]))[:, 1])
        fold.append(np.full(len(te), k))
        test_idx.append(te)
    return np.concatenate(proba), y_all[np.concatenate(test_idx)], np.concatenate(fold)


def flag_top(proba, fold, fraction):
    """各 fold の中で不良確率の上位 fraction を検査対象にする（確率の水準のずれを避ける）。"""
    flagged = np.zeros(len(proba), dtype=bool)
    for k in np.unique(fold):
        idx = np.where(fold == k)[0]
        n = int(np.ceil(fraction * len(idx)))
        flagged[idx[np.argsort(-proba[idx], kind="stable")[:n]]] = True
    return flagged


def inspection_table(proba, y, fold):
    """検査割合ごとの再現率（全体と fold 別）。ランダムに選んだ場合の再現率は検査割合に等しい。"""
    rows = []
    for q in INSPECT_FRACTIONS_REPORTED:
        f = flag_top(proba, fold, q)
        rows.append({
            "inspect_fraction": q,
            "recall": float((f & (y == 1)).sum() / (y == 1).sum()),
            "precision": float((f & (y == 1)).sum() / f.sum()),
            "recall_by_fold": [
                float((f[fold == k] & (y[fold == k] == 1)).sum() / (y[fold == k] == 1).sum())
                for k in np.unique(fold)
            ],
        })
    return rows


def main():
    df, y, ts = load_data()
    print(f"データ読み込み: {df.shape}  不良率 {y.mean() * 100:.2f}%")

    # 厳密評価（リーク排除した時系列CV）
    oof_proba, oof_y, oof_fold = time_series_oof(df, y, ts)
    folds = np.unique(oof_fold)
    auc_by_fold = [roc_auc_score(oof_y[oof_fold == k], oof_proba[oof_fold == k]) for k in folds]
    pr_auc_by_fold = [average_precision_score(oof_y[oof_fold == k], oof_proba[oof_fold == k]) for k in folds]
    table = inspection_table(oof_proba, oof_y, oof_fold)
    print(f"時系列CV: AUC {np.mean(auc_by_fold):.4f} (fold別 {np.round(auc_by_fold, 3).tolist()})"
          f"  PR-AUC {np.mean(pr_auc_by_fold):.4f}")
    for row in table:
        print(f"  上位{row['inspect_fraction']:.0%}を検査 -> 再現率 {row['recall']:.1%}"
              f" (fold別 {[round(r, 2) for r in row['recall_by_fold']]})")

    # 既定しきい値: 最終モデルに一番近い最後の fold（学習件数が最多）の予測分布で、
    # 上位 INSPECT_FRACTION に当たる確率の値。全データで学習する最終モデルはさらに確率が
    # 低く出うるので近似値。運用では直近の予測ログの分位点で置き直す前提。
    last = oof_fold == folds[-1]
    default_threshold = float(np.quantile(oof_proba[last], 1 - INSPECT_FRACTION))
    print(f"既定しきい値: {default_threshold:.4f}（最後の fold で上位{INSPECT_FRACTION:.0%}に当たる値）")

    # 全データで前処理とモデルを学習（本番デプロイ用の最終モデル）
    preprocessor = SecomPreprocessor()
    X = preprocessor.fit_transform(df)
    print("前処理サマリー:", preprocessor.fit_summary_)

    model = build_model()
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
        "evaluation_AUC_strict": round(float(np.mean(auc_by_fold)), 4),
        "evaluation_AUC_strict_by_fold": [round(float(a), 4) for a in auc_by_fold],
        "evaluation_PR_AUC_strict": round(float(np.mean(pr_auc_by_fold)), 4),
        "default_threshold": round(default_threshold, 4),
        "threshold_rule": {
            "rule": f"不良確率の上位{INSPECT_FRACTION:.0%}を検査に回す",
            "inspect_fraction": INSPECT_FRACTION,
            "assumption": "流出1件の損失が検査1件の約20倍。約50倍以上なら全数検査の方が安い",
            "source": "時系列CVの最後の fold の out-of-fold 予測の分位点（最終モデルに対しては近似）",
            "inspection_table": table,
        },
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
