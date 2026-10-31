"""品種グループの情報をモデルに入れる方法の比較（モデルは RF、前処理は現行）。

考察編の「品種情報をモデルに使えるか」の 4 手法を、学習（src/train.py）と同じ時系列CVで
評価し、現行（グループ情報なし）との差を検定する。

- One-Hot: グループ ID を One-Hot で追加（訓練データのグループだけで fit）
- 不良率の特徴量: 訓練データで計算したグループ別不良率を追加（件数 10 相当の平滑化、未知は全体の不良率）
- グループ別 K: 訓練データに K 件以上あるグループだけ専用モデルを学習し、残りは全体モデル

規則について:
- 平均 AUC は公開時に見ていた。差の検定はそのあとで足したので、「結果を見る前に決めた規則」
  ではない。比較の数（5 つ）に Holm 補正をかけ、p < 0.05 でなければ「見分けられない」とする。
- グループの定義（欠損パターン）は全データから作っている。不良ラベルは使わないが、未来の行を
  列の選定と ID の付与に使っている点は厳密ではない。

使い方（リポジトリのルートで）::

    python -m scripts.compare_group_info

出力:
    reports/group_info_comparison.md
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder

from scripts.auc_tests import compare_to_baseline, holm
from scripts.compare_models import conclusion_line, result_table
from src.preprocessing import SecomPreprocessor
from src.train import N_SPLITS, ROOT, build_model, load_data

REPORT_PATH = os.path.join(ROOT, "reports", "group_info_comparison.md")
BASELINE = "現行（グループ情報なし）"
GROUP_MISSING_LOW, GROUP_MISSING_HIGH = 0.10, 0.70
TARGET_ENCODING_SMOOTHING = 10


def missing_pattern_groups(df):
    """欠損率 10〜70% の列の欠損パターンで、多い順にグループ ID を振る（ノートブックと同じ）。"""
    rate = df.isna().mean()
    cols = rate[(rate >= GROUP_MISSING_LOW) & (rate <= GROUP_MISSING_HIGH)].index
    signature = df[cols].isna().astype(int).astype(str).apply("".join, axis=1)
    order = {s: i for i, s in enumerate(signature.value_counts().index)}
    return signature.map(order).to_numpy(), len(cols)


def _oof(df, y, groups, predict_fold):
    """各 fold で前処理を訓練データに閉じ、predict_fold でテスト側の確率を作る。"""
    proba, fold, test_idx = [], [], []
    for k, (tr, te) in enumerate(TimeSeriesSplit(N_SPLITS).split(df)):
        pre = SecomPreprocessor()
        X_tr, X_te = pre.fit_transform(df.iloc[tr]), pre.transform(df.iloc[te])
        proba.append(predict_fold(X_tr, X_te, y[tr], groups[tr], groups[te]))
        fold.append(np.full(len(te), k))
        test_idx.append(te)
    return np.concatenate(proba), y[np.concatenate(test_idx)], np.concatenate(fold)


def baseline(X_tr, X_te, y_tr, g_tr, g_te):
    return build_model().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]


def one_hot(X_tr, X_te, y_tr, g_tr, g_te):
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(g_tr.reshape(-1, 1))
    X_tr = np.hstack([X_tr, enc.transform(g_tr.reshape(-1, 1))])
    X_te = np.hstack([X_te, enc.transform(g_te.reshape(-1, 1))])
    return build_model().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]


def target_encoding(X_tr, X_te, y_tr, g_tr, g_te):
    prior, s = y_tr.mean(), TARGET_ENCODING_SMOOTHING
    rate = {g: (y_tr[g_tr == g].sum() + s * prior) / ((g_tr == g).sum() + s) for g in np.unique(g_tr)}
    f_tr = np.array([rate[g] for g in g_tr])
    f_te = np.array([rate.get(g, prior) for g in g_te])
    return build_model().fit(np.c_[X_tr, f_tr], y_tr).predict_proba(np.c_[X_te, f_te])[:, 1]


def per_group(min_samples):
    def predict(X_tr, X_te, y_tr, g_tr, g_te):
        proba = build_model().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
        for g in np.unique(g_te):
            in_tr = g_tr == g
            # 不良が 0 件か全件のグループは二値分類を学習できないので、全体モデルに任せる。
            if in_tr.sum() >= min_samples and 0 < y_tr[in_tr].sum() < in_tr.sum():
                in_te = g_te == g
                proba[in_te] = build_model().fit(X_tr[in_tr], y_tr[in_tr]).predict_proba(X_te[in_te])[:, 1]
        return proba
    return predict


METHODS = {
    BASELINE: baseline,
    "One-Hot": one_hot,
    "不良率の特徴量": target_encoding,
    "グループ別 100": per_group(100),
    "グループ別 50": per_group(50),
    "グループ別 30": per_group(30),
}


def evaluate():
    """手法ごとの差と Holm 補正後の p を返す（図 4 からも使う）。"""
    df, y, _ = load_data()
    groups, _ = missing_pattern_groups(df)
    proba = {}
    for name, method in METHODS.items():
        print(f"評価中: {name}")
        proba[name], y_oof, fold = _oof(df, y, groups, method)
    tests = compare_to_baseline(proba, BASELINE, y_oof, fold)
    p_holm = dict(zip(tests, holm([t["p"] for t in tests.values()])))
    return proba, y_oof, fold, tests, p_holm


def main():
    proba, y_oof, fold, _, _ = evaluate()
    table = result_table(proba, BASELINE, y_oof, fold, switch_label="現行より良い")
    lines = [
        "# 品種グループの情報を入れる方法の比較（厳密な時系列CV、モデルは RF）",
        "",
        conclusion_line(table),
        "",
        "`python -m scripts.compare_group_info` で再生成。平均 AUC を見た後で検定を足したので、規則は事前に決めたものではない。",
        "",
        f"評価対象: out-of-fold 予測 {len(y_oof)} 件（不良 {int(y_oof.sum())} 件）。"
        "現行との差を fold 別 DeLong 検定で検定し、5 つの比較に Holm 補正。差の 95% 区間は補正前。",
        "",
    ] + table
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n保存: {os.path.relpath(REPORT_PATH, ROOT)}")


if __name__ == "__main__":
    main()
