"""SECOM 用の前処理クラス（データリーク防止版）。

このモジュールは「分析ノートブックで定義したクラスを、推論サーバーから
import できる形に切り出したもの」である。

なぜ独立モジュールにするのか
------------------------------
``joblib.dump(preprocessor)`` で保存した pickle には、クラスそのものではなく
「クラスがどのモジュールに定義されているか（例: ``src.preprocessing.SecomPreprocessor``）」
というパス情報が記録される。読み込み側（FastAPI のプロセス）で同じパスから
import できないと ``joblib.load`` は失敗する。

ノートブック内で定義したまま保存すると、そのパスは ``__main__.SecomPreprocessor``
になり、別プロセスである API からは復元できない。そこで前処理クラスを本モジュールへ
切り出し、学習スクリプトと API の双方が ``from src.preprocessing import SecomPreprocessor``
で参照する。これにより pickle のパスが一貫し、どこからでも安全に読み込める。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SecomPreprocessor(BaseEstimator, TransformerMixin):
    """SECOM 用の前処理クラス（データリーク防止）。

    ``fit()`` で訓練データから削除列・補完値などを学習し、``transform()`` で
    新しいデータに適用する。学習時の統計量だけを使うため、評価時にも本番運用時にも
    テストデータの情報が漏れ込まない（リークしない）。

    Parameters
    ----------
    quasi_const_threshold : float
        準定数列の判定基準（最頻値の出現率がこの値以上なら削除）。
    high_missing_threshold : float
        高欠損列の判定基準（欠損率がこの値以上なら削除）。
    flag_missing_low, flag_missing_high : float
        欠損フラグを特徴量化する欠損率の範囲 [low, high)。
    """

    def __init__(self, quasi_const_threshold=0.95, high_missing_threshold=0.5,
                 flag_missing_low=0.05, flag_missing_high=0.5):
        self.quasi_const_threshold = quasi_const_threshold
        self.high_missing_threshold = high_missing_threshold
        self.flag_missing_low = flag_missing_low
        self.flag_missing_high = flag_missing_high

    def fit(self, X, y=None):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        df.columns = [str(c) for c in df.columns]

        # 1. 定数列の検出
        self.constant_cols_ = df.columns[df.nunique() <= 1].tolist()

        # 2. 準定数列の検出
        self.quasi_constant_cols_ = []
        for col in df.columns:
            if col in self.constant_cols_:
                continue
            if df[col].notna().any():
                top_freq = df[col].value_counts(normalize=True, dropna=True).iloc[0]
                if top_freq >= self.quasi_const_threshold:
                    self.quasi_constant_cols_.append(col)

        drop_cols = set(self.constant_cols_) | set(self.quasi_constant_cols_)
        remaining = [c for c in df.columns if c not in drop_cols]

        # 3. 高欠損列の検出
        missing_rate = df[remaining].isna().sum() / len(df)
        self.high_missing_cols_ = missing_rate[
            missing_rate >= self.high_missing_threshold
        ].index.tolist()
        drop_cols |= set(self.high_missing_cols_)

        # 残す列
        self.keep_cols_ = [c for c in df.columns if c not in drop_cols]

        # 4. 欠損フラグの対象列（欠損率が中程度の列のみ）
        missing_rate_kept = df[self.keep_cols_].isna().sum() / len(df)
        self.flag_cols_ = missing_rate_kept[
            (missing_rate_kept >= self.flag_missing_low) &
            (missing_rate_kept < self.flag_missing_high)
        ].index.tolist()

        # 5. 中央値（補完用、訓練データから算出）
        self.median_values_ = df[self.keep_cols_].median()

        # サマリー
        self.fit_summary_ = {
            'constant_removed': len(self.constant_cols_),
            'quasi_constant_removed': len(self.quasi_constant_cols_),
            'high_missing_removed': len(self.high_missing_cols_),
            'kept_columns': len(self.keep_cols_),
            'missing_flags_added': len(self.flag_cols_),
        }
        return self

    def transform(self, X):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        df.columns = [str(c) for c in df.columns]

        kept = df[self.keep_cols_].copy()
        flags = kept[self.flag_cols_].isna().astype(int)
        flags.columns = [f"missing_flag_{c}" for c in self.flag_cols_]
        kept = kept.fillna(self.median_values_)

        result = pd.concat([
            kept.reset_index(drop=True),
            flags.reset_index(drop=True),
        ], axis=1)
        return result.values

    def get_feature_names_out(self, input_features=None):
        """前処理後の列名（kept 列 + 欠損フラグ列）を返す。"""
        flag_names = [f"missing_flag_{c}" for c in self.flag_cols_]
        return np.array(list(self.keep_cols_) + flag_names)
