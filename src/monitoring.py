"""ドリフト監視ユーティリティ（素の numpy/pandas 実装、外部依存なし）。

このモジュールは「学習時に見た分布」と「新しく入ってきたデータ」を比べ、
モデルが学習時の前提から乖離していないかを定量化する。SECOM の最大の発見である
「欠損パターン＝品種ミックス」をそのまま監視指標に転用している点が特徴。

提供する指標
------------
- :func:`population_stability_index` : 連続値（予測確率など）の分布ずれ（PSI）
- :func:`missing_pattern_signature` : 欠損パターンによる品種グループの署名
- :func:`unknown_group_rate`        : 学習時に存在しなかったグループの出現率
- :func:`classify_severity`         : 上記から 緑/黄/赤 の深刻度を判定

判定思想（記事の運用章と対応）
------------------------------
- 緑: 通常運用。
- 黄: 推論は続けるが警告（通知・ダッシュボード・レスポンスへのフラグ）。
- 赤: 推論は返すが信頼度を下げ、該当ロットを全数検査へ。再学習を起票（自動実行しない）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# --- 分布ドリフト（PSI） ------------------------------------------------------
def population_stability_index(
    expected: Iterable[float], actual: Iterable[float], bins: int = 10, eps: float = 1e-6
) -> float:
    """連続値の Population Stability Index を返す。

    ``expected``（学習時の分布）の分位点でビンを切り、``actual``（新データ）の
    各ビン比率と比べる。PSI の慣用的な目安: <0.1 安定 / 0.1–0.25 要注意 / >0.25 大きな変化。
    """
    expected = np.asarray(list(expected), dtype=float)
    actual = np.asarray(list(actual), dtype=float)
    if expected.size == 0 or actual.size == 0:
        return 0.0

    edges = np.quantile(expected, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)  # 同値による幅ゼロのビンを除去
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    e_ratio = np.histogram(expected, bins=edges)[0] / expected.size
    a_ratio = np.histogram(actual, bins=edges)[0] / actual.size
    e_ratio = np.clip(e_ratio, eps, None)
    a_ratio = np.clip(a_ratio, eps, None)
    return float(np.sum((a_ratio - e_ratio) * np.log(a_ratio / e_ratio)))


# --- 品種グループ（欠損パターン） --------------------------------------------
def missing_pattern_signature(df: pd.DataFrame, flag_cols: Iterable[str]) -> pd.Series:
    """欠損パターンからサンプルごとのグループ署名（"010..." 形式）を作る。

    ``flag_cols`` は前処理が欠損フラグ化した列（中程度の欠損列）。その列が欠損か否かの
    パターンが、どの品種がどの工程を通った（バイパスした）かの代理になる、という仮説に基づく。
    """
    cols = [str(c) for c in flag_cols]
    work = df.copy()
    work.columns = [str(c) for c in work.columns]
    miss = work[cols].isna().astype(int)
    return miss.apply(lambda row: "".join(map(str, row.values)), axis=1)


def unknown_group_rate(
    reference_signatures: Iterable[str], new_signatures: Iterable[str]
) -> float:
    """学習時に存在しなかったグループ署名を持つ新データの割合。"""
    known = set(reference_signatures)
    new = list(new_signatures)
    if not new:
        return 0.0
    unseen = sum(1 for s in new if s not in known)
    return unseen / len(new)


# --- 深刻度判定 ---------------------------------------------------------------
@dataclass
class DriftThresholds:
    """緑/黄/赤を分ける境界値。慣用値＋SECOM の事情を踏まえた既定。"""

    psi_warn: float = 0.10
    psi_alert: float = 0.25
    unknown_warn: float = 0.05
    unknown_alert: float = 0.15


@dataclass
class DriftAssessment:
    level: str                       # "green" | "yellow" | "red"
    reasons: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def classify_severity(
    prediction_psi: float,
    unknown_rate: float,
    pr_auc: Optional[float] = None,
    baseline_pr_auc: Optional[float] = None,
    thresholds: DriftThresholds = DriftThresholds(),
) -> DriftAssessment:
    """各指標から最も深刻なレベルを採用して総合判定する。

    性能指標（``pr_auc``）はラベル確定後にのみ渡せる。ベースラインを下回ったら赤に倒す。
    """
    level = "green"
    reasons: list = []

    def bump(new_level: str, reason: str):
        nonlocal level
        order = {"green": 0, "yellow": 1, "red": 2}
        if order[new_level] > order[level]:
            level = new_level
        reasons.append(reason)

    if prediction_psi >= thresholds.psi_alert:
        bump("red", f"予測分布のPSIが大きい（{prediction_psi:.3f} ≥ {thresholds.psi_alert}）")
    elif prediction_psi >= thresholds.psi_warn:
        bump("yellow", f"予測分布のPSIがやや高い（{prediction_psi:.3f} ≥ {thresholds.psi_warn}）")

    if unknown_rate >= thresholds.unknown_alert:
        bump("red", f"未知グループ出現率が高い（{unknown_rate:.1%} ≥ {thresholds.unknown_alert:.0%}）")
    elif unknown_rate >= thresholds.unknown_warn:
        bump("yellow", f"未知グループが出現（{unknown_rate:.1%} ≥ {thresholds.unknown_warn:.0%}）")

    if pr_auc is not None and baseline_pr_auc is not None and pr_auc < baseline_pr_auc:
        bump("red", f"PR-AUCがベースライン割れ（{pr_auc:.3f} < {baseline_pr_auc:.3f}）")

    metrics = {
        "prediction_psi": prediction_psi,
        "unknown_group_rate": unknown_rate,
        "pr_auc": pr_auc,
        "baseline_pr_auc": baseline_pr_auc,
    }
    return DriftAssessment(level=level, reasons=reasons or ["全指標が安定範囲"], metrics=metrics)


# 検知レベル -> 推奨アクション（記事の運用章と対応。コードは判定まで、実行は人間）。
RECOMMENDED_ACTIONS = {
    "green": "通常運用。指標を記録するのみ。",
    "yellow": "推論は継続しつつ通知・ダッシュボード警告。レスポンスに drift_warning を付与。",
    "red": "推論は返すが信頼度を下げ、該当ロットを全数検査へ。再学習を起票（自動実行しない）。",
}
