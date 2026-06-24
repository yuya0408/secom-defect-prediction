"""時系列リプレイによるドリフト監視の実演。

SECOM の検査タイムスタンプで全データを月ごとに分割し、最初の数か月を「参照（学習）期」、
以降の月を「本番運用月」として順に推論サーバーへ流す状況を再現する。各本番月で
入力ドリフト・予測ドリフト・性能ドリフトを計測し、緑/黄/赤の深刻度を判定する。

凍結データセットの上で監視機構が「実際に動く」ことを示すデモ。本番ラインと違い、
全期間のラベルが揃っているため性能ドリフト（PR-AUC 低下）まで含めて実演できる。

使い方（リポジトリのルートで）::

    python -m scripts.replay_drift

出力:
    reports/drift_report.md          月別の指標・判定サマリー
    reports/figures/fig7_drift_replay.png   指標の時系列プロット
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.monitoring import (
    RECOMMENDED_ACTIONS,
    classify_severity,
    missing_pattern_signature,
    population_stability_index,
    unknown_group_rate,
)
from src.preprocessing import SecomPreprocessor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "secom.data")
LABELS_PATH = os.path.join(ROOT, "data", "secom_labels.data")
REPORTS_DIR = os.path.join(ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# 参照（学習）期として使う月。残りを本番運用月として順に流す。
REFERENCE_MONTHS = ["2008-07", "2008-08"]

# 品種グループ署名に使う列の欠損率レンジ（考察編の品種ミックス分析と同じ定義）。
GROUP_MISSING_LOW, GROUP_MISSING_HIGH = 0.10, 0.70


def _group_columns(ref_df: pd.DataFrame) -> list:
    """参照期で欠損率が中程度（10〜70%）の列＝品種グループ署名の構成列。"""
    rate = ref_df.isna().mean()
    cols = rate[(rate >= GROUP_MISSING_LOW) & (rate <= GROUP_MISSING_HIGH)].index
    return [str(c) for c in cols]


def load_data():
    """特徴量・ラベル・検査タイムスタンプを時系列順で読み込む。"""
    df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, na_values="NaN")
    lab = pd.read_csv(
        LABELS_PATH, sep=r'\s(?=")', header=None,
        names=["label", "dt"], engine="python",
    )
    ts = pd.to_datetime(lab["dt"].str.strip('"'), format="%d/%m/%Y %H:%M:%S")
    y = (lab["label"] == 1).astype(int).to_numpy()
    month = ts.dt.to_period("M").astype(str)
    return df.reset_index(drop=True), y, month.reset_index(drop=True)


def main():
    df, y, month = load_data()

    ref_mask = month.isin(REFERENCE_MONTHS).to_numpy()
    prod_months = [m for m in sorted(month.unique()) if m not in REFERENCE_MONTHS]
    print(f"参照期 {REFERENCE_MONTHS}: {ref_mask.sum()} 件 / 本番月 {prod_months}")

    # --- 参照期だけでモデルを学習（未来を見ない＝リーク防止） ---
    pre = SecomPreprocessor()
    X_ref = pre.fit_transform(df[ref_mask])
    y_ref = y[ref_mask]

    def make_model():
        return RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )

    # 参照期の「素の実力」と「スコア分布」は交差検証の out-of-fold で得る。
    # 学習データそのものへの予測は過学習で 0/1 に張り付き、ベースライン・PSI を歪めるため使わない。
    ref_scores = cross_val_predict(
        make_model(), X_ref, y_ref,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        method="predict_proba", n_jobs=-1,
    )[:, 1]
    baseline_pr_auc = float(average_precision_score(y_ref, ref_scores))

    # 本番月の推論には参照期全体で学習したモデルを使う。
    model = make_model().fit(X_ref, y_ref)

    group_cols = _group_columns(df[ref_mask])
    ref_signatures = missing_pattern_signature(df[ref_mask], group_cols)
    print(f"参照期 PR-AUC（交差検証・基準値）: {baseline_pr_auc:.4f} / "
          f"品種グループ署名 {ref_signatures.nunique()} 種（{len(group_cols)} 列）")

    # --- 本番月を順にリプレイ ---
    rows = []
    for m in prod_months:
        mask = (month == m).to_numpy()
        Xb = df[mask]
        yb = y[mask]
        scores = model.predict_proba(pre.transform(Xb))[:, 1]

        sigs = missing_pattern_signature(Xb, group_cols)
        unk = unknown_group_rate(ref_signatures, sigs)
        psi = population_stability_index(ref_scores, scores)
        pr_auc = float(average_precision_score(yb, scores)) if yb.sum() > 0 else float("nan")
        auc = float(roc_auc_score(yb, scores)) if 0 < yb.sum() < len(yb) else float("nan")

        a = classify_severity(psi, unk, pr_auc=pr_auc, baseline_pr_auc=baseline_pr_auc)
        rows.append({
            "month": m, "n": int(mask.sum()), "fail_rate": float(yb.mean()),
            "unknown_group_rate": unk, "prediction_psi": psi,
            "auc": auc, "pr_auc": pr_auc, "level": a.level,
            "reasons": "; ".join(a.reasons),
        })
        print(f"[{m}] n={mask.sum()} 不良率={yb.mean():.1%} "
              f"未知グループ={unk:.1%} PSI={psi:.3f} PR-AUC={pr_auc:.3f} -> {a.level.upper()}")

    report = pd.DataFrame(rows)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    _write_markdown(report, baseline_pr_auc)
    _plot(report)


def _write_markdown(report: pd.DataFrame, baseline_pr_auc: float):
    path = os.path.join(REPORTS_DIR, "drift_report.md")
    lines = [
        "# ドリフト監視リプレイ結果",
        "",
        f"参照期 {REFERENCE_MONTHS} で学習し、以降の月を順に流した結果。"
        f"基準 PR-AUC（参照期の交差検証 out-of-fold）= {baseline_pr_auc:.4f}。"
        f"品種グループ署名は欠損率 {GROUP_MISSING_LOW:.0%}〜{GROUP_MISSING_HIGH:.0%} の列で定義。",
        "",
        "| 月 | 件数 | 不良率 | 未知グループ率 | 予測PSI | AUC | PR-AUC | 判定 |",
        "|----|------|--------|----------------|---------|-----|--------|------|",
    ]
    for _, r in report.iterrows():
        lines.append(
            f"| {r['month']} | {r['n']} | {r['fail_rate']:.1%} | {r['unknown_group_rate']:.1%} "
            f"| {r['prediction_psi']:.3f} | {r['auc']:.3f} | {r['pr_auc']:.3f} "
            f"| {r['level'].upper()} |"
        )
    lines += ["", "## 判定理由とアクション", ""]
    for _, r in report.iterrows():
        lines.append(f"- **{r['month']} → {r['level'].upper()}**: {r['reasons']}")
        lines.append(f"  - 推奨アクション: {RECOMMENDED_ACTIONS[r['level']]}")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"書き出し: {os.path.relpath(path, ROOT)}")


def _setup_japanese_font() -> bool:
    """日本語フォントが使えれば matplotlib に設定する。使えなければ False。

    環境に依存しないよう、見つかった場合だけ日本語ラベルにし、無い環境では英語に
    フォールバックする（豆腐文字 □ を出さない）。
    """
    import matplotlib.font_manager as fm

    candidates = [
        "IPAexGothic", "IPAGothic", "IPAPGothic", "Noto Sans CJK JP",
        "Noto Sans JP", "TakaoGothic", "VL Gothic", "Hiragino Sans",
        "Yu Gothic", "Meiryo", "Source Han Sans JP",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


def _plot(report: pd.DataFrame):
    color = {"green": "tab:green", "yellow": "tab:orange", "red": "tab:red"}
    x = np.arange(len(report))

    # 日本語フォントがあれば日本語ラベル、無ければ英語にフォールバック。
    jp = _setup_japanese_font()
    L = {
        "unknown": "未知グループ率" if jp else "Unknown-group rate",
        "psi": "予測PSI" if jp else "Prediction PSI",
        "pr_auc": "PR-AUC",
        "ax2": "予測PSI / PR-AUC" if jp else "Prediction PSI / PR-AUC",
        "title": ("時系列リプレイによるドリフト指標（月別）" if jp
                  else "Drift metrics by month (time-series replay)"),
    }

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.bar(x, report["unknown_group_rate"], width=0.4, align="edge",
            color=[color[l] for l in report["level"]], label=L["unknown"], alpha=0.8)
    ax1.set_ylabel(L["unknown"])
    ax1.set_xticks(x + 0.2)
    ax1.set_xticklabels([f"{m}\n({l.upper()})" for m, l in zip(report["month"], report["level"])])

    ax2 = ax1.twinx()
    ax2.plot(x + 0.2, report["prediction_psi"], "o-", color="navy", label=L["psi"])
    ax2.plot(x + 0.2, report["pr_auc"], "s--", color="darkred", label=L["pr_auc"])
    ax2.set_ylabel(L["ax2"])

    ax1.set_title(L["title"])
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig7_drift_replay.png")
    fig.savefig(path, dpi=120)
    print(f"書き出し: {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
