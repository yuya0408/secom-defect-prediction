"""時系列リプレイによるドリフト監視の実演。

SECOM の検査タイムスタンプで全データを月ごとに分割し、最初の数か月を「参照（学習）期」、
以降の月を「リプレイ月」として順に流す。各リプレイ月で入力ドリフト・予測ドリフト・
性能ドリフトを計測し、緑/黄/赤の深刻度を判定する。

凍結データの上で監視の計算と判定が動くことを示す実演で、推論 API には組み込んでいない。
全期間のラベルが揃っているため性能ドリフト（PR-AUC 低下）まで含めて実演できる。

使い方（リポジトリのルートで）::

    python -m scripts.replay_drift

出力:
    reports/drift_report.md          月別の指標・判定サマリー
    reports/figures/fig7_drift_replay.png   指標の時系列プロット
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from scripts import figure_style as fs
from src.monitoring import (
    RECOMMENDED_ACTIONS,
    DriftThresholds,
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

# 参照（学習）期として使う月。残りをリプレイ月として順に流す。
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
    print(f"参照期 {REFERENCE_MONTHS}: {ref_mask.sum()} 件 / リプレイ月 {prod_months}")

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
    # 分割は時間順ではなく不良で層別する。参照期（7〜8月）の不良は 65 件と少なく、層別にすると
    # 各 fold の不良が 13 件にそろう（時間順に切ると 18/21/13/6/7 件と偏る）。
    # ただし未来の行でも学習するため、時間順の評価より楽観的な基準になりうる。
    ref_scores = cross_val_predict(
        make_model(), X_ref, y_ref,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        method="predict_proba", n_jobs=-1,
    )[:, 1]
    baseline_pr_auc = float(average_precision_score(y_ref, ref_scores))

    # リプレイ月の推論には参照期全体で学習したモデルを使う。
    model = make_model().fit(X_ref, y_ref)

    group_cols = _group_columns(df[ref_mask])
    ref_signatures = missing_pattern_signature(df[ref_mask], group_cols)
    print(f"参照期 PR-AUC（交差検証・基準値）: {baseline_pr_auc:.4f} / "
          f"品種グループ署名 {ref_signatures.nunique()} 種（{len(group_cols)} 列）")

    # --- リプレイ月を順に流す ---
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
    _write_markdown(report, baseline_pr_auc)
    _plot(report, baseline_pr_auc, float(y_ref.mean()))


def _write_markdown(report: pd.DataFrame, baseline_pr_auc: float):
    path = os.path.join(REPORTS_DIR, "drift_report.md")
    levels = ", ".join(f"{r['month']} は {r['level'].upper()}" for _, r in report.iterrows())
    lines = [
        "# ドリフト監視リプレイ結果",
        "",
        f"**結論: {levels}。凍結データ上の実演で、API には組み込んでいない。**",
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
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    print(f"書き出し: {os.path.relpath(path, ROOT)}")


def _plot(report: pd.DataFrame, baseline_pr_auc: float, reference_fail_rate: float):
    """指標ごとに軸を分け、判定の線と指標ごとの判定（印の色と文字）、月の判定（目盛り）を描く。"""
    fs.setup()
    th = DriftThresholds()
    panels = [
        ("未知グループ率（%）。高いほど悪い", report["unknown_group_rate"] * 100, [th.unknown_warn * 100, th.unknown_alert * 100], "high"),
        ("予測 PSI。高いほど悪い", report["prediction_psi"], [th.psi_warn, th.psi_alert], "high"),
        ("PR-AUC。低いほど悪い", report["pr_auc"], [baseline_pr_auc], "low"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.6))
    fig.subplots_adjust(left=0.05, right=0.985, top=0.74, bottom=0.17, wspace=0.2)
    x = np.arange(len(report))
    for ax, (label, values, lines, side) in zip(axes, panels):
        top = max(values.max(), max(lines)) * 1.3
        for line in lines:
            ax.axhline(line, color=fs.MUTED, ls="--", lw=1)
        if side == "high":
            ax.axhspan(lines[-1], top, color=fs.STATUS["red"], alpha=0.06, lw=0)
            if len(lines) == 2:
                ax.axhspan(lines[0], lines[1], color=fs.STATUS["yellow"], alpha=0.07, lw=0)
        else:
            ax.axhspan(0, lines[0], color=fs.STATUS["red"], alpha=0.06, lw=0)
        ax.plot(x, values, color=fs.GRAY, lw=1.4, zorder=2)
        rising = values.iloc[-1] > values.iloc[0]
        for i, (xi, v) in enumerate(zip(x, values)):
            level = _metric_level(v, lines, side)
            ax.plot(xi, v, "o", ms=11, color=fs.STATUS[level], zorder=3)
            # 折れ線と重ならない側に置く（上りなら左は下・右は上、下りはその逆）。
            above = (i > 0) == rising
            value_text = f"{v:.1f}" if "%" in label else f"{v:.3f}"
            ax.annotate(f"{value_text}（{level.upper()}）", (xi, v), xytext=(0, 12 if above else -12),
                        textcoords="offset points", ha="center", va="bottom" if above else "top", fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=4)
        for line in lines:
            text = f"{line:g}" if side == "high" else f"基準 {line:.3f}"
            ax.text(1.0, line, text, transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=9, color=fs.MUTED)
        ax.set_title(label, loc="left", fontsize=10)
        if side == "low":
            ticks = [f"{m}\n不良率 {r:.1%}" for m, r in zip(report["month"], report["fail_rate"])]
        else:
            ticks = list(report["month"])
        ax.set_xticks(x, [f"{t}\n月の判定 {l.upper()}" for t, l in zip(ticks, report["level"])])
        for tick, level in zip(ax.get_xticklabels(), report["level"]):
            tick.set_color(fs.STATUS[level])
            tick.set_fontweight("bold")
        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.set_ylim(0, top)
    title = "監視の実演："
    title += "2 か月とも RED" if (report["level"] == "red").all() else "月ごとの判定は目盛りのとおり"
    if report["pr_auc"].iloc[0] >= baseline_pr_auc and report["level"].iloc[0] == "red":
        title += "。9 月は PR-AUC が基準より上でも PSI で RED"
    fs.set_title(
        fig, title,
        "点の色と文字は指標ごとの判定で、月の判定は最も深刻なものを取る。破線が判定の線（未知グループ率の線は仮置き）、薄い網掛けが悪い側。\n"
        f"PR-AUC の基準は不良率 {reference_fail_rate:.1%} の参照期で測った値で、不良率の違う月とそのまま比べている。",
    )
    fs.save(fig, "fig7_drift_replay.png")


def _metric_level(value: float, lines: list, side: str) -> str:
    """classify_severity と同じ向きで、1 つの指標だけの判定を返す（図の印に使う）。"""
    if side == "low":
        return "red" if value < lines[0] else "green"
    if value >= lines[1]:
        return "red"
    return "yellow" if value >= lines[0] else "green"


if __name__ == "__main__":
    main()
