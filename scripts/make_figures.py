"""README の図（fig1〜fig6）を data/ から作り直す。fig7 は scripts/replay_drift.py が作る。

使い方（リポジトリのルートで）::

    python -m scripts.make_figures

出力:
    reports/figures/fig1_group_defect_rate.png   グループ別の不良率
    reports/figures/fig2_monthly_defect_rate.png 月別の不良率
    reports/figures/fig3_correlation.png         月別不良率と高不良グループの割合
    reports/figures/fig4_auc_comparison.png      品種情報を入れる方法と現行の AUC の差
    reports/figures/fig5_fold_new_group.png      fold ごとの新しいグループの割合
    reports/figures/fig6_timeseries_split.png    時系列CVの分割
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from scripts import compare_group_info, figure_style as fs
from scripts.compare_group_info import BASELINE, missing_pattern_groups
from src.train import N_SPLITS, load_data

# ノートブックと同じ定義: 10 件以上のグループのうち、不良率が 10% を超えるもの。
MIN_GROUP_SAMPLES = 10
HIGH_RATE = 0.10
Z_95 = 1.96
ALPHA = 0.05


def wilson(k, n):
    """二項割合の Wilson 95% 区間。少数のグループでも範囲が 0〜1 に収まる。"""
    p = k / n
    d = 1 + Z_95**2 / n
    center = (p + Z_95**2 / (2 * n)) / d
    half = Z_95 * np.sqrt(p * (1 - p) / n + Z_95**2 / (4 * n**2)) / d
    return center - half, center + half


def group_table(groups, y):
    rows = []
    for g in np.unique(groups):
        in_g = groups == g
        if in_g.sum() >= MIN_GROUP_SAMPLES:
            rows.append({"group": g, "n": int(in_g.sum()), "fail": int(y[in_g].sum()), "rate": y[in_g].mean()})
    return pd.DataFrame(rows)


def monthly_table(groups, y, ts, high_groups):
    d = pd.DataFrame({"month": pd.Series(ts).dt.month, "y": y, "high": np.isin(groups, high_groups)})
    return d.groupby("month").agg(n=("y", "count"), fail=("y", "sum"), rate=("y", "mean"), high=("high", "mean"))


def fig1(table, overall, n_total, fail_total):
    t = table.sort_values(["rate", "n"]).reset_index(drop=True)
    n_high = int((t.rate > HIGH_RATE).sum())
    fig, ax = plt.subplots(figsize=(8.2, 8.8))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.885, bottom=0.065)
    for i, r in t.iterrows():
        low, high = wilson(r.fail, r.n)
        color = fs.ACCENT if r.rate > HIGH_RATE else fs.GRAY
        ax.plot([low * 100, high * 100], [i, i], color=color, lw=1.4, solid_capstyle="round")
        # 青のうち区間が全体の線をまたぐものは、全体より高いと言えないので白抜きにする。
        hollow = r.rate > HIGH_RATE and low <= overall
        ax.plot(r.rate * 100, i, "o", ms=6, color=color, mfc="white" if hollow else color, clip_on=False, zorder=3)
    ax.axvline(overall * 100, color=fs.MUTED, ls="--", lw=1)
    ax.text(overall * 100 + 0.4, len(t) - 0.2, f"全体 {overall:.2%}", fontsize=9.5, color=fs.MUTED, va="bottom")
    ax.set_yticks(range(len(t)), [f"G{r.group}（{r.n} 件）" for r in t.itertuples()], fontsize=9)
    ax.set_ylim(-0.7, len(t) + 0.3)
    ax.set_xlim(0, None)
    ax.set_xlabel("不良率（%）")
    ax.grid(axis="x", color="#E6E6E6", lw=0.7)
    ax.set_axisbelow(True)
    fs.set_title(
        fig, f"品種の分析（参考）：不良率は 0〜{t.rate.max():.0%} に分かれるが、区間は広い",
        f"件数 {MIN_GROUP_SAMPLES} 以上の {len(t)} グループ（{t.n.sum():,}/{n_total:,} 件、不良 {t.fail.sum()}/{fail_total}）。"
        f"線は Wilson 95% 区間（多重比較の補正なし）。\n"
        f"青は不良率が {HIGH_RATE:.0%} を超える {n_high} グループ。白抜きは区間が全体の線をまたぐ。",
    )
    fs.save(fig, "fig1_group_defect_rate.png")


def fig2(months):
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.8, bottom=0.11)
    x = np.arange(len(months))
    low, high = zip(*(wilson(r.fail, r.n) for r in months.itertuples()))
    ax.vlines(x, np.array(low) * 100, np.array(high) * 100, color=fs.GRAY, lw=1.6)
    ax.plot(x, months.rate * 100, "-o", color=fs.INK, lw=1.8, ms=7)
    for xi, r in zip(x, months.itertuples()):
        ax.annotate(f"{r.rate:.1%}", (xi, r.rate * 100), xytext=(10, 10), textcoords="offset points")
    ax.set_xticks(x, [f"{m} 月（{r.n} 件）" for m, r in zip(months.index, months.itertuples())])
    ax.set_xlim(-0.3, len(months) - 0.5)
    ax.set_ylabel("不良率（%）")
    ax.set_ylim(0, None)
    ax.grid(axis="y", color="#E6E6E6", lw=0.7)
    ax.set_axisbelow(True)
    fs.set_title(
        fig, f"品種の分析（参考）：月別の不良率は {months.rate.iloc[0]:.0%} から {months.rate.iloc[2]:.0%} まで動く",
        "灰の縦線は Wilson 95% 区間。7 月は 7/19 から、10 月は 10/17 までのデータ。",
    )
    fs.save(fig, "fig2_monthly_defect_rate.png")


def fig3(months, n_high):
    x, y = (months.high * 100).round(2).to_numpy(), (months.rate * 100).round(2).to_numpy()
    r, p = pearsonr(x, y)
    fig, ax = plt.subplots(figsize=(7.4, 5))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.8, bottom=0.12)
    ax.plot(x, y, "o", ms=10, color=fs.INK, clip_on=False)
    for xi, yi, m in zip(x, y, months.index):
        ax.annotate(f"{m} 月", (xi, yi), xytext=(9, 3), textcoords="offset points")
    ax.set_xlim(0, x.max() + 7)
    ax.set_ylim(0, y.max() + 2.5)
    ax.set_xlabel(f"高不良グループ（{MIN_GROUP_SAMPLES} 件以上で不良率 {HIGH_RATE:.0%} 超の {n_high} つ）の割合（%）")
    ax.set_ylabel("月別の不良率（%）")
    ax.grid(color="#E6E6E6", lw=0.7)
    ax.set_axisbelow(True)
    fs.set_title(
        fig, f"品種の分析（参考。判定には使わない）：r = {r:.2f} だが n = {len(x)} で有意ではない",
        f"p = {p:.2f}（α = {ALPHA}）。高不良グループは全期間の不良率で定義したので、両軸は同じラベルに依存する。",
    )
    fs.save(fig, "fig3_correlation.png")


def fig4():
    proba, y_oof, fold, tests, p_holm = compare_group_info.evaluate()
    names = sorted(tests, key=lambda n: tests[n]["diff"])
    baseline_auc = np.mean([roc_auc_score(y_oof[fold == k], proba[BASELINE][fold == k]) for k in np.unique(fold)])
    z = norm.ppf(1 - ALPHA / (2 * len(tests)))  # Bonferroni の同時区間。判定（Holm）と食い違いにくい
    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    fig.subplots_adjust(left=0.17, right=0.75, top=0.78, bottom=0.14)
    for i, name in enumerate(names):
        t = tests[name]
        significant = p_holm[name] < ALPHA
        ax.plot([t["diff"] - z * t["se"], t["diff"] + z * t["se"]], [i, i], color=fs.MUTED, lw=1.6, solid_capstyle="round")
        ax.plot(t["diff"], i, "o", ms=9, mec=fs.MUTED, mew=1.6, mfc="white" if significant else fs.GRAY)
        if not significant:
            verdict = "見分けられない"
        else:
            verdict = "現行より悪い" if t["diff"] < 0 else "現行より良い"
        ax.text(1.02, i, f"{verdict}（p={p_holm[name]:.3f}）", transform=ax.get_yaxis_transform(), va="center", fontsize=10)
    ax.axvline(0, color=fs.ACCENT, lw=1.4)
    ax.text(0, len(names) - 0.42, f"現行 AUC {baseline_auc:.4f}", color=fs.ACCENT, ha="center", va="bottom", fontsize=9.5)
    ax.axvspan(-0.17, 0, color=fs.GRAY, alpha=0.08, lw=0)
    ax.set_xlim(-0.17, 0.05)
    ax.set_yticks(range(len(names)), names)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlabel("現行との AUC の差（← 現行より悪い）")
    ax.grid(axis="x", color="#E6E6E6", lw=0.7)
    ax.set_axisbelow(True)
    fs.set_title(
        fig, "品種情報：どの入れ方でも AUC の上積みは見分けられない",
        "線は Bonferroni 補正の 95% 同時区間、右の p は Holm 補正後。\n"
        "灰の塗りは見分けられない手法、白抜きは差を検出した手法。網掛けは現行より悪い側。",
    )
    fs.save(fig, "fig4_auc_comparison.png")


def fig5(groups):
    splits = list(TimeSeriesSplit(N_SPLITS).split(groups))
    new = [int((~np.isin(groups[te], groups[tr])).sum()) for tr, te in splits]
    totals = [len(te) for _, te in splits]
    rates = np.array(new) / np.array(totals)
    top = int(np.argmax(rates))
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.82, bottom=0.09)
    ax.bar(range(1, N_SPLITS + 1), rates * 100, width=0.55,
           color=[fs.ACCENT if i == top else fs.GRAY for i in range(N_SPLITS)])
    for i, (r, k, n) in enumerate(zip(rates, new, totals)):
        ax.text(i + 1, r * 100 + 0.7, f"{r:.1%}\n{k}/{n} 件", ha="center", fontsize=10)
    ax.set_xticks(range(1, N_SPLITS + 1), [f"fold {i}" for i in range(1, N_SPLITS + 1)])
    ax.set_ylabel("新しいグループの割合（%）")
    ax.set_ylim(0, rates.max() * 100 + 7)
    ax.grid(axis="y", color="#E6E6E6", lw=0.7)
    ax.set_axisbelow(True)
    fs.set_title(
        fig, f"品種の分析（参考）：fold {top + 1} では検証データの {rates[top]:.0%} が学習時に無いグループ",
        "検証データのうち、学習データに無い欠損パターンのサンプルの割合。",
    )
    fs.save(fig, "fig5_fold_new_group.png")


def fig6(n_samples):
    splits = list(TimeSeriesSplit(N_SPLITS).split(np.zeros(n_samples)))
    fig, ax = plt.subplots(figsize=(8.2, 3.7))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.76, bottom=0.14)
    for row, (tr, te) in enumerate(splits):
        y = N_SPLITS - row
        ax.barh(y, len(tr), left=0, height=0.62, color=fs.ACCENT)
        ax.barh(y, len(te), left=len(tr), height=0.62, color=fs.GRAY, edgecolor=fs.INK, lw=0.8)
        ax.text(len(tr) / 2, y, f"学習 {len(tr):,} 件", ha="center", va="center", color="white", fontsize=9)
        ax.text(len(tr) + len(te) / 2, y, f"検証 {len(te)} 件", ha="center", va="center", fontsize=9)
    ax.set_yticks(range(1, N_SPLITS + 1), [f"fold {N_SPLITS - i}" for i in range(N_SPLITS)])
    ax.set_xlim(0, n_samples)
    ax.set_xlabel(f"検査の時刻順に並べた行（全 {n_samples:,} 件。古い → 新しい）")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    fs.set_title(
        fig, "評価の分割：常に過去で学習し、その直後の期間で検証する",
        "前処理・特徴量選択・モデルは学習側だけで fit する。最初の 1/6 は検証に使われない。",
    )
    fs.save(fig, "fig6_timeseries_split.png")


def main():
    fs.setup()
    df, y, ts = load_data()
    groups, _ = missing_pattern_groups(df)
    table = group_table(groups, y)
    high_groups = table[table.rate > HIGH_RATE].group.tolist()
    months = monthly_table(groups, y, ts, high_groups)
    fig1(table, y.mean(), len(df), int(y.sum()))
    fig2(months)
    fig3(months, len(high_groups))
    fig4()
    fig5(groups)
    fig6(len(df))


if __name__ == "__main__":
    main()
