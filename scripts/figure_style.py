"""図の共通設定。色の意味はリポジトリで 1 つに決める。

- ACCENT: 主役（現行のモデル、注目するグループ・fold）
- GRAY: 比べる相手、背景
- STATUS: 監視の判定（緑/黄/赤）。ほかの意味には使わない
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(ROOT, "reports", "figures")

ACCENT = "#2F5D9E"
GRAY = "#A6A6A6"
INK = "#222222"
MUTED = "#666666"
STATUS = {"green": "#2E8B57", "yellow": "#D9A400", "red": "#C0392B"}

JAPANESE_FONTS = [
    "Noto Sans CJK JP", "IPAexGothic", "IPAGothic", "Meiryo", "Yu Gothic",
    "Hiragino Sans", "Source Han Sans JP", "Noto Sans JP",
]


def setup():
    """日本語フォントと共通の見た目を設定する。フォントが無ければ豆腐の図を作らず止める。"""
    # 可変フォントは matplotlib では極細に固定され、太字も効かないので使わない。
    available = {f.name for f in fm.fontManager.ttflist if "-VF" not in os.path.basename(f.fname)}
    font = next((name for name in JAPANESE_FONTS if name in available), None)
    if font is None:
        raise SystemExit(f"図には日本語フォントが要る（次のいずれか: {', '.join(JAPANESE_FONTS)}）")
    print(f"図のフォント: {font}")
    plt.rcParams.update({
        "font.family": font,
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "font.size": 10,
    })


def set_title(fig, title, subtitle=None):
    """タイトルは「段階名：結論の文」。左端にそろえ、副題で読み方を補う。"""
    fig.suptitle(title, x=0.02, y=0.985, ha="left", va="top", fontsize=12.5, fontweight="bold")
    if subtitle:
        fig.text(0.02, 0.985 - 0.33 / fig.get_figheight(), subtitle, ha="left", va="top", fontsize=9.5, color=MUTED)


def save(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, facecolor="white", metadata={"Software": None})
    plt.close(fig)
    print(f"書き出し: {os.path.relpath(path, ROOT)}")
