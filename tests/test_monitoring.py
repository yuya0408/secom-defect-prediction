"""ドリフト監視ユーティリティのテスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.monitoring import (
    classify_severity,
    missing_pattern_signature,
    population_stability_index,
    unknown_group_rate,
)


def test_psi_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    assert population_stability_index(x, x) < 1e-6


def test_psi_grows_with_shift():
    rng = np.random.default_rng(0)
    base = rng.normal(size=2000)
    small = population_stability_index(base, base + 0.3)
    large = population_stability_index(base, base + 1.5)
    assert 0 < small < large


def test_missing_pattern_signature():
    df = pd.DataFrame({"0": [1.0, np.nan], "1": [np.nan, np.nan], "2": [1.0, 1.0]})
    sig = missing_pattern_signature(df, ["0", "1"])
    # 列 "0","1" の欠損: 行0 -> (0,1)="01", 行1 -> (1,1)="11"
    assert sig.tolist() == ["01", "11"]


def test_unknown_group_rate():
    ref = ["00", "01", "10"]
    assert unknown_group_rate(ref, ["00", "01"]) == 0.0
    assert unknown_group_rate(ref, ["00", "11"]) == 0.5  # "11" は未知
    assert unknown_group_rate(ref, []) == 0.0


def test_classify_severity_levels():
    assert classify_severity(0.05, 0.0).level == "green"
    assert classify_severity(0.15, 0.0).level == "yellow"   # PSI が警告域
    assert classify_severity(0.30, 0.0).level == "red"      # PSI が警告域超
    assert classify_severity(0.0, 0.20).level == "red"      # 未知グループが警告域超


def test_classify_severity_performance_trigger():
    # 分布は安定でも、PR-AUC がベースライン割れなら赤に倒す。
    a = classify_severity(0.0, 0.0, pr_auc=0.10, baseline_pr_auc=0.18)
    assert a.level == "red"
    assert any("PR-AUC" in r for r in a.reasons)
