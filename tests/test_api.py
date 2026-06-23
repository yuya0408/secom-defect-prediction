"""推論 API のテスト。

実データの行（不良 1 件・合格 1 件）を入力として、API が妥当な確率を返すこと、
入力バリデーションが働くことを確認する。

実行（リポジトリのルートで）::

    pytest -q
"""

from __future__ import annotations

import math
import os

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import N_FEATURES, app

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
client = TestClient(app)


def _row(idx: int):
    df = pd.read_csv(
        os.path.join(ROOT, "data", "secom.data"),
        sep=r"\s+", header=None, na_values="NaN",
    )
    return [
        None if (isinstance(v, float) and math.isnan(v)) else float(v)
        for v in df.iloc[idx].tolist()
    ]


def _labels():
    lab = pd.read_csv(
        os.path.join(ROOT, "data", "secom_labels.data"),
        sep=r'\s(?=")', header=None, names=["label", "dt"], engine="python",
    )
    return (lab["label"] == 1).astype(int).values


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_metadata():
    r = client.get("/metadata")
    assert r.status_code == 200
    body = r.json()
    assert body["input_features"] == N_FEATURES
    # バージョニング: 学習スクリプトが版と学習日時を記録していること。
    assert body.get("version")
    assert body.get("trained_at")


def test_prediction_carries_model_version():
    """予測レスポンスにモデル版が刻まれ、ログから出自を辿れること。"""
    y = _labels()
    fail_idx = int(next(i for i, v in enumerate(y) if v == 1))
    r = client.post("/predict", json={"features": _row(fail_idx)})
    assert r.status_code == 200
    assert r.json()["model_version"]


def test_predict_returns_valid_probability():
    y = _labels()
    fail_idx = int(next(i for i, v in enumerate(y) if v == 1))
    r = client.post("/predict", json={"features": _row(fail_idx)})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["defect_probability"] <= 1.0
    assert isinstance(body["is_defect"], bool)


def test_pass_row_scores_lower_than_fail_row():
    """既知の不良行は、既知の合格行より高い不良確率になるはず。"""
    y = _labels()
    fail_idx = int(next(i for i, v in enumerate(y) if v == 1))
    pass_idx = int(next(i for i, v in enumerate(y) if v == 0))
    p_fail = client.post("/predict", json={"features": _row(fail_idx)}).json()["defect_probability"]
    p_pass = client.post("/predict", json={"features": _row(pass_idx)}).json()["defect_probability"]
    assert p_fail > p_pass


def test_threshold_override():
    y = _labels()
    fail_idx = int(next(i for i, v in enumerate(y) if v == 1))
    r = client.post("/predict", json={"features": _row(fail_idx), "threshold": 0.99})
    assert r.json()["threshold"] == 0.99


def test_wrong_length_is_rejected():
    r = client.post("/predict", json={"features": [1.0, 2.0, 3.0]})
    assert r.status_code == 422


def test_batch():
    y = _labels()
    fail_idx = int(next(i for i, v in enumerate(y) if v == 1))
    pass_idx = int(next(i for i, v in enumerate(y) if v == 0))
    r = client.post("/predict/batch", json={"records": [_row(fail_idx), _row(pass_idx)]})
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 2
