"""SECOM 不良予測の Gradio デモ（Hugging Face Spaces 用）。

学習済みの ``models/*.pkl`` を読み込み、ブラウザ上から不良確率を試せる UI を提供する。
推論ロジックは推論 API（``src/api.py``）と同じ前処理クラス・モデルを共有しており、
デモと本番サービングで結果が一致する。

ローカル起動::

    python app.py            # http://localhost:7860

Hugging Face Spaces（SDK: gradio）では、このファイルが ``app_file`` として
自動的に起動される。``src/preprocessing.py`` を import できることで、
``joblib.load`` が前処理クラスを復元できる点は API と同じ設計。
"""

from __future__ import annotations

import json
import math
import os
from typing import List, Optional

import gradio as gr
import joblib
import pandas as pd

# joblib.load が pickle 内の SecomPreprocessor を復元するために必要。
from src.preprocessing import SecomPreprocessor  # noqa: F401

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(ROOT, "models"))
DATA_PATH = os.path.join(ROOT, "data", "secom.data")
LABELS_PATH = os.path.join(ROOT, "data", "secom_labels.data")

preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))
model = joblib.load(os.path.join(MODELS_DIR, "model.pkl"))
metadata = joblib.load(os.path.join(MODELS_DIR, "metadata.pkl"))

N_FEATURES = metadata["input_features"]
DEFAULT_THRESHOLD = metadata.get("default_threshold", 0.5)
MODEL_VERSION = metadata.get("version", "unknown")
TRAINED_AT = metadata.get("trained_at", "unknown")


# --- 推論 --------------------------------------------------------------------
def predict_features(features: List[Optional[float]], threshold: float) -> dict:
    """生のセンサー値（590 個）-> 推論結果の辞書。"""
    if len(features) != N_FEATURES:
        raise ValueError(
            f"features は {N_FEATURES} 要素である必要があります（受信: {len(features)} 要素）。"
        )
    X = preprocessor.transform(pd.DataFrame([features]))
    proba = float(model.predict_proba(X)[:, 1][0])
    return {
        "defect_probability": proba,
        "is_defect": bool(proba >= threshold),
        "threshold": threshold,
        "model_version": MODEL_VERSION,
    }


# --- データからの例（data/ が同梱されている場合のみ） -------------------------
def _data_available() -> bool:
    return os.path.exists(DATA_PATH) and os.path.exists(LABELS_PATH)


def _load_row(idx: int) -> List[Optional[float]]:
    df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, na_values="NaN")
    return [
        None if (isinstance(v, float) and math.isnan(v)) else float(v)
        for v in df.iloc[idx].tolist()
    ]


def _load_labels():
    lab = pd.read_csv(
        LABELS_PATH, sep=r'\s(?=")', header=None,
        names=["label", "dt"], engine="python",
    )
    return (lab["label"] == 1).astype(int).tolist()


def predict_sample(idx: int, threshold: float):
    if not _data_available():
        return {"error": "data/ が同梱されていないため、サンプル推論は利用できません。"
                "JSON アップロードのタブを使ってください。"}
    idx = int(idx)
    labels = _load_labels()
    result = predict_features(_load_row(idx), threshold)
    result["true_label"] = "不良(1)" if labels[idx] == 1 else "合格(0)"
    return result


def predict_json(file_obj, threshold: float):
    if file_obj is None:
        return {"error": "JSON ファイルをアップロードしてください。"}
    with open(file_obj.name, encoding="utf-8") as f:
        payload = json.load(f)
    features = payload.get("features", payload)  # {"features":[...]} か素の配列
    return predict_features(features, threshold)


# --- UI ----------------------------------------------------------------------
DESCRIPTION = f"""\
# SECOM 不良予測デモ

半導体製造ラインのセンサー値（**{N_FEATURES} 次元**）から、その製品が不良である確率を返す。
推論 API（FastAPI）と同じ前処理・モデルを共有している。

- モデル: **{metadata.get('model_type', '?')}** / version **{MODEL_VERSION}**（学習日時 {TRAINED_AT}）
- 厳密評価（Nested CV・リーク排除）: **AUC {metadata.get('evaluation_AUC_strict', '?')}** /
  PR-AUC {metadata.get('evaluation_PR_AUC_strict', '?')}

> 精度を競うものではなく、リークを排除した正直な評価と、再現可能なサービングを示すデモ。
> AUC は 0.6 台で実力には限界がある。事後検知の補助としての位置づけ。
"""

with gr.Blocks(title="SECOM 不良予測デモ") as demo:
    gr.Markdown(DESCRIPTION)
    threshold = gr.Slider(
        0.0, 1.0, value=DEFAULT_THRESHOLD, step=0.01,
        label="判定しきい値（これ以上の確率を不良と判定）",
    )

    with gr.Tab("データから例を選んで推論"):
        gr.Markdown(
            "同梱データ（1,567 件）から行番号を選んで推論する。"
            "`true_label` が実際の検査結果。"
        )
        idx = gr.Number(value=0, precision=0, label="行番号 (0〜1566)")
        btn_sample = gr.Button("推論", variant="primary")
        out_sample = gr.JSON(label="推論結果")
        btn_sample.click(predict_sample, [idx, threshold], out_sample)

    with gr.Tab("JSON をアップロードして推論"):
        gr.Markdown(
            "`sample_input.json` の形式（`{\"features\": [590個の値]}`、"
            "欠損は `null`）でアップロードする。"
        )
        file_in = gr.File(label="JSON ファイル", file_types=[".json"])
        btn_json = gr.Button("推論", variant="primary")
        out_json = gr.JSON(label="推論結果")
        btn_json.click(predict_json, [file_in, threshold], out_json)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
