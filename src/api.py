"""SECOM 不良予測の推論 API（FastAPI）。

学習済みの ``preprocessor.pkl`` / ``model.pkl`` / ``metadata.pkl`` を読み込み、
50… ではなく 590 個のセンサー値を受け取って「不良である確率」を返す。

エンドポイント
--------------
- ``GET  /``            : API の概要
- ``GET  /health``      : ヘルスチェック（モデル読み込み状態）
- ``GET  /metadata``    : モデルのメタデータ
- ``POST /predict``     : 1 件の推論
- ``POST /predict/batch``: 複数件の推論

起動（リポジトリのルートで）::

    uvicorn src.api:app --reload

起動後、http://localhost:8000/docs の Swagger UI から動作確認できる。
"""

from __future__ import annotations

import os
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

# src.preprocessing を import しておくことが重要。
# これにより joblib.load が pickle 内の SecomPreprocessor を復元できる。
from src.preprocessing import SecomPreprocessor  # noqa: F401  (復元のため必要)

# --- モデルの読み込み ---------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(ROOT, "models"))

preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))
model = joblib.load(os.path.join(MODELS_DIR, "model.pkl"))
metadata = joblib.load(os.path.join(MODELS_DIR, "metadata.pkl"))

N_FEATURES = metadata["input_features"]            # 590
DEFAULT_THRESHOLD = metadata.get("default_threshold", 0.5)
# 古い metadata.pkl（version を持たない）でも壊れないよう get で取得する。
MODEL_VERSION = metadata.get("version", "unknown")
TRAINED_AT = metadata.get("trained_at")

app = FastAPI(
    title="SECOM 不良予測 API",
    description=(
        "半導体製造ラインのセンサー値（590 次元）から、その製品が不良である確率を返す "
        "推論サービス。検査工程での自動振り分けや日次・週次の品質モニタリングへの "
        "組み込みを想定した、事後検知型の API。"
    ),
    version="1.0.0",
)


# --- 入出力スキーマ -----------------------------------------------------------
class PredictRequest(BaseModel):
    """1 件分の推論リクエスト。"""

    features: List[Optional[float]] = Field(
        ...,
        description=(
            f"{N_FEATURES} 個のセンサー値。欠損しているセンサーは null を指定できる "
            "（API 側で学習済みの中央値補完・欠損フラグ化を行う）。"
        ),
    )
    threshold: Optional[float] = Field(
        None,
        ge=0.0, le=1.0,
        description="不良と判定する確率のしきい値。省略時はモデルの既定値を使用。",
    )

    @field_validator("features")
    @classmethod
    def _check_length(cls, v):
        if len(v) != N_FEATURES:
            raise ValueError(
                f"features は {N_FEATURES} 要素である必要があります（受信: {len(v)} 要素）。"
            )
        return v


class BatchPredictRequest(BaseModel):
    """複数件分の推論リクエスト。"""

    records: List[List[Optional[float]]] = Field(
        ..., description=f"各要素が {N_FEATURES} 個のセンサー値からなる配列。"
    )
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)

    @field_validator("records")
    @classmethod
    def _check_shape(cls, v):
        if not v:
            raise ValueError("records が空です。")
        for i, row in enumerate(v):
            if len(row) != N_FEATURES:
                raise ValueError(
                    f"records[{i}] は {N_FEATURES} 要素である必要があります（受信: {len(row)} 要素）。"
                )
        return v


class PredictResponse(BaseModel):
    # "model_" で始まるフィールドは pydantic v2 の保護名前空間と衝突するため明示的に解除。
    model_config = ConfigDict(protected_namespaces=())

    defect_probability: float = Field(..., description="不良である確率（0〜1）。")
    is_defect: bool = Field(..., description="しきい値を超えたかどうか。")
    threshold: float = Field(..., description="判定に使ったしきい値。")
    model_version: str = Field(
        ..., description="この予測を出したモデルの版。ログと突合して出自を辿るために返す。"
    )


class BatchPredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    results: List[PredictResponse]
    threshold: float
    model_version: str = Field(..., description="推論に使ったモデルの版。")


# --- 推論ロジック -------------------------------------------------------------
def _predict_proba(records: List[List[Optional[float]]]):
    """生のセンサー値の配列 -> 不良確率の配列。"""
    # 列名 0..589（整数）の DataFrame を作る。
    # transform 内で str 化され、学習時の keep_cols_（"0".."589" の部分集合）に揃う。
    raw = pd.DataFrame(records)
    X = preprocessor.transform(raw)
    return model.predict_proba(X)[:, 1]


# --- エンドポイント -----------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "SECOM 不良予測 API",
        "version": app.version,
        "model": metadata["model_type"],
        "model_version": MODEL_VERSION,
        "trained_at": TRAINED_AT,
        "input_features": N_FEATURES,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    ok = preprocessor is not None and model is not None
    return {"status": "ok" if ok else "error", "model_loaded": ok}


@app.get("/metadata")
def get_metadata():
    """学習・評価時の情報。AUC などの値は厳密評価（Nested CV）の結果。"""
    return metadata


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    threshold = req.threshold if req.threshold is not None else DEFAULT_THRESHOLD
    try:
        proba = float(_predict_proba([req.features])[0])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"推論に失敗しました: {exc}") from exc
    return PredictResponse(
        defect_probability=proba,
        is_defect=proba >= threshold,
        threshold=threshold,
        model_version=MODEL_VERSION,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest):
    threshold = req.threshold if req.threshold is not None else DEFAULT_THRESHOLD
    try:
        probas = _predict_proba(req.records)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"推論に失敗しました: {exc}") from exc
    results = [
        PredictResponse(
            defect_probability=float(p),
            is_defect=bool(p >= threshold),
            threshold=threshold,
            model_version=MODEL_VERSION,
        )
        for p in probas
    ]
    return BatchPredictResponse(
        results=results, threshold=threshold, model_version=MODEL_VERSION
    )
