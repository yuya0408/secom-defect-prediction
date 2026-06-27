---
license: mit
library_name: sklearn
tags:
  - tabular-classification
  - manufacturing
  - anomaly-detection
  - secom
metrics:
  - roc_auc
  - pr_auc
---

# SECOM 不良予測モデル

半導体製造ラインのセンサー値（590 次元）から、その製品が不良である確率を出力する
二値分類モデル。Hugging Face Hub に公開する際は、このファイルをモデルリポジトリの
`README.md`（モデルカード）として配置する。

## 概要

| 項目 | 値 |
|------|------|
| アルゴリズム | RandomForestClassifier（`n_estimators=200, max_depth=10, class_weight="balanced"`） |
| 入力 | 生のセンサー値 590 個（欠損は `null`／NaN 可。前処理で中央値補完・欠損フラグ化） |
| 前処理後の特徴量 | 460（436 列 + 欠損フラグ 24 列） |
| 出力 | 不良確率 `[0, 1]`、既定しきい値 0.5 |
| 学習データ | UCI SECOM 1,567 サンプル（不良率 6.64%、強い不均衡） |

## 評価（厳密・リーク排除）

前処理・特徴量選択を含む全工程を各 fold の訓練データ内に閉じ込め、時間順を守る
`TimeSeriesSplit` で測定した「真の実力値」（ハイパーパラメータは固定で、内側ループでの
探索は行っていない）。

| 指標 | 値 | 補足 |
|------|------|------|
| ROC-AUC | **0.6074** | リーク版の見かけ上の 0.72 ではなく、これが実力 |
| PR-AUC | 0.1346 | ベースライン（不良率 0.0664）の約 2 倍 |

## 想定用途と限界

- **用途**: 検査工程での自動振り分け（不良確率の高い製品を優先検査へ）、品質モニタリング、
  根本原因分析の入口。あくまで**事後検知**の補助。
- **限界**: AUC 0.6 台で精度には限界がある。単独の合否判定には使わず、しきい値調整と
  定期的な再学習・再評価を前提とする。入力分布のドリフト（品種ミックスの変動）に注意。

## 入出力例

```json
// 入力（POST /predict）
{"features": [/* 590 個の値。欠損は null */], "threshold": 0.5}

// 出力
{"defect_probability": 0.6597, "is_defect": true, "threshold": 0.5, "model_version": "1.0.0"}
```

## ライセンス・出典

- コード/モデル: MIT License
- データ: McCann, M. & Johnston, A. (2008). *SECOM* [Dataset]. UCI Machine Learning Repository.
