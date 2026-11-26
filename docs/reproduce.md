# 再現と動かし方の詳細

基本の手順（依存・学習・API・テスト）は [README の動かし方](../README.md#動かし方) にある。

## Docker

イメージには `src/` と `models/` だけを含める。

```bash
docker compose up --build
curl http://localhost:8000/health
# => {"status":"ok","model_loaded":true}
```

## 比較・図・デモの再生成

```bash
python -m scripts.compare_models      # → reports/model_comparison.md
python -m scripts.compare_configs     # → reports/config_comparison.md
python -m scripts.compare_group_info  # → reports/group_info_comparison.md
python -m scripts.reproduce_leak      # → reports/leak_check.md
python -m scripts.check_headline      # → reports/headline_checks.md

pip install -r requirements-extras.txt   # 図とデモに要る。図には日本語フォントも要る
python -m scripts.replay_drift           # → reports/drift_report.md、fig7
python -m scripts.make_figures           # → fig1〜fig6
python app.py                            # デモ → http://localhost:7860
```

> `models/*.pkl` はすぐ試せるよう同梱しているが、pickle は読み込み時に任意のコードを実行しうる。信頼できる出所からのみ読み込み、不安なら `python -m src.train` で作り直す。

## エンドポイント

| メソッド | パス | 説明 |
|---|---|---|
| GET | `/` | サービス概要 |
| GET | `/health` | ヘルスチェック |
| GET | `/metadata` | モデル・評価値・しきい値の規則 |
| POST | `/predict` | 1 件の推論（`features`: 590 個、欠損は `null` 可。`threshold` は任意、省略時は既定値） |
| POST | `/predict/batch` | 複数件の推論 |

## ファイルの一覧

```text
secom-defect-prediction/
├── requirements.txt          # API・学習の依存（Docker はこれだけ）
├── requirements-dev.txt      # + テスト（CI が入れる）
├── requirements-extras.txt   # + 図・Gradio デモ
├── Dockerfile / docker-compose.yml
├── sample_input.json         # API の動作確認用（実データの不良行 1 件）
├── app.py                    # Gradio デモ（Hugging Face Spaces へは手で同期）
├── MODEL_CARD.md             # モデルの用途・評価・限界
├── src/
│   ├── preprocessing.py      # 前処理クラス
│   ├── train.py              # 時系列CV、しきい値の決定、最終モデルの学習と保存
│   ├── monitoring.py         # PSI・未知グループ率・深刻度の判定
│   └── api.py                # FastAPI の推論サーバー
├── models/                   # 学習済みの preprocessor / model / metadata
├── data/                     # UCI のオリジナルファイル
├── notebooks/
│   └── secom_strict_analysis.ipynb  # 考察編の分析（Step 2〜5: 評価、Step 6: 品種、Step 7: 組み込み）
├── reports/
│   ├── leak_check.md             # リークの影響
│   ├── model_comparison.md       # モデルの比較
│   ├── config_comparison.md      # 前処理・特徴量の構成の比較
│   ├── group_info_comparison.md  # 品種情報を入れる方法の比較
│   ├── drift_report.md           # 監視のリプレイ結果
│   └── figures/                  # 図（fig1〜fig7）
├── scripts/
│   ├── auc_tests.py              # fold 別 DeLong + Holm
│   ├── reproduce_leak.py         # → leak_check.md
│   ├── compare_models.py         # → model_comparison.md
│   ├── compare_configs.py        # → config_comparison.md
│   ├── compare_group_info.py     # → group_info_comparison.md
│   ├── replay_drift.py           # → drift_report.md、fig7
│   ├── make_figures.py           # → fig1〜fig6
│   ├── figure_style.py           # 図の共通設定
│   └── make_sample_input.py      # → sample_input.json
├── tests/                    # API・監視のテスト（pytest）
└── .github/workflows/ci.yml  # 学習 → テスト
```
