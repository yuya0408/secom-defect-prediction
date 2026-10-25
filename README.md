# SECOM 不良予測 ― データを読み解き、運用可能な形にするまで

半導体製造ラインのセンサーデータ（[UCI SECOM データセット](https://archive.ics.uci.edu/dataset/179/secom)）を題材に、**データリークを排除した厳密な分析**を行い、その結論を再現可能な推論サービス（FastAPI + Docker）と、ドリフト監視の実演までまとめたプロジェクト。

元・電子部品メーカーの製造技術職（不良改善が専門）が、現場感覚とデータ分析を掛け合わせて「このデータが何を物語っているか」を読み解いた記録でもある。

**デモ**: [Hugging Face Spaces](https://huggingface.co/spaces/yuya-h/secom-defect-demo) でブラウザから試せる（推論 API と同じ前処理・モデル）。

> 予測精度を競うプロジェクトではない。「精度に限界のあるモデルでも、誰でも同じ結果を再現でき、使える範囲と使えない範囲が分かる状態にする」までの一連の流れを示すことが目的。

## 解説記事（Zenn）

| 記事 | 内容 | 対応するコード |
|------|------|------|
| [考察編](https://zenn.dev/yuya0408/articles/secom-defect-analysis) | データをどう読むか。厳密な評価で何が分かったか | `notebooks/`, `reports/figures/` |
| [実装編](https://zenn.dev/yuya0408/articles/secom-defect-serving) | API 化・コンテナ化・テストと CI | `src/`, `models/`, `Dockerfile` |
| [運用編](https://zenn.dev/yuya0408/articles/secom-defect-operations) | ドリフト監視と、検知したときの動き方 | `src/monitoring.py`, `scripts/replay_drift.py` |

> 記事を公開したあと、しきい値の決め方とモデル・構成の比較を検定で見直した（2026-10）。結果は下の「[評価と判定](#評価と判定)」にまとめてあり、記事より新しい。

---

## 結果サマリー

| 項目 | 値 |
|------|------|
| データ | 1,567 サンプル × 590 センサー特徴量 |
| ラベル | 合格 1,463 / 不良 104（不良率 **6.64%**、強い不均衡） |
| 欠損 | 41,951 セル（全体の 4.54%） |
| モデル | きちんとした前処理 + Random Forest |
| **厳密評価 AUC** | **0.6074**（fold 別 0.588 / 0.565 / 0.653 / 0.564 / 0.667。全工程を fold 内に閉じ込めた時系列CV） |
| 厳密評価 PR-AUC | 0.1346（評価対象の不良率 0.050 の約 2.7 倍） |
| 既定しきい値 | 0.0612（「不良確率の上位 50% を検査に回す」規則から決めた値） |
| 上位 50% を検査に回したときの再現率 | 66%（fold 別 60〜71%。ランダムに選ぶと 50%） |
| 前処理後の特徴量数 | 590 → 460（436 列 + 欠損フラグ 24 列） |

---

## 分析の要点（考察編）

### 1. リークを除くと AUC は 0.72 → 0.6074 に下がる

当初、特徴量選択で AUC 0.72 が出たが、これは**全データで特徴量重要度を計算してから交差検証する**というデータリークが生んだ見かけの値だった。前処理・特徴量選択を含む全工程を各 fold の訓練データ内に閉じ込め、時間順を守る `TimeSeriesSplit` で測り直すと、実力は **AUC 0.6074** だった（ハイパーパラメータは固定で、内側ループでの探索は行っていない）。

![TimeSeriesSplit による時系列分割](reports/figures/fig6_timeseries_split.png)

時系列特徴量や特徴量選択といった凝った工夫は、見分けられる上積みにならなかった（検定の結果は「[評価と判定](#評価と判定)」）。

![各アプローチのAUC比較](reports/figures/fig4_auc_comparison.png)

### 2. 欠損パターンから「品種ミックス」が見えた

欠損は単なるデータの抜けではなく、**どの品種がどの工程を通った（あるいはバイパスした）か**を表している可能性がある。欠損率 10〜70% の 44 列の欠損パターンでサンプルを分類すると、ユニークなパターンはわずか 49 種類に収束し、グループごとの不良率は 0%〜23% と大きく異なっていた。

![グループ別の不良率](reports/figures/fig1_group_defect_rate.png)

![月別の不良率推移](reports/figures/fig2_monthly_defect_rate.png)

月別不良率の大きな変動（7月 22% → 9月 3%）は、その月にどの品種が多く流れていたか（高不良グループの割合）と強い正の関係が見られた（r ≈ 0.91）。ただし月数は 4 ヶ月のみ（n=4, p≈0.09）で、統計的有意水準には達しない。品種ミックス仮説を支持する材料として示すにとどめる。

![月別不良率と高不良グループ割合の関係](reports/figures/fig3_correlation.png)

### 3. 発見はモデル精度には繋げられなかった

グループ ID の特徴量追加など複数の方法でグループ情報をモデルに教えたが、いずれもベースラインを超えられなかった。理由は 3 つ。(1) グループあたりのサンプル数が少なく安定した学習ができない、(2) 時系列評価では学習時に未知だったグループがテスト期に出現する、(3) 欠損フラグにグループ情報がすでに含まれており新情報にならない。

![各foldでの新グループ出現率](reports/figures/fig5_fold_new_group.png)

> 補足: 前職は電子部品（金属製品）製造で、半導体ウエハ製造そのものの実務経験はない。品種ミックスの解釈は、製造業全般に共通する現場感覚からの推測であって断定ではない。

| 考察編の章 | 場所 |
|---------|------|
| 第2〜4章（前処理・リーク排除した時系列CV・パターン比較） | `notebooks/secom_strict_analysis.ipynb`（Step 2〜5）, `src/preprocessing.py` |
| 第5章（品種ミックス） | `notebooks/`（Step 6: 欠損パターン分析） |
| 第6章（月別不良率の正体） | `notebooks/`（Step 6 後半: 相関分析） |
| 第7章（品種情報をモデルに使えるか） | `notebooks/`（Step 7: 組み込み試行） |

---

## 評価と判定

数字はすべて、学習（`src/train.py`）と同じ厳密な時系列CVの out-of-fold 予測（1,305 件、うち不良 65 件）から出している。最初の 1/6 は学習専用で予測を持たないため、評価対象の不良率は 5.0% になる。

### しきい値：「上位 50% を検査に回す」から決める

未知のデータに対する不良確率は最大でも 0.36 で、しきい値 0.5 では 1 件も不良と判定されない。しかも学習データが増えるほど確率は低く出る（fold ごとの平均 0.153 → 0.065）ため、確率の値で決めたしきい値は再学習のたびにずれる。そこで、順位で決める規則にした。

| 検査に回す割合 | 再現率 | ランダムに選んだ場合 | fold 別の再現率 |
|---|---|---|---|
| 30% | 38.5% | 30% | 36 / 17 / 60 / 23 / 50 |
| 50% | 66.2% | 50% | 64 / 67 / 60 / 69 / 71 |
| 70% | 84.6% | 70% | 82 / 83 / 80 / 92 / 86 |
| 90% | 92.3% | 90% | 91 / 100 / 90 / 92 / 93 |

規則は**業務上の仮定**である。1 個あたりの費用を「検査に回す割合 + r × 見逃した不良の割合」（r = 流出 1 件の損失 ÷ 検査 1 件の費用）とすると、50% は r ≈ 20 で費用が最小になる点（46%）に近い。**r が約 50 を超える工程では全数検査の方が安く、このモデルは検査の優先順位づけにしか使えない。**

`src/train.py` は学習のたびにこの表を作り直し、最後の fold（最終モデルに最も近い）の予測分布で上位 50% の境目を既定しきい値にする。全データで学習する最終モデルは確率がさらに低く出うるので、これは近似値である。規則・前提・表は `models/metadata.pkl` に記録され、API の `/metadata` から読める。

### モデル・構成の比較：入れ替えるほどの差は無い

現行（きちんとした前処理 + RF）と、モデル 2 種・前処理/特徴量の構成 5 種を同じ CV で比べた。RF は比較して選んだモデルではなく、高次元・少サンプルの表形式データでチューニングをほとんどせずに崩れにくい出発点として置いたもので、ここで初めて比べている。

- **検定**: fold ごとに DeLong 検定の分散を出して fold 平均の AUC 差にまとめ、Holm 補正をかける（[`scripts/auc_tests.py`](scripts/auc_tests.py)）
- **規則**（結果を見る前に決めた）: Holm 補正後 p < 0.05 で相手が良いときだけ入れ替える。替えると API・監視の基準・文書をやり直す費用がかかるため

| 比較 | AUC | 現行との差 | 95%区間 | p（Holm） |
|---|---|---|---|---|
| ロジスティック回帰 | 0.6141 | +0.007 | [−0.098, +0.112] | 0.900 |
| HistGradientBoosting | 0.5717 | −0.036 | [−0.108, +0.036] | 0.663 |
| A: 雑な前処理 | 0.5210 | −0.086 | [−0.163, −0.010] | 0.136 |
| D: + 時系列特徴量 | 0.5985 | −0.009 | [−0.075, +0.057] | 1.000 |
| E-100: + 時系列特徴量 + 特徴量選択（上位100列） | 0.6108 | +0.003 | [−0.064, +0.071] | 1.000 |

どれも現行と見分けられず、RF のままにした。区間の幅は ±0.07〜0.1 あり、AUC でそれ未満の差はこのデータ量では見分けられない（「同等」という意味ではない）。全構成の表は [`reports/model_comparison.md`](reports/model_comparison.md)・[`reports/config_comparison.md`](reports/config_comparison.md)。

```bash
python -m scripts.compare_models    # モデルの比較 → reports/model_comparison.md
python -m scripts.compare_configs   # 構成の比較 → reports/config_comparison.md
```

---

## 実装のポイント（実装編）

**学習済みの前処理クラスを、推論サーバーから安全に復元できるようにしている。** `joblib.dump()` で保存した pickle には、クラス本体ではなく「クラスがどのモジュールに定義されているか」というパスだけが記録される。ノートブックの中で定義したまま保存するとパスが `__main__.SecomPreprocessor` になり、別プロセスの API からは復元できない。そこで前処理クラスを [`src/preprocessing.py`](src/preprocessing.py) に切り出し、学習スクリプトと API の双方が `from src.preprocessing import SecomPreprocessor` で参照する。

**評価値としきい値は、学習スクリプトが計算してモデルと一緒に保存する。** 手で書いた値を持たないので、データやモデルが変わっても評価値・しきい値とモデルの組み合わせがずれない。CI は push・PR のたびに `data/` からモデルを再生成し、テストを回す。

**依存は役割で分けている。** `requirements.txt`（API・学習）、`requirements-dev.txt`（+ テスト。CI が入れる）、`requirements-extras.txt`（+ ノートブック・図・Gradio デモ）。Docker イメージは `requirements.txt` だけを入れる。

---

## ドリフト監視（運用編、時系列リプレイの実演）

時系列評価では、学習時に未知だった品種グループがテスト期に出現する。そのドリフトの監視を、凍結データの上で**実際に動かして見せる**。`src/monitoring.py`（外部依存なしの素の実装）と `scripts/replay_drift.py` から成る。

```bash
python -m scripts.replay_drift
# => reports/drift_report.md と reports/figures/fig7_drift_replay.png を生成
```

検査タイムスタンプで全データを月ごとに分割し、**2008-07〜08 を参照（学習）期、09・10 を本番月**として順に流す。各本番月で 3 種のドリフトを計測し、緑/黄/赤を判定する。

| 指標 | 見るもの | ラベル要否 |
|------|----------|-----------|
| 未知グループ率 | 欠損パターン（品種ミックス）が学習時に無かった割合 | 不要（即時） |
| 予測 PSI | 不良確率分布のずれ（参照期の交差検証 out-of-fold が基準） | 不要（即時） |
| PR-AUC | 実際の検査結果に対する性能（基準を下回ったら赤） | 必要（ラベル確定後） |

| 月 | 未知グループ率 | 予測PSI | PR-AUC | 判定 |
|----|----------------|---------|--------|------|
| 2008-09 | 9.2% | 0.279 | 0.202 | RED |
| 2008-10 | **35.9%** | 0.397 | 0.141 | RED |

未知グループ率は **9% → 36%** と月を追って増え、考察編の発見（時系列で新グループが出現）が運用指標として再現される。

境界値と基準について:

- 予測 PSI の 0.10 / 0.25 は PSI で慣用される目安。未知グループ率の 5% / 15% は**仮置き**で、実運用では誤報と見逃しの実績から決め直す前提
- 参照期の基準（PSI の分布・PR-AUC）は、時間順ではなく不良で層別した 5 分割の out-of-fold 予測で作っている。参照期の不良が 65 件と少なく、各 fold の不良数をそろえるため。時間順の評価より楽観的な基準になりうる

判定はコード（`classify_severity`）で行い、**実行は人間が承認する**設計（`RECOMMENDED_ACTIONS`）。

- **緑**: 通常運用。指標を記録するのみ
- **黄**: 推論は継続しつつ通知・ダッシュボード警告。レスポンスに `drift_warning` を付与
- **赤**: 推論は返すが信頼度を下げ、該当ロットを全数検査へ。**再学習を起票（自動実行しない）**

これは凍結データ上の実演で、本番ラインの監視ではない。アラート・ダッシュボード・再学習パイプラインといった常時稼働の基盤は作っていない。

---

## 動かし方

```bash
# 1. 依存パッケージ（ノートブックや Gradio デモも使うなら requirements-extras.txt）
pip install -r requirements-dev.txt

# 2. モデルを data/ から学習し直す（評価値・既定しきい値もここで計算される）
python -m src.train

# 3. API 起動 → http://localhost:8000/docs
uvicorn src.api:app --reload

# 4. 推論を試す
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_input.json
# => {"defect_probability": 0.6597, "is_defect": true, "threshold": 0.0612, "model_version": "1.1.0"}

# 5. テスト
pytest -q
```

`sample_input.json` は実データの不良だった行で、**学習データに含まれる**。そのため確率は 0.66 と高く出るが、未知のデータでの確率は最大でも 0.36 程度である（「[評価と判定](#評価と判定)」）。

Docker で起動する場合（イメージには `src/` と `models/` だけを含める）:

```bash
docker compose up --build
curl http://localhost:8000/health
```

Gradio デモをローカルで起動する場合（→ http://localhost:7860）:

```bash
pip install -r requirements-extras.txt
python app.py
```

> **モデルファイルについて**: `models/*.pkl` はすぐ試せるよう同梱しているが、pickle は load 時に任意コードを実行しうる。信頼できる出所からのみ load し、不安なら `python -m src.train` で `data/` から再生成すれば、他人のバイナリを読まずに済む。

### API エンドポイント

| メソッド | パス | 説明 |
|---------|------|------|
| GET | `/` | サービス概要 |
| GET | `/health` | ヘルスチェック |
| GET | `/metadata` | モデル・評価値・しきい値の規則 |
| POST | `/predict` | 1 件の推論（`features`: 590 個、欠損は `null` 可。`threshold` 任意、省略時は既定値） |
| POST | `/predict/batch` | 複数件の推論 |

モデルの用途と限界は [`MODEL_CARD.md`](MODEL_CARD.md) にまとめてある。

---

## ディレクトリ構成

```
secom-defect-prediction/
├── requirements.txt          # API・学習の依存（固定バージョン。Docker はこれだけ）
├── requirements-dev.txt      # + テスト（CI が入れる）
├── requirements-extras.txt   # + ノートブック・図の再生成・Gradio デモ
├── Dockerfile / docker-compose.yml
├── sample_input.json         # API 動作確認用のサンプル（実データの不良行 1 件）
├── app.py                    # Gradio デモ
├── MODEL_CARD.md             # モデルカード（用途・評価・限界）
├── src/
│   ├── preprocessing.py      # 前処理クラス（リーク防止 / 独立モジュール）
│   ├── train.py              # 時系列CVでの評価・しきい値の決定・最終モデルの学習と保存
│   ├── monitoring.py         # ドリフト監視（PSI / 未知グループ率 / 深刻度判定）
│   └── api.py                # FastAPI 推論サーバー
├── models/                   # 学習済み preprocessor / model / metadata
├── data/                     # UCI 公式のオリジナルファイル
├── notebooks/
│   └── secom_strict_analysis.ipynb  # 考察編の分析
├── reports/
│   ├── model_comparison.md   # モデルの比較（DeLong + Holm）
│   ├── config_comparison.md  # 前処理・特徴量の構成の比較
│   ├── drift_report.md       # ドリフト監視のリプレイ結果
│   └── figures/              # 図表（fig1〜fig7）
├── scripts/
│   ├── auc_tests.py          # fold 別 DeLong 検定 + Holm 補正
│   ├── compare_models.py     # モデルの比較
│   ├── compare_configs.py    # 構成の比較
│   ├── replay_drift.py       # 時系列リプレイによるドリフト監視の実演
│   ├── regen_fig3.py         # 図 3 の再生成
│   └── make_sample_input.py  # sample_input.json の生成
├── tests/                    # API・監視のテスト（pytest）
└── .github/workflows/ci.yml  # 学習 → テストの CI
```

---

## データセットについて

McCann, M. & Johnston, A. (2008). *SECOM* [Dataset]. UCI Machine Learning Repository.

`data/` の `secom.data`（スペース区切りのセンサー値）・`secom_labels.data`（ラベルと検査タイムスタンプ）・`secom.names`（データセットの説明）は、[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/179/secom) で配布されているオリジナルファイルをそのまま収録している。
