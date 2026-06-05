# SECOM 不良予測 API のコンテナイメージ
FROM python:3.12-slim

# Python のログを即時出力し、.pyc を作らない
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 依存だけ先にインストールしてレイヤーキャッシュを効かせる
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリ本体と学習済みモデルをコピー
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

# src.api:app を起動。src.preprocessing が import 可能なため pickle を復元できる。
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
