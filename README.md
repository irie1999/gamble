# gamble

競艇（boatrace.jp）の出走表・結果データを取得し、各艇の1着確率を予測するパイプライン。
構造は競馬予想システム（スクレイパー → データ整形 → 特徴量 → モデル → 予測）と同じ流れです。

## ディレクトリ構成

```
.
├── requirements.txt
├── scripts/
│   ├── scrape_dataset.py   # 期間×場のデータをスクレイプして保存
│   ├── train_model.py      # 特徴量生成 + LightGBM 学習
│   └── predict_race.py     # 当日の特定レースに対する予測
├── src/
│   ├── scraper/
│   │   ├── http_client.py        # リトライ・スロットリング付き HTTP クライアント
│   │   └── boatrace_scraper.py   # boatrace.jp パーサ
│   ├── data/preprocessor.py      # スクレイプ結果 → DataFrame
│   ├── features/feature_engineering.py  # 特徴量生成（リーク防止済み）
│   ├── models/
│   │   ├── train.py              # 学習（時系列分割 + LightGBM）
│   │   └── predict.py            # 推論（レース内ソフトマックス正規化）
│   └── utils/{config.py,logger.py}
├── data/{raw,processed,models}/  # データ・モデル成果物（git管理外）
└── tests/                        # 最小単体テスト
```

## セットアップ

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 使い方

### 1. データ取得（スクレイピング）

```bash
# 例: 2024年1月の住之江(12)・福岡(22) を取得
python -m scripts.scrape_dataset --from 2024-01-01 --to 2024-01-31 --venues 12 22
# 全場なら --venues を省略
python -m scripts.scrape_dataset --from 2024-01-01 --to 2024-01-31
```

出力: `data/raw/races.parquet`（1艇1行のlong形式）。

### 2. 学習

```bash
python -m scripts.train_model --input data/raw/races.parquet --target is_win
```

- 特徴量: `src/features/feature_engineering.py::FEATURE_COLUMNS`
- 時系列で末尾20%を検証に使用、early stopping。
- 出力: `data/models/lgb_win_model.joblib`, `data/models/metrics.json`

### 3. 予測

```bash
python -m scripts.predict_race --venue 12 --race 11 --date 2024-08-15
```

- レース内6艇でソフトマックス正規化した1着確率を出力。
- 単勝オッズが手元にあれば `expected_value_table` で期待値も計算可能。

## 実装メモ

- スクレイピングは `User-Agent` 付与・リクエスト間隔1秒・指数バックオフでマナー厳守。
- `boatrace.jp` のHTML構造に依存しているため、サイト改修時は `boatrace_scraper.py` のセレクタ調整が必要。
- 特徴量側では「当該レース日より前」の情報のみを使う（rolling/expanding を `shift(1)` 経由で集計）ことで時系列リークを防止。
- 法令・各サイトの規約・節度ある利用を前提に、自己責任で利用してください。
