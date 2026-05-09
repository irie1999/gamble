# gamble

競艇（boatrace.jp）の出走表・結果データを取得し、各艇の1着確率を予測するパイプライン。
構造は競馬予想システム（スクレイパー → データ整形 → 特徴量 → モデル → 予測）と同じ流れです。

## 採用戦略（学術的根拠）

1. **条件付きロジット (Conditional Logit / Plackett-Luce)** — Bolton & Chapman (1986)
   レース内6艇の予測スコアをソフトマックス正規化することで「1着は1艇のみ」制約を
   反映した推定確率を得る。
2. **Benter式マーケットブレンド** — Benter (1994)
   モデル確率と公衆オッズが示唆する確率を対数線形でブレンド。
   `p_blend ∝ p_model^α × p_market^(1-α)` （α ≈ 0.7 推奨）
3. **EV閾値ベット + favorite-longshot 補正** — Hausch, Ziemba & Rubinstein (1981)
   `EV = p × odds > 1.05` のときのみベット候補。`p > 0.05` で極端な穴狙いを排除。
4. **Fractional Kelly 基準** — Kelly (1956), MacLean-Thorp-Ziemba (2010)
   推定誤差対策で 0.25 Kelly。バンクロールに対する1ベット上限も併用。

`src/strategy/backtest.py` は上記をパイプライン化し、過去データに対する
ROI / 勝率 / 最大ドローダウン / エクイティカーブを算出。

## ディレクトリ構成

```
.
├── requirements.txt
├── scripts/
│   ├── scrape_dataset.py   # 期間×場のデータをスクレイプして保存（出走表 + 払戻）
│   ├── train_model.py      # 特徴量生成 + LightGBM 学習
│   ├── predict_race.py     # 当日の特定レースに対する予測
│   └── backtest.py         # 過去データでのバックテスト
├── src/
│   ├── scraper/
│   │   ├── http_client.py        # リトライ・スロットリング付き HTTP クライアント
│   │   └── boatrace_scraper.py   # boatrace.jp パーサ
│   ├── data/preprocessor.py      # スクレイプ結果 → DataFrame
│   ├── features/feature_engineering.py  # 特徴量生成（リーク防止済み）
│   ├── models/
│   │   ├── train.py              # 学習（時系列分割 + LightGBM）
│   │   └── predict.py            # 推論（レース内ソフトマックス正規化）
│   ├── strategy/
│   │   ├── blending.py           # Benter式 マーケットブレンド
│   │   ├── kelly.py              # Kelly基準 / fractional Kelly
│   │   ├── ev.py                 # EV計算とベット選別
│   │   └── backtest.py           # バックテストエンジン
│   └── utils/{config.py,logger.py}
├── data/{raw,processed,models}/  # データ・モデル成果物（git管理外）
└── tests/                        # 単体テスト（ネット不要）
```

## セットアップ

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 使い方

### 1. データ取得

デフォルトは **公式LZHデータ** (`--source official`)。1日1ファイル＝高速・安定。

```bash
# 1日分（住之江と福岡のみ）
python -m scripts.scrape_dataset --from 2024-08-15 --to 2024-08-15 --venues 12 22 --out data/raw/races.csv

# 1ヶ月分・全場
python -m scripts.scrape_dataset --from 2024-08-01 --to 2024-08-31 --out data/raw/races.parquet
```

出力: `data/raw/races.parquet` と `data/raw/races_payouts.parquet`。
LZHキャッシュは `data/raw/official/{B,K}/` に保存されるため、再実行は爆速。

#### パース確認用 inspect ツール

公式ファイルの中身（生テキスト）を見たい場合:

```bash
python -m scripts.inspect_official --date 2024-08-15 --type B --venue 住之江 --race 1
```

#### HTMLスクレイプにフォールバックしたい場合

```bash
python -m scripts.scrape_dataset --source html --from 2024-08-15 --to 2024-08-15 --venues 12
```

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

### 4. バックテスト

```bash
# Kelly基準（推奨）
python -m scripts.backtest \
    --features data/processed/features.parquet \
    --payouts  data/raw/races_payouts.parquet \
    --strategy kelly --since 2024-06-01

# 比較用ベンチマーク
python -m scripts.backtest --features ... --payouts ... --strategy always_top1
python -m scripts.backtest --features ... --payouts ... --strategy model_top1
python -m scripts.backtest --features ... --payouts ... --strategy flat
```

出力: `data/models/backtest/summary_<strategy>.json`、
`bets_<strategy>.csv`、`equity_<strategy>.csv`。

`summary` には `roi` / `hit_rate` / `max_drawdown` / `ending_bankroll` を含む。

## 参考文献

- Benter, W. (1994). *Computer Based Horse Race Handicapping and Wagering Systems*.
  In Hausch, Lo & Ziemba (eds.), Efficiency of Racetrack Betting Markets.
- Bolton, R. N. & Chapman, R. G. (1986). *Searching for Positive Returns at the Track:
  A Multinomial Logit Model for Handicapping Horse Races.* Management Science 32(8).
- Hausch, D. B., Ziemba, W. T. & Rubinstein, M. (1981). *Efficiency of the Market for
  Racetrack Betting.* Management Science 27(12).
- Kelly, J. L. (1956). *A New Interpretation of Information Rate.* Bell System Tech. J.
- MacLean, L. C., Thorp, E. O. & Ziemba, W. T. (2010). *The Kelly Capital Growth
  Investment Criterion.* World Scientific.

## 実装メモ

- スクレイピングは `User-Agent` 付与・リクエスト間隔1秒・指数バックオフでマナー厳守。
- `boatrace.jp` のHTML構造に依存しているため、サイト改修時は `boatrace_scraper.py` のセレクタ調整が必要。
- 特徴量側では「当該レース日より前」の情報のみを使う（rolling/expanding を `shift(1)` 経由で集計）ことで時系列リークを防止。
- 法令・各サイトの規約・節度ある利用を前提に、自己責任で利用してください。
