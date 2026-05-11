"""当日の全場・全レースから lane1_value/lane1_kelly のベット候補を抽出する。

朝〜昼に実行して、当日賭ける候補レース一覧を出力する。
出走表→確率予測→単勝オッズ取得→Benterブレンド→EVフィルタ→Kellyステーク。

使い方:
    python -m scripts.recommend_today
    python -m scripts.recommend_today --date 2026-05-10 --ev-threshold 1.10
    python -m scripts.recommend_today --workers 8        # 並列度を上げる
    python -m scripts.recommend_today --venues 04 12     # 特定場のみ
    python -m scripts.recommend_today --flat 1000 --kelly-fraction 0

    # HTMLレポートを併せて出力（ブラウザで一覧確認・列ソート可・各行から公式オッズへ）
    python -m scripts.recommend_today --html data/processed/signals_today.html

各レースについて、現在オッズで評価する。締切直前にもう一度回すのが理想。
"""
from __future__ import annotations

import argparse
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from threading import local
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.data.preprocessor import build_dataset
from src.features.feature_engineering import build_features
from src.models.predict import _softmax
from src.scraper.boatrace_scraper import BoatraceScraper, RaceResult
from src.scraper.http_client import HttpClient
from src.scraper.odds_scraper import OddsScraper
from src.strategy.blending import add_blended_probability
from src.strategy.kelly import kelly_stake
from src.utils.config import BASE_URL, MODELS_DIR, VENUE_CODES
from src.utils.logger import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore")  # LightGBM の warning 抑制
logging.getLogger("lightgbm").setLevel(logging.ERROR)


# スレッドごとに独立した HttpClient/Scraper を持たせる（HttpClient の throttle が共有されないように）
_TLS = local()


def _scrapers() -> tuple[BoatraceScraper, OddsScraper]:
    if not hasattr(_TLS, "br"):
        client_a = HttpClient(interval_sec=0.3)
        client_b = HttpClient(interval_sec=0.3)
        _TLS.br = BoatraceScraper(client=client_a)
        _TLS.od = OddsScraper(client=client_b)
    return _TLS.br, _TLS.od


def _predict_with_bundle(features_df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """predict_win_probability の bundle 受け取り版。joblib.load を毎回しない。"""
    model = bundle["model"]
    calibrator = bundle.get("calibrator")
    feat_cols = bundle["feature_columns"]
    X = features_df[feat_cols]
    raw = model.predict_proba(X)[:, 1]
    score = calibrator.transform(raw) if calibrator is not None else raw
    out = features_df.copy()
    out["raw_score"] = score
    out["pred_win_prob"] = (
        out.groupby("race_id")["raw_score"]
        .transform(lambda s: _softmax(s.to_numpy()))
    )
    return out


def _process_race(
    jcd: str, race_no: int, target: date, bundle: dict,
    blend_alpha: float, takeout: float,
) -> Optional[dict]:
    br, od = _scrapers()
    try:
        card = br.fetch_race_card(jcd, race_no, target)
    except Exception as e:
        logger.debug("racelist failed jcd=%s rno=%s: %s", jcd, race_no, e)
        return None
    empty = RaceResult(race_date=card.race_date, venue_code=card.venue_code, race_no=card.race_no)
    df = build_dataset([(card, empty)])
    feats = build_features(df)
    pred = _predict_with_bundle(feats, bundle)

    try:
        wo = od.fetch_win_odds(jcd, race_no, target)
    except Exception as e:
        logger.debug("odds failed jcd=%s rno=%s: %s", jcd, race_no, e)
        return None
    if not wo.odds or len(wo.odds) < 6:
        return None
    pred = pred.copy()
    pred["odds_win"] = pred["lane"].map(wo.odds).astype(float)
    if pred["odds_win"].isna().any():
        return None
    blended = add_blended_probability(pred, alpha=blend_alpha, takeout=takeout)
    pred["blended_win_prob"] = blended["blended_win_prob"].values

    lane1 = pred[pred["lane"] == 1].iloc[0]
    return {
        "venue": jcd,
        "venue_name": VENUE_CODES.get(jcd, "?"),
        "race_no": race_no,
        "lane": 1,
        "racer": lane1.get("name", ""),
        "p_model": float(lane1["pred_win_prob"]),
        "p_blend": float(lane1["blended_win_prob"]),
        "odds_win": float(lane1["odds_win"]),
        "ev": float(lane1["blended_win_prob"]) * float(lane1["odds_win"]),
    }


def _scan_targets(target: date, venues: list[str], workers: int) -> list[tuple[str, int]]:
    """各場の開催レース番号を並列に取得。"""
    out: list[tuple[str, int]] = []
    def task(jcd: str) -> tuple[str, list[int]]:
        br, _ = _scrapers()
        try:
            return jcd, br.fetch_race_index(jcd, target)
        except Exception as e:
            logger.warning("raceindex failed jcd=%s: %s", jcd, e)
            return jcd, []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for jcd, rnos in ex.map(task, venues):
            for r in rnos:
                out.append((jcd, r))
    return out


SIGNAL_HTML_TEMPLATE = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>Signals {date}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 1100px; margin: 24px auto; padding: 0 16px; color: #1a1a1a; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  .meta {{ color: #666; font-size: 13px; }}
  .kpi {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin: 16px 0 24px; }}
  .kpi .card {{ background: #f7f7f9; border-radius: 8px; padding: 12px; }}
  .kpi .label {{ color: #666; font-size: 12px; }}
  .kpi .value {{ font-size: 22px; font-weight: 600; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align: right; padding: 6px 10px; border-bottom: 1px solid #eee; }}
  th:nth-child(-n+3), td:nth-child(-n+3) {{ text-align: left; }}
  th {{ background: #f7f7f9; cursor: pointer; user-select: none; }}
  th:hover {{ background: #eef; }}
  tr:hover td {{ background: #fafafa; }}
  .ev-high {{ color: #1f883d; font-weight: 600; }}
  .ev-mid {{ color: #b3870e; }}
  a {{ color: #1f6feb; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>

<h1>当日シグナル: {date}</h1>
<div class="meta">
  generated: {generated} &nbsp;|&nbsp; strategy: lane1_kelly &nbsp;|&nbsp; ev_threshold: {ev_threshold} &nbsp;|&nbsp; kelly_fraction: {kelly_fraction}
</div>

<div class="kpi">
  <div class="card"><div class="label">推奨ベット</div><div class="value">{n_bets}件</div></div>
  <div class="card"><div class="label">合計ステーク</div><div class="value">¥{total_stake}</div></div>
  <div class="card"><div class="label">最高EV</div><div class="value">{max_ev}</div></div>
  <div class="card"><div class="label">平均オッズ</div><div class="value">{avg_odds}</div></div>
</div>

<table id="signals">
  <thead><tr>
    <th>場</th><th>R</th><th>選手</th>
    <th>p_model</th><th>p_blend</th><th>オッズ</th><th>EV</th><th>ステーク</th><th>リンク</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table>

<script>
// 列ヘッダクリックで並べ替え（数値列は数値、文字列列は文字列としてソート）
document.querySelectorAll('#signals th').forEach((th, idx) => {{
  let asc = false;
  th.addEventListener('click', () => {{
    const tbody = th.closest('table').querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    rows.sort((a, b) => {{
      const av = a.children[idx].dataset.sort ?? a.children[idx].textContent;
      const bv = b.children[idx].dataset.sort ?? b.children[idx].textContent;
      const an = parseFloat(av), bn = parseFloat(bv);
      const cmp = (!isNaN(an) && !isNaN(bn)) ? an - bn : String(av).localeCompare(String(bv), 'ja');
      return asc ? cmp : -cmp;
    }});
    asc = !asc;
    rows.forEach(r => tbody.appendChild(r));
  }});
}});
</script>

</body>
</html>
"""


def _ev_class(ev: float) -> str:
    if ev >= 1.20:
        return "ev-high"
    if ev >= 1.10:
        return "ev-mid"
    return ""


def _signal_row(r: dict, date_str: str) -> str:
    url = f"{BASE_URL}/oddstf?rno={r['race_no']}&jcd={r['venue']}&hd={date_str}"
    racer = str(r.get("racer", "")).strip() or "-"
    ev_cls = _ev_class(r["ev"])
    return (
        f"<tr>"
        f"<td>{r['venue_name']}({r['venue']})</td>"
        f"<td data-sort='{r['race_no']}'>{r['race_no']}R</td>"
        f"<td>{racer}</td>"
        f"<td>{r['p_model']:.3f}</td>"
        f"<td>{r['p_blend']:.3f}</td>"
        f"<td>{r['odds_win']:.2f}</td>"
        f"<td class='{ev_cls}'>{r['ev']:.3f}</td>"
        f"<td>¥{r['stake_yen']:,}</td>"
        f"<td><a href='{url}' target='_blank'>オッズ</a></td>"
        f"</tr>"
    )


def _write_html(rows: list[dict], target: date, out_path: Path,
                ev_threshold: float, kelly_fraction: float) -> None:
    from datetime import datetime
    date_str = target.strftime("%Y%m%d")
    total_stake = sum(r["stake_yen"] for r in rows)
    max_ev = max((r["ev"] for r in rows), default=0.0)
    avg_odds = sum(r["odds_win"] for r in rows) / len(rows) if rows else 0.0
    html = SIGNAL_HTML_TEMPLATE.format(
        date=target.isoformat(),
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        ev_threshold=f"{ev_threshold:.2f}",
        kelly_fraction=f"{kelly_fraction:.2f}",
        n_bets=len(rows),
        total_stake=f"{total_stake:,}",
        max_ev=f"{max_ev:.3f}" if rows else "-",
        avg_odds=f"{avg_odds:.2f}" if rows else "-",
        rows="".join(_signal_row(r, date_str) for r in rows),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=None, help="YYYY-MM-DD（デフォルト今日）")
    p.add_argument("--venues", nargs="*", default=None, help="対象場コード。省略時は全24場")
    p.add_argument("--exclude-venues", nargs="*", default=["24"],
                   help="除外場コード（デフォルト 24=大村）")
    p.add_argument("--ev-threshold", type=float, default=1.05)
    p.add_argument("--max-odds", type=float, default=None,
                   help="このオッズ超の1号艇は除外（高オッズ1号艇=構造的に弱いレース対策）")
    p.add_argument("--blend-alpha", type=float, default=0.7)
    p.add_argument("--takeout", type=float, default=0.25)
    p.add_argument("--bankroll", type=float, default=100_000.0)
    p.add_argument("--kelly-fraction", type=float, default=0.25,
                   help="0 にすると --flat のフラットベットになる")
    p.add_argument("--flat", type=float, default=1_000.0,
                   help="kelly_fraction=0 の時のステーク額")
    p.add_argument("--workers", type=int, default=6, help="並列度（HTTP同時接続数）")
    p.add_argument("--out", default=None, help="CSV 出力先（任意）")
    p.add_argument("--html", default=None,
                   help="HTMLレポート出力先（任意）。例: data/processed/signals_today.html")
    args = p.parse_args()

    target = date.fromisoformat(args.date) if args.date else date.today()
    all_codes = list(VENUE_CODES.keys())
    venues = [v for v in (args.venues or all_codes) if v not in set(args.exclude_venues)]
    print(f"対象日={target} 対象場={','.join(venues)} workers={args.workers}")

    bundle = joblib.load(MODELS_DIR / "lgb_win_model.joblib")

    # ① 各場の開催レース番号
    targets = _scan_targets(target, venues, args.workers)
    print(f"開催レース総数: {len(targets)} → 各レースで racelist+odds の2リクエスト")
    if not targets:
        print("(本日は対象レースなし)")
        return

    # ② レース毎に並列処理
    rows: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(_process_race, jcd, rno, target, bundle,
                      args.blend_alpha, args.takeout): (jcd, rno)
            for jcd, rno in targets
        }
        for fut in as_completed(futures):
            done += 1
            res = fut.result()
            if done % 20 == 0 or done == len(futures):
                print(f"  進捗 {done}/{len(futures)}", flush=True)
            if res is None or res["ev"] <= args.ev_threshold:
                continue
            if args.max_odds is not None and res["odds_win"] > args.max_odds:
                continue
            if args.kelly_fraction > 0:
                stake = kelly_stake(
                    bankroll=args.bankroll,
                    p=res["p_blend"], odds=res["odds_win"],
                    fraction=args.kelly_fraction,
                )
            else:
                stake = args.flat
            if stake <= 0:
                continue
            res["stake_yen"] = int(stake)
            rows.append(res)

    if not rows:
        print(f"\n(EV>{args.ev_threshold} を満たすレースなし)")
        return

    out = pd.DataFrame(rows).sort_values("ev", ascending=False)
    print(f"\n=== {target} 推奨ベット ({len(out)}件) ===")
    print(out.to_string(index=False, formatters={
        "p_model": "{:.3f}".format, "p_blend": "{:.3f}".format,
        "odds_win": "{:.2f}".format, "ev": "{:.3f}".format,
    }))
    print(f"\n合計ステーク: ¥{out['stake_yen'].sum():,}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False)
        print(f"\nsaved: {args.out}")

    if args.html:
        html_path = Path(args.html)
        _write_html(rows, target, html_path,
                    ev_threshold=args.ev_threshold,
                    kelly_fraction=args.kelly_fraction)
        print(f"\nHTML saved: {html_path.resolve()}")
        print(f"open in browser: file://{html_path.resolve()}")


if __name__ == "__main__":
    main()
