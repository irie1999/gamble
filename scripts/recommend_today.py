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
import time
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
        # read timeout=15s: boatrace.jp は正常時1〜2秒で返るが、メンテ明けや
        # ピーク時は5〜10秒かかることがある。短すぎると正常応答も切ってしまう。
        client_a = HttpClient(interval_sec=0.3, timeout_sec=15)
        client_b = HttpClient(interval_sec=0.3, timeout_sec=15)
        _TLS.br = BoatraceScraper(client=client_a)
        _TLS.od = OddsScraper(client=client_b)
    return _TLS.br, _TLS.od


def _result_scraper() -> BoatraceScraper:
    """結果取得用の別スクレイパー（タイムアウトは長め10s）。

    結果ページはレース直後（数分以内）は生成中で遅いことがあるため、odds 用クライアントとは
    別に保持する。スレッドローカルで HttpClient のスロットリングを独立させる目的もある。
    """
    if not hasattr(_TLS, "br_result"):
        client = HttpClient(interval_sec=0.3, timeout_sec=10)
        _TLS.br_result = BoatraceScraper(client=client)
    return _TLS.br_result


def _silence_lightgbm(bundle: dict) -> None:
    """sklearn alias と LightGBM native 名のパラメータ衝突を解消し、

    predict 毎の C++ 由来の "X will be ignored" 警告を消す。
    （学習時の native 名: feature_fraction / bagging_fraction / bagging_freq）
    """
    model = bundle.get("model")
    if model is None or not hasattr(model, "set_params"):
        return
    try:
        booster_params = getattr(model, "booster_", None)
        native = booster_params.params if booster_params is not None else {}
        updates: dict = {}
        if "feature_fraction" in native:
            updates["colsample_bytree"] = float(native["feature_fraction"])
        if "bagging_fraction" in native:
            updates["subsample"] = float(native["bagging_fraction"])
        if "bagging_freq" in native:
            updates["subsample_freq"] = int(native["bagging_freq"])
        if updates:
            model.set_params(**updates)
    except Exception as e:
        logger.debug("could not silence lightgbm warnings: %s", e)


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


def _fetch_odds_robust(jcd: str, race_no: int, target: date,
                       max_attempts: int = 6) -> tuple[Optional[dict], Optional[str]]:
    """単勝オッズを最大 max_attempts 回まで指数バックオフでリトライ取得。

    戻り値: ({lane: odds} dict, reason_when_none)
      - 成功: ({1: 4.2, 2: ...}, None)
      - 中止/休場（リトライ無意味）: (None, "no_data")
      - 全試行失敗: (None, "failed_after_N_attempts")
    """
    _, od = _scrapers()
    last_err = "unknown"
    for attempt in range(1, max_attempts + 1):
        try:
            wo = od.fetch_win_odds(jcd, race_no, target,
                                   max_retries=1, read_timeout=8.0)
        except Exception as e:
            last_err = f"{type(e).__name__}"
        else:
            if not wo.odds:
                # データなし: 中止/休場/開催前。リトライしても変わらない。
                return None, "no_data"
            if len(wo.odds) >= 6:
                return wo.odds, None
            last_err = f"partial_{len(wo.odds)}"
        if attempt < max_attempts:
            backoff = min(2 ** (attempt - 1), 8)
            time.sleep(backoff)
    logger.warning("odds FINAL FAIL jcd=%s rno=%s d=%s after %d attempts (last: %s)",
                   jcd, race_no, target, max_attempts, last_err)
    return None, f"failed_{last_err}"


def _process_race(
    jcd: str, race_no: int, target: date, bundle: dict,
    blend_alpha: float, takeout: float,
    max_odds: Optional[float] = None,
) -> Optional[dict]:
    br, _ = _scrapers()
    # 先に odds を取得：max_odds で弾くレースは racecard を取らずに早期スキップ。
    odds_map, _reason = _fetch_odds_robust(jcd, race_no, target)
    if odds_map is None:
        return None
    if max_odds is not None:
        lane1_odds = odds_map.get(1)
        if lane1_odds is None or lane1_odds > max_odds:
            return None

    try:
        card = br.fetch_race_card(jcd, race_no, target,
                                  max_retries=3, read_timeout=8.0)
    except Exception as e:
        logger.warning("racelist FINAL FAIL jcd=%s rno=%s d=%s: %s",
                       jcd, race_no, target, type(e).__name__)
        return None
    empty = RaceResult(race_date=card.race_date, venue_code=card.venue_code, race_no=card.race_no)
    df = build_dataset([(card, empty)])
    feats = build_features(df)
    pred = _predict_with_bundle(feats, bundle)

    pred = pred.copy()
    pred["odds_win"] = pred["lane"].map(odds_map).astype(float)
    if pred["odds_win"].isna().any():
        return None
    blended = add_blended_probability(pred, alpha=blend_alpha, takeout=takeout)
    pred["blended_win_prob"] = blended["blended_win_prob"].values

    lane1 = pred[pred["lane"] == 1].iloc[0]
    # racer name フォールバック: name が空なら racer_id を表示用に使う
    racer_name = str(lane1.get("name", "") or "").strip()
    if not racer_name:
        rid = str(lane1.get("racer_id", "") or "").strip()
        racer_name = f"#{rid}" if rid else ""
    return {
        "venue": jcd,
        "venue_name": VENUE_CODES.get(jcd, "?"),
        "race_no": race_no,
        "lane": 1,
        "racer": racer_name,
        "p_model": float(lane1["pred_win_prob"]),
        "p_blend": float(lane1["blended_win_prob"]),
        "odds_win": float(lane1["odds_win"]),
        "ev": float(lane1["blended_win_prob"]) * float(lane1["odds_win"]),
    }


def _fetch_result(jcd: str, race_no: int, target: date,
                  max_attempts: int = 4) -> tuple[Optional[dict], Optional[str]]:
    """レース結果を最大 max_attempts 回まで指数バックオフで取得。

    戻り値: (result_dict or None, reason_when_none)
      reason: "no_rows"（未開催/結果未公開・リトライ無意味）
             / "no_winner"（全艇失格レアケース）
             / "failed_after_N_attempts"（HTTP/timeout）
    """
    br = _result_scraper()
    last_err = "unknown"
    for attempt in range(1, max_attempts + 1):
        try:
            result = br.fetch_race_result(jcd, race_no, target,
                                          max_retries=1, read_timeout=10.0)
        except Exception as e:
            last_err = f"{type(e).__name__}"
            if attempt < max_attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
            continue

        if not result.rows:
            # 未開催 or 結果未公開: リトライしても変わらない
            return None, "no_rows"
        winner = next((r for r in result.rows if r.rank == 1), None)
        if winner is None:
            return None, "no_winner"
        win_payout = None
        for combo, amt in result.payouts.get("win", []):
            try:
                if int(combo) == winner.lane:
                    win_payout = amt
                    break
            except (ValueError, TypeError):
                continue
        return {"winner_lane": int(winner.lane), "win_payout_yen": win_payout}, None

    logger.warning("result FINAL FAIL jcd=%s rno=%s d=%s after %d attempts (last: %s)",
                   jcd, race_no, target, max_attempts, last_err)
    return None, f"failed_{last_err}"


def _attach_results(rows: list[dict], target: date, workers: int) -> dict:
    """各シグナル行にレース結果（あれば）を in-place で付与。

    戻り値: 失敗内訳のカウント {fetch_failed: N, no_rows: N, ...}
    """
    def task(idx_row: tuple[int, dict]) -> tuple[int, Optional[dict], Optional[str]]:
        idx, r = idx_row
        res, reason = _fetch_result(r["venue"], r["race_no"], target)
        return idx, res, reason

    failures: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for idx, res, reason in ex.map(task, list(enumerate(rows))):
            r = rows[idx]
            if res is None:
                r["result_status"] = "未確定"
                r["result_reason"] = reason
                r["actual_pnl"] = None
                r["actual_return"] = None
                r["winner_lane"] = None
                failures[reason or "unknown"] = failures.get(reason or "unknown", 0) + 1
                continue
            wl = res["winner_lane"]
            payout = res["win_payout_yen"]
            stake = r["stake_yen"]
            hit = (wl == 1)
            if hit and payout is not None:
                ret = stake * payout // 100
                r["result_status"] = "hit"
                r["actual_return"] = int(ret)
                r["actual_pnl"] = int(ret - stake)
            elif hit and payout is None:
                r["result_status"] = "hit_no_payout"
                r["actual_return"] = None
                r["actual_pnl"] = None
            else:
                r["result_status"] = "miss"
                r["actual_return"] = 0
                r["actual_pnl"] = -stake
            r["winner_lane"] = wl
    return failures


class MaintenanceMode(RuntimeError):
    """boatrace.jp がシステムメンテナンス中。これ以上のリクエストは無意味。"""


# プロセスで一度メンテ検出したら全スレッドの再試行を即座に止めるためのフラグ
_MAINTENANCE_DETECTED = False


def _check_maintenance(html: str) -> bool:
    """レスポンス本文にメンテナンス告知が含まれているか。

    boatrace.jp は 22:00〜翌朝にメンテすると、どの URL でも「システムメンテナンス」を
    含む案内ページを返してくる。タイムアウトより前に検出できるケースもある。
    """
    return ("システムメンテナンス" in html) or ("ご利用になれません" in html)


def _scan_targets(target: date, venues: list[str], workers: int,
                  max_attempts: int = 4) -> list[tuple[str, int]]:
    """各場の開催レース番号を並列に取得。失敗した場は最大 max_attempts まで再試行。

    メンテナンス検出 or 全場初回失敗の時点で早期に MaintenanceMode を投げて呼び出し側を
    短絡させる（無駄なリトライで数十秒消費しないため）。
    """
    global _MAINTENANCE_DETECTED
    out: list[tuple[str, int]] = []

    def task(jcd: str) -> tuple[str, list[int]]:
        global _MAINTENANCE_DETECTED
        br, _ = _scrapers()
        last_err = "unknown"
        for attempt in range(1, max_attempts + 1):
            if _MAINTENANCE_DETECTED:
                return jcd, []
            try:
                rnos = br.fetch_race_index(jcd, target)
                return jcd, rnos
            except Exception as e:
                last_err = f"{type(e).__name__}"
                # HTTPエラー本文を見られる場合はメンテ判定
                resp_text = getattr(getattr(e, "response", None), "text", "") or ""
                if _check_maintenance(resp_text):
                    _MAINTENANCE_DETECTED = True
                    return jcd, []
                if attempt < max_attempts:
                    time.sleep(min(2 ** (attempt - 1), 8))
        logger.warning("raceindex FINAL FAIL jcd=%s after %d attempts (last: %s)",
                       jcd, max_attempts, last_err)
        return jcd, []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(task, venues))

    # メンテ確定はレスポンス本文で「システムメンテナンス」を検出した時のみ。
    # 全タイムアウトの場合はサーバー過負荷 or ネットワーク問題なので、別メッセージ。
    if _MAINTENANCE_DETECTED:
        raise MaintenanceMode("boatrace.jp 公式がシステムメンテナンス中")
    failed = [jcd for jcd, rnos in results if not rnos]
    if len(failed) == len(venues) and len(venues) >= 3:
        raise MaintenanceMode(
            f"boatrace.jp 全 {len(venues)} 場の応答無し（タイムアウト連発）。"
            "サイト応答が遅いか、ネットワーク不安定の可能性。--workers を下げるか時間を空けて再試行。"
        )

    for jcd, rnos in results:
        for r in rnos:
            out.append((jcd, r))
    return out


SIGNAL_HTML_TEMPLATE = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>Signals {date}</title>
<style>
  :root {{
    --bg: #0f1117;
    --bg-card: #1a1d27;
    --bg-header: #1f2330;
    --bg-hover: #252937;
    --border: #2a2f3d;
    --text: #e4e6eb;
    --text-dim: #9aa0b0;
    --pos: #4ade80;
    --neg: #f87171;
    --warn: #fbbf24;
    --link: #60a5fa;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", "Hiragino Sans", "Yu Gothic", sans-serif;
    max-width: 1200px; margin: 0 auto; padding: 24px 16px 60px;
    background: var(--bg); color: var(--text); font-size: 14px;
  }}
  h1 {{ font-size: 24px; margin: 0 0 6px; font-weight: 700; }}
  h2 {{ font-size: 17px; margin: 32px 0 12px; padding-bottom: 6px; border-bottom: 1px solid var(--border); color: var(--text-dim); font-weight: 600; }}
  .meta {{ color: var(--text-dim); font-size: 12px; }}
  .meta code {{ background: var(--bg-card); padding: 2px 6px; border-radius: 4px; }}
  .kpi {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 20px 0 8px; }}
  .kpi .card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }}
  .kpi .label {{ color: var(--text-dim); font-size: 12px; letter-spacing: 0.02em; }}
  .kpi .value {{ font-size: 24px; font-weight: 700; margin-top: 6px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 8px; background: var(--bg-card); border-radius: 10px; overflow: hidden; }}
  th, td {{ text-align: right; padding: 10px 12px; border-bottom: 1px solid var(--border); }}
  th:nth-child(-n+3), td:nth-child(-n+3) {{ text-align: left; }}
  th:last-child, td:last-child {{ text-align: center; }}
  th {{ background: var(--bg-header); cursor: pointer; user-select: none; font-size: 12px; color: var(--text-dim); font-weight: 600; letter-spacing: 0.03em; }}
  th:hover {{ background: var(--bg-hover); color: var(--text); }}
  tbody tr:last-child td {{ border-bottom: none; }}
  tbody tr:hover td {{ background: var(--bg-hover); }}
  .pos {{ color: var(--pos); font-weight: 600; }}
  .neg {{ color: var(--neg); font-weight: 600; }}
  .ev-high {{ color: var(--pos); font-weight: 700; }}
  .ev-mid {{ color: var(--warn); font-weight: 600; }}
  .pending {{ color: var(--text-dim); }}
  a {{ color: var(--link); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; }}
  .badge-hit {{ background: rgba(74,222,128,0.15); color: var(--pos); }}
  .badge-miss {{ background: rgba(248,113,113,0.15); color: var(--neg); }}
  .badge-pending {{ background: rgba(154,160,176,0.12); color: var(--text-dim); }}
  .totals-row td {{ background: var(--bg-header); font-weight: 700; border-top: 2px solid var(--border); }}
</style>
</head>
<body>

<h1>当日シグナル: {date}</h1>
<div class="meta">
  generated: {generated} &nbsp;|&nbsp; strategy: <code>lane1_kelly</code> &nbsp;|&nbsp; ev_threshold: <code>{ev_threshold}</code> &nbsp;|&nbsp; kelly_fraction: <code>{kelly_fraction}</code>
</div>

<div class="kpi">
  <div class="card"><div class="label">推奨ベット</div><div class="value">{n_bets}件</div></div>
  <div class="card"><div class="label">合計ステーク</div><div class="value">¥{total_stake}</div></div>
  <div class="card"><div class="label">最高EV</div><div class="value">{max_ev}</div></div>
  <div class="card"><div class="label">平均オッズ</div><div class="value">{avg_odds}</div></div>
  {result_kpis}
</div>

<h2>推奨ベット一覧（列ヘッダクリックで並べ替え）</h2>
<table id="signals">
  <thead><tr>
    <th>場</th><th>R</th><th>選手(1号艇)</th>
    <th>p_model</th><th>p_blend</th><th>オッズ</th><th>EV</th><th>ステーク</th>
    <th>結果</th><th>PnL</th><th>リンク</th>
  </tr></thead>
  <tbody>{rows}{totals_row}</tbody>
</table>

<script>
document.querySelectorAll('#signals th').forEach((th, idx) => {{
  let asc = false;
  th.addEventListener('click', () => {{
    const tbody = th.closest('table').querySelector('tbody');
    const allRows = Array.from(tbody.querySelectorAll('tr'));
    // totals 行（class=totals-row）は固定で末尾に残す
    const totals = allRows.filter(r => r.classList.contains('totals-row'));
    const rows = allRows.filter(r => !r.classList.contains('totals-row'));
    rows.sort((a, b) => {{
      const av = a.children[idx].dataset.sort ?? a.children[idx].textContent;
      const bv = b.children[idx].dataset.sort ?? b.children[idx].textContent;
      const an = parseFloat(av), bn = parseFloat(bv);
      const cmp = (!isNaN(an) && !isNaN(bn)) ? an - bn : String(av).localeCompare(String(bv), 'ja');
      return asc ? cmp : -cmp;
    }});
    asc = !asc;
    rows.forEach(r => tbody.appendChild(r));
    totals.forEach(r => tbody.appendChild(r));
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


def _result_cell(r: dict) -> str:
    """結果セル（バッジ表記）。"""
    status = r.get("result_status")
    if status == "hit":
        wl = r.get("winner_lane", 1)
        return f"<td data-sort='2'><span class='badge badge-hit'>🟢 1着 (1号艇)</span></td>"
    if status == "hit_no_payout":
        return f"<td data-sort='2'><span class='badge badge-hit'>🟢 1着</span></td>"
    if status == "miss":
        wl = r.get("winner_lane")
        wl_txt = f"{wl}号艇1着" if wl else "不的中"
        return f"<td data-sort='0'><span class='badge badge-miss'>✕ {wl_txt}</span></td>"
    return f"<td data-sort='1'><span class='badge badge-pending'>未確定</span></td>"


def _pnl_cell(r: dict) -> str:
    pnl = r.get("actual_pnl")
    if pnl is None:
        return f"<td class='pending' data-sort='0'>-</td>"
    cls = "pos" if pnl > 0 else ("neg" if pnl < 0 else "")
    sign = "+" if pnl > 0 else ""
    ret = r.get("actual_return")
    ret_txt = f" (返¥{ret:,})" if ret is not None and pnl > 0 else ""
    return f"<td class='{cls}' data-sort='{pnl}'>{sign}¥{pnl:,}{ret_txt}</td>"


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
        f"{_result_cell(r)}"
        f"{_pnl_cell(r)}"
        f"<td><a href='{url}' target='_blank'>公式</a></td>"
        f"</tr>"
    )


def _totals_row(rows: list[dict]) -> str:
    """確定済みベットの集計行（全件未確定なら空文字）。"""
    decided = [r for r in rows if r.get("actual_pnl") is not None]
    if not decided:
        return ""
    total_stake = sum(r["stake_yen"] for r in decided)
    total_pnl = sum(r["actual_pnl"] for r in decided)
    total_ret = total_stake + total_pnl
    n_hit = sum(1 for r in decided if r["result_status"] in ("hit", "hit_no_payout"))
    cls = "pos" if total_pnl > 0 else ("neg" if total_pnl < 0 else "")
    sign = "+" if total_pnl > 0 else ""
    return (
        f"<tr class='totals-row'>"
        f"<td colspan='7'>確定済み合計（{len(decided)}件 / 的中 {n_hit}件・勝率 {n_hit/len(decided)*100:.0f}%）</td>"
        f"<td>¥{total_stake:,}</td>"
        f"<td></td>"
        f"<td class='{cls}'>{sign}¥{total_pnl:,} (返¥{total_ret:,})</td>"
        f"<td></td>"
        f"</tr>"
    )


def _result_kpis(rows: list[dict]) -> str:
    """確定済みベットがあれば、勝率・実PnLの KPI カードを追加。"""
    decided = [r for r in rows if r.get("actual_pnl") is not None]
    if not decided:
        return ""
    n_hit = sum(1 for r in decided if r["result_status"] in ("hit", "hit_no_payout"))
    total_pnl = sum(r["actual_pnl"] for r in decided)
    total_stake = sum(r["stake_yen"] for r in decided)
    roi = (total_pnl / total_stake * 100) if total_stake else 0.0
    pnl_cls = "pos" if total_pnl > 0 else ("neg" if total_pnl < 0 else "")
    sign = "+" if total_pnl > 0 else ""
    return (
        f"<div class='card'><div class='label'>確定済み</div>"
        f"<div class='value'>{len(decided)}/{len(rows)}件</div></div>"
        f"<div class='card'><div class='label'>実勝率</div>"
        f"<div class='value'>{n_hit/len(decided)*100:.1f}%</div></div>"
        f"<div class='card'><div class='label'>実PnL</div>"
        f"<div class='value {pnl_cls}'>{sign}¥{total_pnl:,}</div></div>"
        f"<div class='card'><div class='label'>実ROI</div>"
        f"<div class='value {pnl_cls}'>{sign}{roi:.1f}%</div></div>"
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
        result_kpis=_result_kpis(rows),
        rows="".join(_signal_row(r, date_str) for r in rows),
        totals_row=_totals_row(rows),
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
    p.add_argument("--no-results", action="store_true",
                   help="結果取得を無効化（デフォルトはON。終わったレースは結果とPnL、未開催は『未確定』）")
    args = p.parse_args()

    target = date.fromisoformat(args.date) if args.date else date.today()
    all_codes = list(VENUE_CODES.keys())
    venues = [v for v in (args.venues or all_codes) if v not in set(args.exclude_venues)]
    print(f"対象日={target} 対象場={','.join(venues)} workers={args.workers}")

    bundle = joblib.load(MODELS_DIR / "lgb_win_model.joblib")
    _silence_lightgbm(bundle)  # LightGBM の alias 衝突警告を抑止（高速化＆ログ削減）

    # ① 各場の開催レース番号
    try:
        targets = _scan_targets(target, venues, args.workers)
    except MaintenanceMode as e:
        print()
        print("=" * 60)
        print("⚠ boatrace.jp はシステムメンテナンス中です")
        print("=" * 60)
        print("公式の定例メンテは概ね 22:00〜翌朝6:30。再開後に再実行してください。")
        print(f"詳細: {e}")
        return
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
                      args.blend_alpha, args.takeout, args.max_odds): (jcd, rno)
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

    # 結果取得: デフォルト ON（未開催レースは自動で「未確定」表示）。--no-results で抑止。
    if not args.no_results:
        print(f"\n推奨ベット {len(rows)}件のレース結果を取得中（未開催分は『未確定』）...")
        failures = _attach_results(rows, target, args.workers)
        decided = [r for r in rows if r.get("actual_pnl") is not None]
        if decided:
            n_hit = sum(1 for r in decided if r["result_status"] in ("hit", "hit_no_payout"))
            total_pnl = sum(r["actual_pnl"] for r in decided)
            total_stake = sum(r["stake_yen"] for r in decided)
            print(f"確定済み: {len(decided)}/{len(rows)}件 | 的中 {n_hit}件 "
                  f"| 実PnL ¥{total_pnl:+,} (ステーク ¥{total_stake:,})")
        else:
            print(f"確定済み: 0/{len(rows)}件（全て未開催/未確定）")
        if failures:
            print(f"結果取得失敗内訳: {failures}（fetch_failed=HTTP/timeout, no_rows=未公開/未開催）")

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
