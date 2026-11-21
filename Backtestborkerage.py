import base64
import copy
import hashlib
import html
import io
import json
import math
import os
import re
import shutil
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from datetime import time as dtime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import requests
import streamlit as st

IST = "Asia/Kolkata"
MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
CANDLE_KEYS = ("prev_O", "prev_H", "prev_L", "prev_C", "sig_O", "sig_H", "sig_L", "sig_C")
STORE_KEYS = CANDLE_KEYS + ("hist",)          # hist = last 5 candles [O,H,L,C] (oldest -> newest) saved with every entry


# ============================== 10 CANDLE PATTERNS (candles.py rules - "Candlestick_Patterns_Logic_Comparison.docx") ==============================
# Sheet columns: B=Open, C=High, D=Low, E=Close.  Each function gets the pattern's candles OLDEST -> NEWEST
# (k[0] = first / "row 2" candle ... k[-1] = latest closed candle) and returns True/False (Series for a whole DataFrame).
def _body(k):
    return (k.c - k.o).abs()


def _rng(k):
    return k.h - k.l


def _upsh(k):           # upper shadow
    return k.h - np.maximum(k.o, k.c)


def _losh(k):           # lower shadow
    return np.minimum(k.o, k.c) - k.l


# candles.py thresholds
TWEEZER_LOW_TOL = 0.001     # two lows count as "the same" when within 0.1%
DOWNTREND_LOOKBACK = 5      # "in a downtrend": close of the checked candle < close this many candles earlier
# downtrend(i) of candles.py: which candle's close is checked, counted back from the LAST candle of the pattern (0 = last candle)
#   Bullish Engulfing -> downtrend(i-1) = its 1st candle | Hammer -> downtrend(i) = the hammer itself | Inverted Hammer -> downtrend(i) = its 1st candle
DOWNTREND_AT = {"Bullish Engulfing": 1, "Hammer": 0, "Inverted Hammer": 1}
DOWNTREND_PATTERNS = tuple(DOWNTREND_AT)


def _p_engulf(k):       # bearish candle whose body is fully covered by the next bullish candle (downtrend filter applied in detect_patterns)
    a, b = k
    return (a.o > a.c) & (b.o < b.c) & (b.o <= a.c) & (b.c >= a.o)


def _p_harami(k):       # big bearish candle (body >= 50% of its range), bullish body completely inside it
    a, b = k
    return (a.o > a.c) & (b.o < b.c) & (_rng(a) > 0) & (_body(a) >= 0.50 * _rng(a)) & (b.o > a.c) & (b.c < a.o)


def _p_marubozu(k):     # bullish, body >= 90% of range, both shadows <= 5% of range
    (a,) = k
    r = _rng(a)
    return (a.c > a.o) & (r > 0) & (_body(a) >= 0.90 * r) & (_upsh(a) <= 0.05 * r) & (_losh(a) <= 0.05 * r)


def _p_hammer(k):       # body <= 35% of range, lower shadow >= 2x body, upper shadow <= 15% of range (downtrend filter applied in detect_patterns)
    (a,) = k
    r = _rng(a)
    return (r > 0) & (_body(a) <= 0.35 * r) & (_losh(a) >= 2 * _body(a)) & (_upsh(a) <= 0.15 * r)


def _p_inv_hammer(k):   # 1st: body <= 35% of range, upper shadow >= 2x body, lower shadow <= 15% of range (+ downtrend) / 2nd: bullish and closes above the 1st close
    a, b = k
    r = _rng(a)
    return (r > 0) & (_body(a) <= 0.35 * r) & (_upsh(a) >= 2 * _body(a)) & (_losh(a) <= 0.15 * r) & (b.c > b.o) & (b.c > a.c)


def _p_morning(k):      # 1st bearish, body >= 50% of range / 2nd body <= 35% of range with its HIGH below the 1st close / 3rd bullish, closes above the 1st body midpoint
    a, b, c = k
    return ((a.c < a.o) & (_rng(a) > 0) & (_body(a) >= 0.50 * _rng(a))
            & (_rng(b) > 0) & (_body(b) <= 0.35 * _rng(b)) & (b.h < a.c)
            & (c.c > c.o) & (c.c > (a.o + a.c) / 2))


def _p_piercing(k):     # bullish candle opens below the previous low, closes above the middle of the prior bearish body (same in both files)
    a, b = k
    return (a.c < a.o) & (b.c > b.o) & (b.o < a.l) & (b.c > a.o - (a.o - a.c) / 2) & (b.c < a.o)


def _p_rising3(k):      # 1st bullish (body >= 40% of range), 3 bearish candles with smaller bodies INSIDE its high-low range, 5th bullish closes above the 1st HIGH
    a, b, c, d, e = k
    ok = (a.c > a.o) & (_rng(a) > 0) & (_body(a) >= 0.40 * _rng(a))
    for m in (b, c, d):
        ok = ok & (m.c < m.o) & (_body(m) <= _body(a)) & (np.maximum(m.o, m.c) <= a.h) & (np.minimum(m.o, m.c) >= a.l)
    return ok & (e.c > e.o) & (e.c > a.h)


def _p_soldiers(k):     # 3 bullish candles, rising closes, EXACT open check (no gap-down), upper wick <= 30% of range, body >= 20% of range
    a, b, c = k
    ok = (b.c > a.c) & (c.c > b.c) & (b.o >= a.o) & (c.o >= b.o)
    for m in (a, b, c):
        r = _rng(m)
        ok = ok & (m.c > m.o) & (r > 0) & (_upsh(m) <= 0.30 * r) & (_body(m) >= 0.20 * r)
    return ok


def _p_tweezer(k):      # two candles share the same low (0.1%), 3rd candle bullish AND closes above the 2nd candle's HIGH
    a, b, c = k
    return ((b.l - a.l).abs() <= TWEEZER_LOW_TOL * a.l) & (c.c > c.o) & (c.c > b.h)


# (name, candles used, python check, Excel/Sheets formula builder: rows -> formula)   Excel columns: B=Open C=High D=Low E=Close
# (the downtrend condition for Engulfing / Hammer / Inverted Hammer is applied in Python only - see DOWNTREND_AT)
PATTERN_DEFS = (
    ("Bullish Engulfing", 2, _p_engulf,
     lambda r: f'=IF(AND(B{r[0]}>E{r[0]},B{r[1]}<E{r[1]},B{r[1]}<=E{r[0]},E{r[1]}>=B{r[0]}),"BULLISH ENGULFING ","")'),
    ("Bullish Harami", 2, _p_harami,
     lambda r: f'=IF(AND(B{r[0]}>E{r[0]},B{r[1]}<E{r[1]},C{r[0]}>D{r[0]},(B{r[0]}-E{r[0]})>=0.5*(C{r[0]}-D{r[0]}),B{r[1]}>E{r[0]},E{r[1]}<B{r[0]}),"BULLISH harami","")'),
    ("Bullish Marubozu", 1, _p_marubozu,
     lambda r: f'=IF(AND(E{r[0]}>B{r[0]},C{r[0]}>D{r[0]},(E{r[0]}-B{r[0]})>=0.9*(C{r[0]}-D{r[0]}),(C{r[0]}-E{r[0]})<=0.05*(C{r[0]}-D{r[0]}),(B{r[0]}-D{r[0]})<=0.05*(C{r[0]}-D{r[0]})),"Bullish Marubozu","")'),
    ("Hammer", 1, _p_hammer,
     lambda r: f'=IF(AND(C{r[0]}>D{r[0]},ABS(E{r[0]}-B{r[0]})<=0.35*(C{r[0]}-D{r[0]}),(MIN(B{r[0]},E{r[0]})-D{r[0]})>=2*ABS(E{r[0]}-B{r[0]}),(C{r[0]}-MAX(B{r[0]},E{r[0]}))<=0.15*(C{r[0]}-D{r[0]})),"HAMMER","")'),
    ("Inverted Hammer", 2, _p_inv_hammer,
     lambda r: f'=IF(AND(C{r[0]}>D{r[0]},ABS(E{r[0]}-B{r[0]})<=0.35*(C{r[0]}-D{r[0]}),(C{r[0]}-MAX(B{r[0]},E{r[0]}))>=2*ABS(E{r[0]}-B{r[0]}),(MIN(B{r[0]},E{r[0]})-D{r[0]})<=0.15*(C{r[0]}-D{r[0]}),E{r[1]}>B{r[1]},E{r[1]}>E{r[0]}),"INVERTED HAMMER","")'),
    ("Morning Star", 3, _p_morning,
     lambda r: f'=IF(AND(E{r[0]}<B{r[0]},C{r[0]}>D{r[0]},(B{r[0]}-E{r[0]})>=0.5*(C{r[0]}-D{r[0]}),C{r[1]}>D{r[1]},ABS(E{r[1]}-B{r[1]})<=0.35*(C{r[1]}-D{r[1]}),C{r[1]}<E{r[0]},E{r[2]}>B{r[2]},E{r[2]}>(B{r[0]}+E{r[0]})/2),"MORNING STAR","")'),
    ("Piercing", 2, _p_piercing,
     lambda r: f'=IF(AND(E{r[0]}<B{r[0]},E{r[1]}>B{r[1]},B{r[1]}<D{r[0]},E{r[1]}>B{r[0]}-(B{r[0]}-E{r[0]})/2,E{r[1]}<B{r[0]}),"PIERCING","")'),
    ("Rising Three Methods", 5, _p_rising3,
     lambda r: (f'=IF(AND(E{r[0]}>B{r[0]},C{r[0]}>D{r[0]},(E{r[0]}-B{r[0]})>=0.4*(C{r[0]}-D{r[0]}),'
                + ','.join(f'E{x}<B{x},ABS(E{x}-B{x})<=ABS(E{r[0]}-B{r[0]}),MAX(B{x},E{x})<=C{r[0]},MIN(B{x},E{x})>=D{r[0]}' for x in r[1:4])
                + f',E{r[4]}>B{r[4]},E{r[4]}>C{r[0]}),"RISING THREE METHODS","")')),
    ("Three White Soldiers", 3, _p_soldiers,
     lambda r: ('=IF(AND(' + ','.join(
         f'E{x}>B{x},C{x}>D{x},(C{x}-E{x})<=0.3*(C{x}-D{x}),(E{x}-B{x})>=0.2*(C{x}-D{x})' for x in r)
         + f',E{r[1]}>E{r[0]},E{r[2]}>E{r[1]},B{r[1]}>=B{r[0]},B{r[2]}>=B{r[1]}),"THREE WHITE SOLDIERS","")')),
    ("Tweezer Bottom", 3, _p_tweezer,
     lambda r: f'=IF(AND(ABS(D{r[1]}-D{r[0]})<={TWEEZER_LOW_TOL}*D{r[0]},E{r[2]}>B{r[2]},E{r[2]}>C{r[1]}),"TWEEZER BOTTOM","")'),
)
PATTERNS = tuple(d[0] for d in PATTERN_DEFS)

SL_PCT = 0.0
MAX_HOLD_DAYS = 1
TREND_LOOKBACK = 0
TWEEZER_TOL_PCT = 0.10
FUTURES_MARGIN_PCT = 10.0
BROKERAGE_PCT = 0.0
BROKERAGE_BUY_PCT = 0.5     # brokerage on the BUY leg  = buy value  x this %
BROKERAGE_SELL_PCT = 0.5    # brokerage on the SELL leg = sell value x this %
SLIPPAGE_PCT = 0.0
FRESH_SECONDS = 120
REFRESH_SECONDS = 5

TIMEFRAMES = {
    "1 minute": ("ONE_MINUTE", 1),
    "3 minutes": ("THREE_MINUTE", 3),
    "5 minutes": ("FIVE_MINUTE", 5),
    "10 minutes": ("TEN_MINUTE", 10),
    "15 minutes": ("FIFTEEN_MINUTE", 15),
    "30 minutes": ("THIRTY_MINUTE", 30),
    "1 hour": ("ONE_HOUR", 60),
    "4 hours": ("FOUR_HOUR", 240),     # built from 1-hour candles (Angel One has no 4H interval)
    "1 day": ("ONE_DAY", 1440),
}

OWNER_ID = "owner"
# Absolute data root so saves survive cwd changes on Streamlit Cloud.
# Override with env DATA_DIR if needed. App reboot on free Cloud can still wipe disk.
_DATA_DIR = os.environ.get("DATA_DIR") or os.path.join(os.path.expanduser("~"), "paper_trading_data")
STRATEGIES_ROOT = os.path.join(_DATA_DIR, "strategies")
os.makedirs(STRATEGIES_ROOT, exist_ok=True)


# ============================== CONFIG ==============================
@dataclass
class Config:
    capital: float = 100000.0
    max_open: int = 5
    target_pct: float = 1.0
    sl_pct: float = SL_PCT
    max_hold: int = MAX_HOLD_DAYS
    tweezer_tol_pct: float = TWEEZER_TOL_PCT
    trend_lookback: int = TREND_LOOKBACK
    brokerage_pct: float = BROKERAGE_PCT
    brokerage_buy_pct: float = BROKERAGE_BUY_PCT      # % of buy value (entry x qty) charged on the buy order
    brokerage_sell_pct: float = BROKERAGE_SELL_PCT    # % of sell value (exit x qty) charged on the sell order
    brokerage_on: bool = True                       # False = no brokerage at all (switch in Results)
    brokerage_mode: str = "pct"                      # "pct" = % of trade value | "flat" = rupees per order
    brokerage_buy_flat: float = 0.0                 # flat mode: rupees charged on every buy order
    brokerage_sell_flat: float = 0.0                # flat mode: rupees charged on every sell order
    compound: bool = False                          # True = every new position is sized from CURRENT equity (profits re-invested)
    slippage_pct: float = SLIPPAGE_PCT
    interval: str = "FIVE_MINUTE"
    tf_min: int = 5
    patterns: tuple = PATTERNS
    position_amount: float = 0.0
    size_mode: str = "amount"           # "amount" | "qty" | "equal" (equal weight)
    position_qty: int = 0               # used when size_mode == "qty" (shares; lots for NFO/MCX)
    n_stocks: int = 0                   # selected stocks - equal weight divides capital by this when no max-position limit
    no_limit: bool = False
    cloud_filter: str = ""              # optional Ichimoku cloud filter: "" (off) | "Above" | "Center" | "Below"
    cloud_exit: str = ""                # optional Ichimoku cloud EXIT: "" (off) | "Above" | "Center" | "Below" (comma separated for several)
    cloud_exit_mode: str = "OR"         # how Target and Cloud exit combine: "OR" (first one hit exits) | "AND" (both must be satisfied)
    entry_times: str = ""               # OPTIONAL time condition: comma separated candle times "HH:MM,HH:MM" ("" = all times)
    exit_at: str = ""                   # TIME CONDITION: square-off time (HH:MM) ("" = default 15:15 NSE / 23:15 MCX)

    @property
    def per_position(self):
        if self.size_mode == "equal":
            n = max(self.n_stocks, 1) if self.no_limit else max(self.max_open, 1)
            return self.capital / n
        return self.position_amount if self.position_amount > 0 else self.capital / self.max_open

    @property
    def size_text(self):
        if self.size_mode == "qty":
            return f"qty/position {int(self.position_qty)}"
        return ("equal weight " if self.size_mode == "equal" else "amount/position ") + f"{self.per_position:,.0f}"

    def size_qty(self, one_lot, lot, cash, equity=None):
        """(qty the sizing rule alone allows, qty after the free-cash cap) - in shares (lots x lot size for NFO/MCX).
           NORMAL   : the size of every position comes from the STARTING capital (fixed).
           COMPOUND : the amount / qty is scaled by (current equity / starting capital) - profits are re-invested,
                      losses make the next positions smaller. Qty mode: the qty is scaled the same way and rounded down."""
        f = 1.0
        if self.compound and equity is not None and self.capital > 0:
            f = max(float(equity), 0.0) / self.capital
        if self.size_mode == "qty":
            n = max(int(math.floor(self.position_qty * f + 1e-9)), 0)
            n_cash = math.floor(max(cash, 0) / one_lot) if one_lot > 0 else 0
            return n * lot, min(n, n_cash) * lot
        per = self.per_position * f
        return (math.floor(per / one_lot) * lot,
                math.floor(min(per, cash) / one_lot) * lot)

def _cfg_from_data(d):
    """Config from a saved config.json dict (paper and live files)."""
    return Config(
        capital=float(d.get("capital", 0) or 0),
        max_open=int(d.get("max_open", 5)),
        target_pct=float(d.get("target_pct", 1.0)),
        sl_pct=float(d.get("sl_pct", 0) or 0),
        interval=d.get("interval", "FIVE_MINUTE"),
        tf_min=int(d.get("tf_min", 5)),
        position_amount=float(d.get("position_amount") or 0),
        size_mode=str(d.get("size_mode") or "amount"),
        position_qty=int(d.get("position_qty") or 0),
        n_stocks=int(d.get("n_stocks") or len(d.get("items") or [])),
        no_limit=bool(d.get("no_limit", False)),
        cloud_filter=cloud_text(d.get("cloud_filter")),
        cloud_exit=cloud_text(d.get("cloud_exit")),
        brokerage_buy_pct=float(d.get("brokerage_buy_pct", BROKERAGE_BUY_PCT) or 0),
        brokerage_sell_pct=float(d.get("brokerage_sell_pct", BROKERAGE_SELL_PCT) or 0),
        brokerage_on=bool(d.get("brokerage_on", True)),
        brokerage_mode="flat" if str(d.get("brokerage_mode") or "").lower() == "flat" else "pct",
        brokerage_buy_flat=float(d.get("brokerage_buy_flat") or 0),
        brokerage_sell_flat=float(d.get("brokerage_sell_flat") or 0),
        compound=bool(d.get("compound", False)),
        slippage_pct=float(d.get("slippage_pct") or 0),
        cloud_exit_mode="AND" if str(d.get("cloud_exit_mode") or "").upper() == "AND" else "OR",
        entry_times=str(d.get("entry_times") or ""),
        exit_at=str(d.get("exit_at") or ""),
        max_hold=int(d.get("max_hold", MAX_HOLD_DAYS) or MAX_HOLD_DAYS),
        patterns=tuple(p for p in (d.get("patterns") or PATTERNS) if p in PATTERNS) or PATTERNS,
    )


# ============================== TIME ==============================
def now_ist():
    return pd.Timestamp.now(tz=IST).tz_localize(None)


def market_open(now, exchange="NSE"):
    start, end = (dtime(9, 0), dtime(23, 30)) if exchange == "MCX" else (dtime(9, 15), dtime(15, 30))
    return now.weekday() < 5 and start <= now.time() <= end


def exit_time(exchange):
    return dtime(23, 15) if exchange == "MCX" else dtime(15, 15)


def _hhmm(s):
    """'HH:MM' -> datetime.time (None when empty / invalid)."""
    try:
        h, m = str(s).strip().split(":")[:2]
        return dtime(int(h), int(m))
    except Exception:
        return None


def eff_exit(cfg, exchange):
    """Square-off time actually used: the chosen exit time, else the default exit_time()."""
    return _hhmm(getattr(cfg, "exit_at", "")) or exit_time(exchange)


def tf_time_grid(exchange, tf_min):
    """Every candle time of the timeframe for one session -> (candle START times, candle END times) as 'HH:MM'.
       Entry is taken at the OPEN of a candle (start times); exits happen when a candle closes (end times).
       1-day candles use an hourly grid for the exit."""
    sh, sm, eh, em = (9, 0, 23, 30) if exchange == "MCX" else (9, 15, 15, 30)
    start, close = sh * 60 + sm, eh * 60 + em
    step = int(tf_min) if int(tf_min) < 1440 else 60
    starts, ends, t = [], [], start
    while t < close:
        starts.append(f"{t // 60:02d}:{t % 60:02d}")
        e = min(t + step, close)
        ends.append(f"{e // 60:02d}:{e % 60:02d}")
        t += step
    return starts, ends


def session_done(now):
    return now.weekday() < 5 and now.time() > dtime(15, 30)


# ============================== PATTERNS ==============================
def detect_patterns(df, cfg=None, downtrend=True):
    """True/False per candle for every pattern: the pattern ENDS on that candle (its last candle)."""
    o, h, l, c = (df[x].astype(float) for x in ("open", "high", "low", "close"))

    def candle(sh):
        return SimpleNamespace(o=o.shift(sh), h=h.shift(sh), l=l.shift(sh), c=c.shift(sh))

    sig = pd.DataFrame(
        {name: fn([candle(n - 1 - j) for j in range(n)]) for name, n, fn, _ in PATTERN_DEFS},
        index=df.index,
    )
    lb = int(getattr(cfg, "trend_lookback", 0) or 0)
    if lb > 0:                                              # optional extra filter for ALL patterns (off by default)
        sig = sig.mul(c.shift(1) < c.shift(lb + 1), axis=0).astype(bool)
    for name in (DOWNTREND_PATTERNS if downtrend else ()):  # candles.py: Engulfing / Hammer / Inverted Hammer need "in a downtrend"
        if name in sig.columns:
            s = DOWNTREND_AT[name]                          # close of the checked candle < close DOWNTREND_LOOKBACK candles earlier
            sig[name] = sig[name].astype(bool) & (c.shift(s) < c.shift(s + DOWNTREND_LOOKBACK)).fillna(False)
    return sig.fillna(False).astype(bool)


def classify_candles(r, only=None):
    """Re-check the candles saved with an entry (hist, or the old prev/sig pair) against the formulas."""
    hist = r.get("hist")
    if not isinstance(hist, (list, tuple)) or not hist:
        try:
            hist = [[r["prev_O"], r["prev_H"], r["prev_L"], r["prev_C"]],
                    [r["sig_O"], r["sig_H"], r["sig_L"], r["sig_C"]]]
        except (KeyError, TypeError):
            return None
    try:
        df = pd.DataFrame(hist, columns=["open", "high", "low", "close"]).astype(float)
    except (ValueError, TypeError):
        return None
    if df.empty or df.iloc[-1].isna().any():
        return None
    sig = detect_patterns(df, downtrend=False)              # saved hist is too short for the downtrend check
    names = [n for n in (only or PATTERNS) if n in sig.columns and bool(sig[n].iat[-1])]
    return " + ".join(names) if names else "No pattern"


# ============================== ICHIMOKU CLOUD (optional entry filter) ==============================
CLOUD_OPTIONS = ("Above", "Center", "Below")      # last closed candle: above the cloud / inside the cloud / below the cloud
CLOUD_LABEL = {"Above": "Above cloud", "Center": "Center (inside cloud)", "Below": "Below cloud"}
CLOUD_BY_LABEL = {v: k for k, v in CLOUD_LABEL.items()}


def cloud_list(v):
    """Saved cloud filter -> list of positions. Accepts '' / 'Above' / 'Above,Center' / a list."""
    parts = v if isinstance(v, (list, tuple, set)) else str(v or "").split(",")
    got = {str(x).strip() for x in parts}
    return [k for k in CLOUD_OPTIONS if k in got]


def cloud_text(v):
    """Canonical saved form: '' (filter off) or e.g. 'Above,Center'."""
    return ",".join(cloud_list(v))
CLOUD_MIN_BARS = 78                               # Senkou B needs 52 candles + the cloud is drawn 26 candles ahead
# calendar days of history to load when the cloud filter is ON (so the cloud has >= CLOUD_MIN_BARS candles)
CLOUD_DAYS = {"ONE_MINUTE": 5, "THREE_MINUTE": 5, "FIVE_MINUTE": 6, "TEN_MINUTE": 8, "FIFTEEN_MINUTE": 10,
              "THIRTY_MINUTE": 18, "ONE_HOUR": 35, "FOUR_HOUR": 130, "ONE_DAY": 160}


def ichimoku_cloud(df, tenkan=9, kijun=26, senkou_b=52, shift=26):
    """Standard Ichimoku (9, 26, 52, 26). Row t holds the cloud as DRAWN under candle t
       (Senkou A / B calculated `shift` candles earlier)."""
    h, l = df["high"].astype(float), df["low"].astype(float)

    def mid(n):
        return (h.rolling(n).max() + l.rolling(n).min()) / 2

    span_a = ((mid(tenkan) + mid(kijun)) / 2).shift(shift)
    span_b = mid(senkou_b).shift(shift)
    return pd.DataFrame({"span_a": span_a, "span_b": span_b,
                         "top": np.maximum(span_a, span_b), "bottom": np.minimum(span_a, span_b)}, index=df.index)


def cloud_position(df):
    """'Above' / 'Center' / 'Below' for the LAST candle's close vs the cloud, or None when there are too few candles."""
    if df is None or len(df) < CLOUD_MIN_BARS:
        return None
    last = ichimoku_cloud(df).iloc[-1]
    if pd.isna(last["top"]) or pd.isna(last["bottom"]):
        return None
    close = float(df["close"].iat[-1])
    return "Above" if close > last["top"] else "Below" if close < last["bottom"] else "Center"


def cloud_exit_events(df, cfg, exchange):
    """{candle END time: 'Above'/'Center'/'Below'} for every candle whose CLOSE sits in one of the chosen exit cloud positions.
       Empty dict when the cloud exit is off."""
    sel = cloud_list(cfg.cloud_exit)
    if not sel or df is None or df.empty:
        return {}
    cl = ichimoku_cloud(df)
    close = df["close"].astype(float)
    pos = np.where(cl["top"].isna(), "n/a", np.where(close > cl["top"], "Above",
                   np.where(close < cl["bottom"], "Below", "Center")))
    tf = pd.Timedelta(minutes=cfg.tf_min)
    return {candle_end(t, tf, exchange): p for t, p in zip(df["date"], pos) if p in sel}


# ============================== ANGEL ONE ==============================
def angel_login(api_key, client_id, mpin, totp_secret):
    try:
        import pyotp
        from SmartApi import SmartConnect
    except ImportError:
        return None, "Install first:  pip install smartapi-python pyotp"
    try:
        smart = SmartConnect(api_key=api_key.strip())
        otp = pyotp.TOTP(totp_secret.replace(" ", "").strip()).now()
        res = smart.generateSession(client_id.strip(), mpin.strip(), otp)
        if not res or not res.get("status"):
            return None, f"Login failed: {(res or {}).get('message', 'unknown error')}"
        return smart, None
    except Exception as ex:
        return None, f"Login error: {ex}"


def _to_ist_naive(series):
    return pd.to_datetime(series, utc=True).dt.tz_convert(IST).dt.tz_localize(None)


CANDLE_GAP_MIN, CANDLE_GAP_MAX = 0.36, 1.2          # Angel limit ~3 calls/sec -> start near it, back off only if it complains
CANDLE_CALL_GAP = CANDLE_GAP_MIN
_GAP_LOCK, _GAP_LAST = threading.Lock(), {}
QUOTE_BACKOFF = [0.0]


def _gap(kind, gap):
    """Minimum seconds between two calls of the same Angel API kind (all threads share it)."""
    with _GAP_LOCK:
        wait = _GAP_LAST.get(kind, 0.0) + gap - time.time()
        if wait > 0:
            time.sleep(wait)
        _GAP_LAST[kind] = time.time()


def _is_rate_err(msg):
    m = str(msg).lower()
    return "access rate" in m or "exceeding" in m or "too many" in m
_RATE_LOCK, _RATE_LAST = threading.Lock(), [0.0]


def _gap_adjust(rate_error):
    """Adaptive gap: +0.15s on every rate-limit error, shrinks slowly back to the minimum on success."""
    global CANDLE_CALL_GAP
    with _RATE_LOCK:
        if rate_error:
            CANDLE_CALL_GAP = min(CANDLE_GAP_MAX, CANDLE_CALL_GAP + 0.15)
        else:
            CANDLE_CALL_GAP = max(CANDLE_GAP_MIN, CANDLE_CALL_GAP - 0.01)


def _rate_wait():
    with _RATE_LOCK:
        wait = _RATE_LAST[0] + CANDLE_CALL_GAP - time.time()
        if wait > 0:
            time.sleep(wait)
        _RATE_LAST[0] = time.time()


def _candles(smart, item, interval, start, end):
    params = {
        "exchange": item["exchange"], "symboltoken": str(item["token"]), "interval": interval,
        "fromdate": start.strftime("%Y-%m-%d %H:%M"), "todate": end.strftime("%Y-%m-%d %H:%M"),
    }
    for _ in range(4):
        _rate_wait()
        try:
            res = smart.getCandleData(params)
        except Exception as ex_:
            res = None
            LAST_API_ERROR["candle"] = f"{type(ex_).__name__}: {ex_}"
        if res and res.get("status") and res.get("data") is not None:
            _gap_adjust(False)
            if not res["data"]:
                return pd.DataFrame()
            d = pd.DataFrame(res["data"], columns=["date", "open", "high", "low", "close", "volume"])
            d["date"] = _to_ist_naive(d["date"])
            return d
        rate_err = _is_rate_err(LAST_API_ERROR.get("candle", "")) or _is_rate_err((res or {}).get("message", ""))
        _gap_adjust(rate_err)
        time.sleep(3 if rate_err else 2)
    return pd.DataFrame()


def _resample_4h(d, exchange):
    """1-hour candles -> 4-hour candles aligned to the session start (NSE/BSE/NFO 09:15, MCX 09:00)."""
    sh, sm = (9, 0) if exchange == "MCX" else (9, 15)
    d = d.sort_values("date").copy()
    day = d["date"].dt.normalize()
    mins = (d["date"] - day).dt.total_seconds() // 60 - (sh * 60 + sm)
    d["_g"], d["_d"] = (mins // 240).astype(int), day
    out = d.groupby(["_d", "_g"], sort=True).agg(
        date=("date", "first"), open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
    ).reset_index(drop=True)
    return out


def candle_end(ts, tf, exchange):
    """When a candle that starts at ts is complete. 4H / day candles end at the market close, not ts + tf."""
    end = ts + tf
    if tf >= pd.Timedelta(minutes=240):
        close = ts.normalize() + (pd.Timedelta(hours=23, minutes=30) if exchange == "MCX"
                                  else pd.Timedelta(hours=15, minutes=30))
        end = min(end, close)
    return end


def fetch_candles(smart, item, interval, now, cloud=False):
    def days(default):                      # cloud filter ON -> load enough history for the Ichimoku cloud
        return max(default, CLOUD_DAYS.get(interval, default)) if cloud else default

    if interval == "FOUR_HOUR":
        d = _candles(smart, item, "ONE_HOUR", now - pd.Timedelta(days=days(45)), now)
        if not d.empty:
            d = _resample_4h(d, item["exchange"])
    elif interval == "ONE_DAY":
        d = _candles(smart, item, "ONE_DAY", now - pd.Timedelta(days=days(90)), now)
    else:
        d = _candles(smart, item, interval, now - pd.Timedelta(days=days(5)), now)
    return d[["date", "open", "high", "low", "close"]] if not d.empty else d


def fetch_bars(smart, item, since, now):
    d = _candles(smart, item, "ONE_MINUTE", since, now)
    return d.set_index("date")[["open", "high", "low", "close"]] if not d.empty else d


def fetch_ltp(smart, item):
    try:
        _gap("ltp", 0.2)
        res = smart.ltpData(item["exchange"], item["tradingsymbol"], str(item["token"]))
        return float(res["data"]["ltp"])
    except Exception:
        return None


LAST_API_ERROR = {"quote": "", "bulk": ""}


def _quote_from(d):
    ltp = float(d["ltp"])
    return {"ts": time.time(),
        "open": float(d.get("open", ltp)), "high": float(d.get("high", ltp)),
        "low": float(d.get("low", ltp)), "close": float(d.get("close", ltp)), "ltp": ltp,
    }


def fetch_quote(smart, item):
    try:
        _gap("ltp", 0.2)
        res = smart.ltpData(item["exchange"], item["tradingsymbol"], str(item["token"]))
        d = (res or {}).get("data")
        if not d:
            LAST_API_ERROR["quote"] = str((res or {}).get("message") or res)
            return None
        return _quote_from(d)
    except Exception as ex:
        LAST_API_ERROR["quote"] = f"{type(ex).__name__}: {ex}"
        return None


QUOTE_CHUNK = 50          # Angel getMarketData accepts max 50 tokens per request -> send big lists in chunks


def fetch_quotes_bulk(smart, items):
    """{token: quote}. Any number of stocks (sent in chunks of 50). Returns None only when NOTHING came back
       (failed / rate limited) - the caller must NOT hammer per-item calls."""
    if time.time() < QUOTE_BACKOFF[0]:
        LAST_API_ERROR["bulk"] = "rate limited - waiting"
        return None
    out = {}
    try:
        by_ex = {}
        for it in items:
            by_ex.setdefault(it["exchange"], []).append(str(it["token"]))
        reqs = [{ex: toks[i:i + QUOTE_CHUNK]} for ex, toks in by_ex.items() for i in range(0, len(toks), QUOTE_CHUNK)]
        for req in reqs:
            _gap("bulk", 1.2)
            res = smart.getMarketData("OHLC", req)
            fetched = ((res or {}).get("data") or {}).get("fetched") or []
            if not fetched:
                msg = str((res or {}).get("message") or res)[:200]
                LAST_API_ERROR["bulk"] = msg
                if _is_rate_err(msg):
                    QUOTE_BACKOFF[0] = time.time() + 4
                    break                      # stop now, keep what we already got
                continue                       # other chunks may still work
            for r in fetched:
                try:
                    out[str(r["symbolToken"])] = _quote_from(r)
                except (KeyError, TypeError, ValueError):
                    continue
    except Exception as ex:
        msg = f"{type(ex).__name__}: {ex}"
        LAST_API_ERROR["bulk"] = msg
        if _is_rate_err(msg):
            QUOTE_BACKOFF[0] = time.time() + 4
    return out or None


# ============================== INSTRUMENTS ==============================
NIFTY50_FALLBACK = (
    "ADANIENT ADANIPORTS APOLLOHOSP ASIANPAINT AXISBANK BAJAJ-AUTO BAJFINANCE BAJAJFINSV BEL BHARTIARTL CIPLA "
    "COALINDIA DRREDDY EICHERMOT ETERNAL GRASIM HCLTECH HDFCBANK HDFCLIFE HEROMOTOCO HINDALCO HINDUNILVR ICICIBANK "
    "INDUSINDBK INFY ITC JIOFIN JSWSTEEL KOTAKBANK LT M&M MARUTI NESTLEIND NTPC ONGC POWERGRID RELIANCE SBILIFE "
    "SBIN SHRIRAMFIN SUNPHARMA TATACONSUM TATAMOTORS TATASTEEL TCS TECHM TITAN TRENT ULTRACEMCO WIPRO"
).split()


@st.cache_data(ttl=86400, show_spinner=False)
def nifty50():
    try:
        r = requests.get(
            "https://niftyindices.com/IndexConstituent/ind_nifty50list.csv",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=15,
        )
        r.raise_for_status()
        syms = pd.read_csv(io.StringIO(r.text))["Symbol"].str.strip().tolist()
        if len(syms) >= 40:
            return syms
    except Exception:
        pass
    return list(NIFTY50_FALLBACK)


MASTER_CACHE = "scrip_master_cache.pkl"


@st.cache_resource(ttl=86400, show_spinner=False)
def load_master():
    if os.path.exists(MASTER_CACHE) and pd.Timestamp(os.path.getmtime(MASTER_CACHE), unit="s").date() == pd.Timestamp.now().date():
        try:
            cached = pd.read_pickle(MASTER_CACHE)
            if "tick_size" in cached.columns:
                return cached
        except Exception:
            pass
    df = pd.DataFrame(requests.get(MASTER_URL, timeout=120).json())
    df.columns = [c.strip().lower() for c in df.columns]
    keep = [c for c in ("exch_seg", "symbol", "name", "token", "lotsize", "instrumenttype", "expiry", "tick_size") if c in df.columns]
    df = df[keep]
    try:
        df.to_pickle(MASTER_CACHE)
    except Exception:
        pass
    return df


def search_item(smart, exchange, text):
    cache = st.session_state.setdefault("search_cache", {})
    key = f"{exchange}:{text}"
    if key not in cache:
        cache[key] = None
        try:
            rows = (smart.searchScrip(exchange, text) or {}).get("data") or []
            rows = [r for r in rows if r["tradingsymbol"].upper() in (text, text + "-EQ")] or rows
            if rows:
                r = rows[0]
                cache[key] = {
                    "symbol": r["tradingsymbol"].replace("-EQ", ""),
                    "tradingsymbol": r["tradingsymbol"],
                    "token": r["symboltoken"], "exchange": exchange, "lot": 1,
                }
        except Exception:
            pass
    return cache[key]


SENSEX30 = set(
    "ADANIPORTS ASIANPAINT AXISBANK BAJFINANCE BAJAJFINSV BHARTIARTL ETERNAL HCLTECH HDFCBANK HINDUNILVR ICICIBANK "
    "INFY ITC KOTAKBANK LT M&M MARUTI NESTLEIND NTPC POWERGRID RELIANCE SBIN SUNPHARMA TATAMOTORS TATASTEEL TCS "
    "TECHM TITAN TRENT ULTRACEMCO".split()
)
MCX_MAIN = set("GOLD GOLDM GOLDPETAL SILVER SILVERM CRUDEOIL CRUDEOILM NATURALGAS COPPER ZINC ALUMINIUM LEAD NICKEL".split())
NFO_INDEX = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}


NSE_INDEX_FILES = {          # niftyindices.com constituent CSVs (name shown in the Index box -> file name)
    # broad market
    "Nifty 50": "ind_nifty50list", "Nifty Next 50": "ind_niftynext50list", "Nifty 100": "ind_nifty100list",
    "Nifty 200": "ind_nifty200list", "Nifty 500": "ind_nifty500list",
    "Nifty Total Market": "ind_niftytotalmarket_list",
    # mid / small / micro cap
    "Nifty Midcap 50": "ind_niftymidcap50list", "Nifty Midcap 100": "ind_niftymidcap100list",
    "Nifty Midcap 150": "ind_niftymidcap150list",
    "Nifty Smallcap 50": "ind_niftysmallcap50list", "Nifty Smallcap 100": "ind_niftysmallcap100list",
    "Nifty Smallcap 250": "ind_niftysmallcap250list",
    "Nifty LargeMidcap 250": "ind_niftylargemidcap250list", "Nifty MidSmallcap 400": "ind_niftymidsmallcap400list",
    "Nifty Microcap 250": "ind_niftymicrocap250_list",
    # sectors
    "Nifty Bank": "ind_niftybanklist", "Nifty Private Bank": "ind_nifty_privatebanklist",
    "Nifty PSU Bank": "ind_niftypsubanklist", "Nifty Financial Services": "ind_niftyfinancelist",
    "Nifty IT": "ind_niftyitlist", "Nifty Auto": "ind_niftyautolist", "Nifty Pharma": "ind_niftypharmalist",
    "Nifty Healthcare": "ind_niftyhealthcarelist", "Nifty FMCG": "ind_niftyfmcglist",
    "Nifty Metal": "ind_niftymetallist", "Nifty Realty": "ind_niftyrealtylist", "Nifty Energy": "ind_niftyenergylist",
    "Nifty Oil & Gas": "ind_niftyoilgaslist", "Nifty Infrastructure": "ind_niftyinfralist",
    "Nifty Media": "ind_niftymedialist", "Nifty Consumer Durables": "ind_niftyconsumerdurableslist",
    # themes
    "Nifty PSE": "ind_niftypselist", "Nifty CPSE": "ind_niftycpselist", "Nifty Commodities": "ind_niftycommoditieslist",
    "Nifty India Consumption": "ind_niftyconsumptionlist", "Nifty MNC": "ind_niftymnclist",
    "Nifty Services Sector": "ind_niftyservicesectorlist", "Nifty Alpha 50": "ind_niftyalpha50list",
}
MCX_GROUPS = {
    "Bullion": {"GOLD", "GOLDM", "SILVER", "SILVERM"},
    "Energy": {"CRUDEOIL", "CRUDEOILM", "NATURALGAS"},
    "Base Metals": {"COPPER", "ZINC", "ALUMINIUM", "LEAD", "NICKEL"},
}


def index_options(exchange):
    """Names for the 'Index' box under Exchange."""
    if exchange == "NSE":
        return list(NSE_INDEX_FILES)
    if exchange == "BSE":
        return ["Sensex 30"]
    if exchange == "NFO":
        return ["Index futures", "Nifty 50 stock futures"]
    return list(MCX_GROUPS)


@st.cache_data(ttl=86400, show_spinner=False)
def nse_index_members(index):
    """Symbols of one NSE index ([] when the download fails)."""
    if index == "Nifty 50":
        return nifty50()
    f = NSE_INDEX_FILES.get(index)
    if not f:
        return []
    for url in (f"https://niftyindices.com/IndexConstituent/{f}.csv",
                f"https://archives.nseindia.com/content/indices/{f}.csv"):
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            r.raise_for_status()
            out = [x.strip() for x in pd.read_csv(io.StringIO(r.text))["Symbol"].astype(str) if x.strip()]
            if out:
                return out
        except Exception:
            continue
    return []


def index_members(exchange, index):
    """Upper-case names of the chosen indices (tuple of names; None / empty -> the exchange's default list)."""
    if not index:
        if exchange == "NFO":
            return NFO_INDEX | set(nifty50())
        if exchange == "MCX":
            return MCX_MAIN
        return set(SENSEX30) if exchange == "BSE" else {x.upper() for x in nifty50()}
    out = set()
    for ix in index:
        if exchange == "NSE":
            out |= {x.upper() for x in nse_index_members(ix)}
        elif exchange == "BSE":
            out |= set(SENSEX30)
        elif exchange == "NFO":
            out |= NFO_INDEX if ix == "Index futures" else set(nifty50())
        else:
            out |= set(MCX_GROUPS.get(ix, MCX_MAIN))
    return out


def build_index_map(items, exchange, index_pick):
    """{symbol: ['Nifty 50', 'Nifty Bank', ...]} - which of the chosen indices each selected stock belongs to
       (a stock in two indices is listed under both)."""
    out = {}
    for ix in (index_pick or []):
        mem = index_members(exchange, (ix,))
        for it in items:
            keys = {str(it.get("name", "")).upper(), str(it["symbol"]).upper(), str(it.get("tradingsymbol", "")).upper()}
            if keys & mem:
                out.setdefault(it["symbol"], []).append(ix)
    return out


def exchange_list(df, exchange, index=None):
    index = tuple(index) if index else None
    d = df[df["exch_seg"] == exchange].copy()
    nm = d["name"].fillna("").str.upper()
    sy = d["symbol"].fillna("").str.upper()
    base = index_members(exchange, index)
    if exchange == "NSE":
        d = d[sy.str.endswith("-EQ") & nm.isin(base)]
        d["label"] = d["name"]
    elif exchange == "BSE":
        d = d[nm.isin(SENSEX30) | sy.isin(SENSEX30)]
        d["label"] = d["name"].where(nm.isin(SENSEX30), d["symbol"])
    else:
        itype = d["instrumenttype"].fillna("") if "instrumenttype" in d else pd.Series("", index=d.index)
        if exchange == "NFO" and not index:
            base = NFO_INDEX | set(nifty50())
        d = d[itype.str.startswith("FUT") & nm.isin(base)]
        d["exp"] = pd.to_datetime(d["expiry"], format="%d%b%Y", errors="coerce")
        d = d[d["exp"] >= pd.Timestamp.now().normalize()].sort_values("exp")
        d = d.drop_duplicates("name")
        d["label"] = d["name"]                    # GOLDM / SILVERM ... - the contract is chosen per date in the backtest
    return d.drop_duplicates("label").sort_values("label")


@st.cache_data(ttl=3600, show_spinner=False)
def exchange_list_cached(exchange, index=None):
    return exchange_list(load_master(), exchange, tuple(index) if index else None)


def no_dim():
    st.markdown(
        """<style>
        [data-stale="true"], .stale-element, [data-testid="stElementContainer"][data-stale="true"],
        [data-testid="stSidebar"] [data-stale="true"] { opacity: 1 !important; transition: none !important; }
        </style>""",
        unsafe_allow_html=True,
    )


def to_item(row, exchange):
    try:
        lot = max(int(float(row.get("lotsize") or 1)), 1) if exchange in ("NFO", "MCX") else 1
    except ValueError:
        lot = 1
    try:
        tick = float(row.get("tick_size") or 5) / 100.0          # master gives tick in paise
        tick = tick if tick > 0 else 0.05
    except (TypeError, ValueError):
        tick = 0.05
    return {
        "symbol": row["label"], "tradingsymbol": row["symbol"], "token": row["token"],
        "exchange": exchange, "lot": lot, "tick": tick, "name": str(row.get("name") or row["label"]),
    }



def margin_factor(exchange):
    return FUTURES_MARGIN_PCT / 100 if exchange in ("NFO", "MCX") else 1.0



def _csv_button(label, df, name, key):
    empty = df is None or df.empty
    st.download_button(
        label, b"" if empty else df.to_csv(index=False).encode(),
        f"{name}_{now_ist():%Y%m%d}.csv", "text/csv", key=key, disabled=empty,
    )



def _ui_css():
    """ONE style block per form: popover pickers look like the dropdowns, labels / sections line up."""
    st.markdown(
        "<style>"
        "div[data-testid='stElementContainer']:has(> div[data-testid='stMarkdown'] style){display:none}"
        "div[data-testid='stPopover'] button{display:flex;justify-content:space-between;align-items:center;"
        "width:100%;min-height:auto;text-align:left;border-radius:.5rem;border:1px solid transparent;"
        "background:rgba(128,128,128,.16);background:color-mix(in srgb,currentColor 10%,transparent);"
        "color:inherit;font-weight:400;transition:border-color .15s}"
        "div[data-testid='stPopover'] button:hover{border-color:var(--primary-color,#ff4b4b)}"
        "div[data-testid='stPopover'] button > div{flex:1;text-align:left}"
        "div[data-testid='stPopover'] button p{text-align:left;margin:0}"
        "div[data-testid='stPopoverBody']{border-radius:.75rem;box-shadow:0 10px 28px rgba(0,0,0,.30);"
        "min-width:min(34rem,90vw)}"
        "div[data-testid='stPopoverBody'] div[data-testid='stHorizontalBlock']{flex-wrap:nowrap !important;gap:.5rem}"
        "div[data-testid='stPopoverBody'] div[data-testid='stColumn'],"
        "div[data-testid='stPopoverBody'] div[data-testid='column']{min-width:0 !important}"
        ".fld-l{font-size:.875rem;line-height:1.4;margin:0 0 .3rem 0;opacity:.92}"
        ".sec-h{display:flex;align-items:center;gap:.55rem;font-weight:600;font-size:1.05rem;margin:0 0 .6rem 0}"
        ".sec-h i{display:inline-block;width:.32rem;height:1.15rem;border-radius:3px;background:var(--primary-color,#ff4b4b)}"
        ".sec-h small{font-weight:400;opacity:.6;font-size:.8rem}"
        "div[data-testid='stVerticalBlockBorderWrapper']{border-radius:.9rem}"
        "</style>",
        unsafe_allow_html=True,
    )


def _lbl(text, help=None):
    """Field label drawn the same way for every control (so rows line up)."""
    tip = f" title=\"{help}\"" if help else ""
    st.markdown(f"<div class='fld-l'{tip}>{text}{' &#9432;' if help else ''}</div>", unsafe_allow_html=True)


def _section(title, sub=""):
    st.markdown(f"<div class='sec-h'><i></i>{title}" + (f" <small>{sub}</small>" if sub else "") + "</div>",
                unsafe_allow_html=True)


def _stock_picker(label, labels, key, default=None, button_label="Select stocks"):
    """Stock picker: dropdown-style button -> tick-box list; small 'Select all' / 'Clear' on its right side."""
    big = len(labels) > 300                      # huge lists (NFO etc.) keep the normal multiselect
    ck = {lb: f"{key}__ck_{lb}" for lb in labels}
    if big:
        all_key = f"{key}__all"
        if key not in st.session_state:
            st.session_state[key] = [x for x in (default or []) if x in labels]
    else:
        dset = set(default or [])
        for lb, k_ in ck.items():
            if k_ not in st.session_state:
                st.session_state[k_] = lb in dset

    def _sel_all():
        if big:
            st.session_state[all_key] = True
        else:
            for k_ in ck.values():
                st.session_state[k_] = True

    def _clear():
        if big:
            st.session_state[all_key] = False
            st.session_state[key] = []
        else:
            for k_ in ck.values():
                st.session_state[k_] = False

    if big:
        _lbl(label)
        n_big = len(labels) if st.session_state.get(all_key) else len(st.session_state.get(key, []))
        with st.popover(f"{button_label} · {n_big} selected" if n_big else button_label, use_container_width=True):
            lst, side = st.columns([6, 1.4])
            with lst:
                if st.session_state.get(all_key):
                    st.success(f"✅ All {len(labels)} selected")
                else:
                    st.multiselect(label, labels, key=key, label_visibility="collapsed")
            with side:
                st.button("Select all", key=f"{key}__btn_all", on_click=_sel_all, type="tertiary", use_container_width=True)
                st.button("Clear", key=f"{key}__btn_clr", on_click=_clear, type="tertiary", use_container_width=True)
        return list(labels) if st.session_state.get(all_key) else list(st.session_state.get(key, []))
    _lbl(label)
    n_sel = sum(1 for k_ in ck.values() if st.session_state.get(k_))
    with st.popover(f"{button_label} · {n_sel} selected" if n_sel else button_label, use_container_width=True):
        lst, side = st.columns([6, 1.4])
        with lst:
            with st.container(height=min(320, 38 * len(labels) + 12)):
                for lb in labels:
                    st.checkbox(lb, key=ck[lb])
        with side:
            st.button("Select all", key=f"{key}__btn_all", on_click=_sel_all, type="tertiary", use_container_width=True)
            st.button("Clear", key=f"{key}__btn_clr", on_click=_clear, type="tertiary", use_container_width=True)
    return [lb for lb in labels if st.session_state.get(ck[lb])]


SIZE_LABELS = {"amount": "Amount", "qty": "Qty", "equal": "Equal weight"}


def _equal_amt(capital, max_open, no_limit, n_items):
    """Equal weight = capital / max positions (or / number of selected stocks when there is no limit)."""
    n = n_items if no_limit else max_open
    return round(float(capital) / n, 2) if capital and n else None


def _sizing_inputs(c_mode, c_val, capital, max_open, no_limit, n_items, key,
                   cur_mode="amount", default_amt=None, cur_qty=None):
    """Dropdown (Amount / Qty / Equal weight) + the value box for the chosen option.
       Returns (size_mode, amount, qty). amount = Rs per position (equal weight: auto-calculated)."""
    keys = list(SIZE_LABELS)
    pick = c_mode.selectbox("Position", [SIZE_LABELS[k] for k in keys],
                            index=keys.index(cur_mode) if cur_mode in keys else 0, key=f"{key}_size_mode",
                            help="Amount = oru position-ku evvalavu Rs | Qty = oru position-ku evvalavu qty "
                                 "(shares; NFO/MCX-la lots) | Equal weight = capital / max positions "
                                 "(no limit capital / selected stocks)")
    mode = keys[[SIZE_LABELS[k] for k in keys].index(pick)]
    eq = _equal_amt(capital, max_open, no_limit, n_items)
    if mode == "qty":
        q = c_val.number_input("Qty", min_value=1, value=int(cur_qty) if cur_qty else None,
                               step=1, placeholder="Qty", key=f"{key}_size_qty")
        return mode, float(eq or 0), (int(q) if q else None)
    if mode == "equal":
        c_val.text_input("Capital / positions (auto)", value=f"{eq:,.2f}" if eq else "", disabled=True)
        return mode, eq, 0
    amt = c_val.number_input("Amount", min_value=1.0,
                             value=float(default_amt) if default_amt else None, step=1000.0, placeholder="Amount")
    return mode, amt, 0



# ============================== BACKTEST ==============================
# Same entry / exit / sizing rules as the paper engine, but run over HISTORY (From date -> To date).
#   Entry : a pattern completes on candle i  ->  buy at the OPEN of candle i+1 (paper buys at LTP right after the close)
#   Exit  : target % / optional SL % / time exit at square-off (max_hold days) - same as check_exits()
#   Sizing: Config.size_qty()  (amount / qty / equal weight), max open positions, free-cash cap - same as Engine._enter()
import bisect
from concurrent.futures import ThreadPoolExecutor, as_completed

# Angel One getCandleData max days per request (kept a little below the limit)
BT_CHUNK_DAYS = {"ONE_MINUTE": 28, "THREE_MINUTE": 58, "FIVE_MINUTE": 95, "TEN_MINUTE": 95,
                 "FIFTEEN_MINUTE": 190, "THIRTY_MINUTE": 190, "ONE_HOUR": 380, "ONE_DAY": 1900}
BT_CACHE_DIR = os.path.join(_DATA_DIR, "backtest_cache")
os.makedirs(BT_CACHE_DIR, exist_ok=True)
_BT_ORIGIN = pd.Timestamp("2015-01-01")          # fixed chunk grid -> past chunks are re-used from disk on every run


# ---- FUTURES CONTRACT REGISTRY: the scrip master only lists LIVE contracts, so every contract ever seen is remembered
#      on disk (token, symbol, expiry). The backtest then uses, on every date, the contract that was the front month then.
CONTRACTS_FILE = os.path.join(_DATA_DIR, "contracts_registry.json")
ROLL_DAYS = 3                                   # switch to the next contract this many days before the expiry day
_REG_LOCK = threading.Lock()


def _registry_load():
    try:
        with open(CONTRACTS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def registry_update(master):
    """Add every live NFO / MCX futures contract of the scrip master to the on-disk registry."""
    try:
        d = master[master["exch_seg"].isin(["NFO", "MCX"])]
        d = d[d["instrumenttype"].fillna("").astype(str).str.startswith("FUT")]
        with _REG_LOCK:
            reg, changed = _registry_load(), False
            cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=1100)
            for r in d.to_dict("records"):
                exp = pd.to_datetime(r.get("expiry"), format="%d%b%Y", errors="coerce")
                if pd.isna(exp) or exp < cutoff:
                    continue
                try:
                    lot = max(int(float(r.get("lotsize") or 1)), 1)
                except (TypeError, ValueError):
                    lot = 1
                grp = reg.setdefault(f"{r['exch_seg']}|{str(r['name']).upper()}", {})
                tok = str(r["token"])
                if tok not in grp:
                    grp[tok] = {"token": tok, "symbol": r["symbol"], "expiry": exp.strftime("%Y-%m-%d"), "lot": lot}
                    changed = True
            if changed:
                _write_json(CONTRACTS_FILE, reg)
    except Exception:
        pass


@st.cache_resource(ttl=86400, show_spinner=False)
def _registry_sync():
    registry_update(load_master())
    return True


def _contracts(item):
    """Known futures contracts of this name, earliest expiry first."""
    key = f"{item['exchange']}|{str(item.get('name') or item['symbol']).upper()}"
    rows = list(_registry_load().get(key, {}).values())
    if not any(str(r["token"]) == str(item["token"]) for r in rows):      # registry not synced yet -> at least the selected contract
        rows.append({"token": str(item["token"]), "symbol": item["tradingsymbol"], "expiry": "2100-01-01"})
    return sorted(rows, key=lambda r: r["expiry"])


def _bt_load_cont(smart, item, interval, start, end):
    """Candles for one stock. NFO / MCX futures: the contract that was the FRONT month on each date is stitched into one series
       (new column 'seg' = which contract; it changes at every roll). Equities: same as _bt_load (seg = 0)."""
    if item["exchange"] not in ("NFO", "MCX"):
        d = _bt_load(smart, item, interval, start, end)
        if not d.empty:
            d["seg"] = 0
        return d
    parts, lo = [], pd.Timestamp(start)
    for n, c in enumerate(_contracts(item)):
        hi = pd.Timestamp(c["expiry"]) - pd.Timedelta(days=ROLL_DAYS)            # this contract is used until here (exclusive)
        a, b = max(lo, pd.Timestamp(start)), min(hi, pd.Timestamp(end))
        lo = max(lo, hi)
        if a >= b:
            continue
        d = _bt_load(smart, {**item, "token": c["token"], "tradingsymbol": c["symbol"]}, interval, a, b)
        if d is not None and not d.empty:
            parts.append(d.assign(seg=n))
        if b >= pd.Timestamp(end):
            break
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts).drop_duplicates("date").sort_values("date").reset_index(drop=True)


def _bt_load(smart, item, interval, start, end):
    """Candles from start to end for one stock (date, open, high, low, close), fetched in API-sized chunks.
       Chunks that lie completely in the past are cached on disk, so changing target % / patterns is instant."""
    base = "ONE_HOUR" if interval == "FOUR_HOUR" else interval
    step = pd.Timedelta(days=BT_CHUNK_DAYS[base])
    today = now_ist().normalize()
    cur = _BT_ORIGIN + ((pd.Timestamp(start) - _BT_ORIGIN) // step) * step
    parts = []
    while cur < end:
        nxt = cur + step
        fn = os.path.join(BT_CACHE_DIR, f"{item['exchange']}_{item['token']}_{base}_{cur:%Y%m%d}.pkl")
        full = nxt <= today                              # chunk is complete -> safe to cache
        d = None
        if full and os.path.exists(fn):
            try:
                d = pd.read_pickle(fn)
            except Exception:
                d = None
        if d is None:
            d = _candles(smart, item, base, cur, nxt - pd.Timedelta(minutes=1))
            if full and d is not None and not d.empty:
                try:
                    d.to_pickle(fn)
                except Exception:
                    pass
        if d is not None and not d.empty:
            parts.append(d)
        cur = nxt
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    d = d[(d["date"] >= start) & (d["date"] < end)].reset_index(drop=True)
    if interval == "FOUR_HOUR" and not d.empty:
        d = _resample_4h(d, item["exchange"])
    return d[["date", "open", "high", "low", "close"]] if not d.empty else d


def _bt_deadline(etime, ex, max_hold, xt=None):
    """Square-off moment: the exit time (default exit_time()) on the max_hold-th business day, counting the entry day as day 1."""
    day = np.busday_offset(np.datetime64(etime.date()), max(int(max_hold), 1) - 1, roll="forward")
    xt = xt or exit_time(ex)
    return pd.Timestamp(day) + pd.Timedelta(hours=xt.hour, minutes=xt.minute)


def _bt_exit(T, O, H, L, C, i0, deadline, tgt, sl, cx=None, S=None, mode="OR"):
    """Walk forward from bar i0. SL is checked before target inside a bar (same as _check_exit).
       cx (optional) = per-bar cloud-exit position ('' = no signal).
       mode "OR"  : the FIRST of target / cloud exit exits the trade (target at its price, cloud at that bar's close).
       mode "AND" : BOTH must be satisfied. The target counts as satisfied once price has touched it (stays satisfied);
                    the trade then exits at the close of the first bar that closes in a chosen cloud position
                    (that same bar counts). SL and the time exit always stay active.
       Returns (price, time, reason) or None when there is no bar to walk."""
    n, k, last = len(T), i0, None
    both = (mode == "AND" and cx is not None)
    tgt_hit = False
    s0 = S[i0] if (S is not None and i0 < n) else None
    while k < n and T[k] < deadline:
        if S is not None and S[k] != s0 and last is not None:       # the contract rolled -> prices are no longer comparable
            return float(C[last]), T[last], "Contract roll"
        if sl:
            if O[k] <= sl:
                return float(O[k]), T[k], "SL (gap)"
            if L[k] <= sl:
                return float(sl), T[k], "SL"
        if both:
            if not tgt_hit and H[k] >= tgt:                         # target touched -> satisfied from now on
                tgt_hit = True
            if tgt_hit and cx[k]:                                   # ... and a bar closes in the chosen cloud position
                return float(C[k]), T[k], f"Target + Cloud ({cx[k]})"
        else:
            if O[k] >= tgt:
                return float(O[k]), T[k], "Target (gap)"
            if H[k] >= tgt:
                return float(tgt), T[k], "Target"
            if cx is not None and cx[k]:
                return float(C[k]), T[k], f"Cloud exit ({cx[k]})"
        last = k
        k += 1
    if last is None:
        return None
    if k < n:
        return float(C[last]), deadline, "Time exit"
    return float(C[last]), T[last], "Open at end of data"


def _bt_candidates(it, df, cfg, start, end, order):
    """Every pattern signal of one stock inside [start, end) -> (entry candidates, skipped-signal records)."""
    ex, sym = it["exchange"], it["symbol"]
    tf = pd.Timedelta(minutes=cfg.tf_min)
    sig = detect_patterns(df, cfg)
    pats = [p for p in cfg.patterns if p in sig.columns]
    if not pats or len(df) < 3:
        return [], []
    cloud_sel = cloud_list(cfg.cloud_filter)                # optional filter (off by default); the cloud column is ALWAYS filled
    cl = ichimoku_cloud(df)
    close = df["close"].astype(float)
    cpos = np.where(cl["top"].isna(), "n/a", np.where(close > cl["top"], "Above",
                    np.where(close < cl["bottom"], "Below", "Center")))
    T = df["date"].tolist()
    O = df["open"].astype(float).to_numpy()
    seg = df["seg"].to_numpy() if "seg" in df.columns else None
    open_off = pd.Timedelta(hours=9) if ex == "MCX" else pd.Timedelta(hours=9, minutes=15)
    xt_eff = eff_exit(cfg, ex)                              # chosen exit time (default square-off)
    cands, skipped = [], []
    for i in np.flatnonzero(sig[pats].any(axis=1).to_numpy()):
        ts = T[i]
        if ts < start or ts >= end:
            continue
        rec = {"symbol": sym, "exchange": ex, "candle": ts,
               "pattern": "+".join(p for p in pats if bool(sig[p].iat[i])),
               "cloud": cpos[i],
               "_et": T[i + 1] if i + 1 < len(T) else None}          # candle where the entry would happen (used by the time condition)
        if cloud_sel and cpos[i] not in cloud_sel:
            skipped.append({**rec, "status": f"SKIPPED (cloud {cpos[i]} - filter needs {' / '.join(cloud_sel)})"})
            continue
        cend = candle_end(ts, tf, ex)
        intraday = tf < pd.Timedelta(minutes=240)
        if intraday and cend.time() >= xt_eff:
            skipped.append({**rec, "status": f"SKIPPED (no new entries after {xt_eff:%H:%M} (exit time))"})
            continue
        if i + 1 >= len(df):
            continue                                        # no next candle in the data yet
        j = i + 1
        if seg is not None and seg[max(i - 4, 0)] != seg[j]:
            skipped.append({**rec, "status": "SKIPPED (pattern candles span a contract roll)"})
            continue
        if intraday and (T[j].date() != ts.date() or T[j] - cend > tf):
            skipped.append({**rec, "status": "SKIPPED (data gap after signal candle)"})
            continue
        etime = T[j] if tf < pd.Timedelta(minutes=1440) else T[j].normalize() + open_off
        cands.append({**rec, "etime": etime, "entry_raw": float(O[j]), "j": j, "item": it, "order": order})
    return cands, skipped


def _brokerage(cfg, buy_value, sell_value):
    """(buy-leg brokerage, sell-leg brokerage).  Off -> 0.   "pct": value x %   |   "flat": rupees per order."""
    if not cfg.brokerage_on:
        return 0.0, 0.0
    if cfg.brokerage_mode == "flat":
        return float(cfg.brokerage_buy_flat), float(cfg.brokerage_sell_flat)
    return buy_value * cfg.brokerage_buy_pct / 100, sell_value * cfg.brokerage_sell_pct / 100


def _ts(v):
    """Timestamp / ISO text / NaT / None -> naive Timestamp (or None)."""
    if v is None or (not isinstance(v, str) and pd.isna(v)) or (isinstance(v, str) and not v.strip()):
        return None
    t = pd.Timestamp(v)
    return t.tz_localize(None) if t.tzinfo is not None else t


def _bt_portfolio(cands, cfg):
    """Replays the entries in time order with ONE shared capital / max-position pool (like Engine._enter).
       Every candidate already carries its exit ("hit" = price, time, reason, release time) - the exit does not depend on the
       position size or the brokerage - so the same signals can be replayed again with other compound / brokerage settings
       without touching the candle data."""
    cands = sorted(cands, key=lambda r: (r["etime"], r["order"]))
    open_, closed, skipped = [], [], []
    realized = 0.0                                          # net P&L of the positions that are already closed
    for c in cands:
        still = []
        for p in open_:                                    # positions that already exited free their slot + cash
            if p["_xt"] <= c["etime"]:
                closed.append(p)
                realized += p["pnl"]
            else:
                still.append(p)
        open_ = still
        rec = {k: c[k] for k in ("symbol", "exchange", "candle", "pattern", "cloud", "_et")}
        ex = c["exchange"]
        if any(p["symbol"] == c["symbol"] and p["exchange"] == ex for p in open_):
            skipped.append({**rec, "status": "SKIPPED (already in position)"})
            continue
        if len(open_) >= cfg.max_open:
            skipped.append({**rec, "status": f"SKIPPED (positions full {len(open_)}/{cfg.max_open})"})
            continue
        entry = c["entry_raw"] * (1 + cfg.slippage_pct / 100)
        lot, mf = int(c.get("lot", 1)), margin_factor(ex)
        equity = max(cfg.capital + realized, 0.0)           # COMPOUND: the new position is sized from this
        cash = cfg.capital + realized - sum(p["entry"] * p["qty"] * p["_mf"] for p in open_)
        _, qty = cfg.size_qty(entry * lot * mf, lot, cash, equity)
        if qty < 1:
            skipped.append({**rec, "status": f"SKIPPED (1 unit needs {entry * lot * mf:,.0f}; {cfg.size_text}, free cash {cash:,.0f})"})
            continue
        hit = c.get("hit")
        if hit is None:
            skipped.append({**rec, "status": "SKIPPED (no price data after entry)"})
            continue
        raw_px, xt, reason, release = hit
        exit_px = raw_px * (1 - cfg.slippage_pct / 100)
        buy_value, sell_value = entry * qty, exit_px * qty
        brk_buy, brk_sell = _brokerage(cfg, buy_value, sell_value)     # optional - see Config.brokerage_on / brokerage_mode
        gross = sell_value - buy_value                                  # P&L before brokerage
        pnl = gross - brk_buy - brk_sell                                # net P&L after both legs
        open_.append({
            "symbol": c["symbol"], "exchange": ex, "pattern": c["pattern"], "cloud": c["cloud"],
            "signal_candle": c["candle"], "entry_time": c["etime"], "entry": round(entry, 2),
            "target": c["tgt"], "stop_loss": c["sl"], "qty": int(qty), "invested": round(entry * qty * mf, 2),
            "exit_time": xt, "exit": round(exit_px, 2), "reason": reason,
            "gross_pnl": round(gross, 2), "brokerage_buy": round(brk_buy, 2), "brokerage_sell": round(brk_sell, 2),
            "brokerage_total": round(brk_buy + brk_sell, 2),
            "pnl": round(pnl, 2), "pnl_pct": round(pnl / buy_value * 100, 2),
            "_xt": release, "_mf": mf,
        })
        skipped.append({**rec, "status": "ENTERED"})
    closed += open_
    return closed, skipped


REPLAY_CAND_COLS = ["symbol", "exchange", "lot", "order", "candle", "pattern", "cloud", "et", "etime", "entry_raw", "tgt", "sl",
                    "has_hit", "x_px", "x_time", "x_reason", "x_release"]
REPLAY_SIG_COLS = ["symbol", "exchange", "candle", "pattern", "cloud", "et", "status"]


def _bt_make_replay(cands, sigs, cfg):
    """Everything needed to replay a finished run with other compound / brokerage settings (no candle data needed):
       one row per entry signal (entry price + its exit), the signals skipped before any sizing, and the settings of the run."""
    rows = []
    for c in cands:
        h = c.get("hit")
        rows.append({"symbol": c["symbol"], "exchange": c["exchange"], "lot": int(c["item"].get("lot", 1)), "order": int(c["order"]),
                     "candle": c["candle"], "pattern": c["pattern"], "cloud": c["cloud"], "et": c.get("_et"), "etime": c["etime"],
                     "entry_raw": float(c["entry_raw"]), "tgt": c["tgt"], "sl": c["sl"],
                     "has_hit": h is not None, "x_px": h[0] if h else None, "x_time": h[1] if h else None,
                     "x_reason": h[2] if h else "", "x_release": h[3] if h else None})
    srows = [{"symbol": r["symbol"], "exchange": r["exchange"], "candle": r["candle"], "pattern": r["pattern"],
              "cloud": r["cloud"], "et": r.get("_et"), "status": r["status"]} for r in sigs]
    return {"cands": pd.DataFrame(rows, columns=REPLAY_CAND_COLS), "sigs": pd.DataFrame(srows, columns=REPLAY_SIG_COLS),
            "cfg": asdict(cfg)}


def bt_replay(replay, cfg):
    """Replays the stored signals with the compound / brokerage settings of cfg -> (trades, signals)."""
    cands = []
    for r in replay["cands"].to_dict("records"):
        hit = None
        if bool(r.get("has_hit")):
            hit = (float(r["x_px"]), _ts(r["x_time"]), str(r["x_reason"]), _ts(r["x_release"]))
        sl = r.get("sl")
        cands.append({"symbol": r["symbol"], "exchange": r["exchange"], "lot": int(r.get("lot") or 1), "order": int(r["order"]),
                      "candle": _ts(r["candle"]), "pattern": r["pattern"], "cloud": r["cloud"], "_et": _ts(r.get("et")),
                      "etime": _ts(r["etime"]), "entry_raw": float(r["entry_raw"]), "tgt": float(r["tgt"]),
                      "sl": None if (sl is None or pd.isna(sl)) else float(sl), "hit": hit})
    closed, entered = _bt_portfolio(cands, cfg)
    sigs = [{"symbol": r["symbol"], "exchange": r["exchange"], "candle": _ts(r["candle"]), "pattern": r["pattern"],
             "cloud": r["cloud"], "_et": _ts(r.get("et")), "status": r["status"]} for r in replay["sigs"].to_dict("records")]
    # TIME CONDITION (optional) - applied LAST: every other condition (pattern, cloud, positions, cash...) has already passed;
    # only the results whose entry time is one of the chosen times are shown.
    t_set = {t for t in (_hhmm(x) for x in str(cfg.entry_times or "").split(",")) if t}
    if t_set and int(cfg.tf_min) < 1440:
        closed = [p for p in closed if pd.Timestamp(p["entry_time"]).time() in t_set]
        sigs = [r for r in sigs if r.get("_et") is not None and pd.Timestamp(r["_et"]).time() in t_set]
        entered = [r for r in entered if r.get("_et") is not None and pd.Timestamp(r["_et"]).time() in t_set]
    for r in sigs + entered:
        r.pop("_et", None)
    trades = pd.DataFrame(closed).drop(columns=["_xt", "_mf"], errors="ignore")
    if not trades.empty:
        trades = trades.sort_values(["exit_time", "entry_time"]).reset_index(drop=True)
    signals = pd.DataFrame(sigs + entered)
    if not signals.empty:
        signals = signals.sort_values(["candle", "symbol"]).reset_index(drop=True)
    return trades, signals


def run_backtest(smart, cfg, items, exchange, d_from, d_to, precise=False, progress=None):
    """-> dict(trades, signals, errors). precise=True walks 1-minute candles for target / SL (slower, closer to live)."""
    tf = pd.Timedelta(minutes=cfg.tf_min)
    start, end = pd.Timestamp(d_from), pd.Timestamp(d_to) + pd.Timedelta(days=1)          # end = exclusive
    warm = pd.Timedelta(days=CLOUD_DAYS.get(cfg.interval, 10) + 5)      # always load enough history for the reference cloud column
    tail = pd.Timedelta(days=int(cfg.max_hold) * 2 + 5)                                    # room for exits after To date
    load_to = min(end + tail, now_ist() + pd.Timedelta(days=1))
    errors, cands, sigs, bars, cxs = [], [], [], {}, {}

    def load(it):
        df = _bt_load_cont(smart, it, cfg.interval, start - warm, load_to)
        b1 = _bt_load_cont(smart, it, "ONE_MINUTE", start, load_to) if precise and not df.empty else None
        return it, df, b1

    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(load, it): n for n, it in enumerate(items)}
        for f in as_completed(futs):
            order = futs[f]
            done += 1
            if progress:
                progress(done, len(items), items[order]["symbol"])
            try:
                it, df, b1 = f.result()
            except Exception as ex_:
                errors.append(f"{items[order]['symbol']}: {ex_}")
                continue
            if df is None or df.empty:
                errors.append(f"{it['symbol']}: no candle data for this period")
                continue
            c, s = _bt_candidates(it, df, cfg, start, end, order)
            cands += c
            sigs += s
            src = b1 if (b1 is not None and not b1.empty) else df
            bars[(it["exchange"], it["symbol"])] = (
                src["date"].tolist(), src["open"].astype(float).to_numpy(), src["high"].astype(float).to_numpy(),
                src["low"].astype(float).to_numpy(), src["close"].astype(float).to_numpy(), src is df,
                src["seg"].to_numpy() if "seg" in src.columns else None)
            if cloud_list(cfg.cloud_exit):                  # cloud EXIT: mark every walk bar that closes a matching cloud candle
                cmap = cloud_exit_events(df, cfg, it["exchange"])
                bl = pd.Timedelta(minutes=cfg.tf_min) if src is df else pd.Timedelta(minutes=1)
                cxs[(it["exchange"], it["symbol"])] = [cmap.get(candle_end(t, bl, it["exchange"]), "") for t in src["date"]]
            if precise and (b1 is None or b1.empty):
                errors.append(f"{it['symbol']}: no 1-minute data, used {cfg.interval} candles for exits")

    def exits_for(c, tgt, sl):
        T, O, H, L, C, same_tf, S = bars[(c["exchange"], c["symbol"])]
        i0 = c["j"] if same_tf else bisect.bisect_left(T, c["etime"])
        deadline = _bt_deadline(c["etime"], c["exchange"], cfg.max_hold, eff_exit(cfg, c["exchange"]))
        hit = _bt_exit(T, O, H, L, C, i0, deadline, tgt, sl, cxs.get((c["exchange"], c["symbol"])), S, cfg.cloud_exit_mode)
        if hit is None:
            return None
        px, xt, reason = hit
        # a target / SL hit happens INSIDE its candle -> the slot + cash are free only when that candle ends
        blen = tf if same_tf else pd.Timedelta(minutes=1)
        release = xt if reason == "Time exit" else min(xt + blen, deadline) if reason != "Open at end of data" else xt + blen
        return px, xt, reason, release

    # the exit of every signal does not depend on the position size or the brokerage -> worked out once and stored, so Results can
    # re-calculate Normal / Compound / another brokerage instantly (bt_replay) without loading any candle again
    for c in cands:
        entry = c["entry_raw"] * (1 + cfg.slippage_pct / 100)
        c["tgt"] = round(entry * (1 + cfg.target_pct / 100), 2)
        c["sl"] = round(entry * (1 - cfg.sl_pct / 100), 2) if cfg.sl_pct > 0 else None
        c["hit"] = exits_for(c, c["tgt"], c["sl"])
    replay = _bt_make_replay(cands, sigs, cfg)
    trades, signals = bt_replay(replay, cfg)                # NORMAL result with the brokerage chosen in the setup
    return {"trades": trades, "signals": signals, "errors": errors, "tf_min": int(cfg.tf_min), "replay": replay}


def bt_stats(trades, capital, compound=False):
    """compound=True -> the drawdown is measured as % of the running equity peak (positions grow with the equity)."""
    if trades is None or trades.empty:
        return {}
    pnl = trades["pnl"].astype(float)
    win = pnl > 0
    gp, gl = pnl[win].sum(), -pnl[~win].sum()
    if "brokerage_buy" in trades.columns and "brokerage_sell" in trades.columns:
        brk_buy, brk_sell = float(trades["brokerage_buy"].astype(float).sum()), float(trades["brokerage_sell"].astype(float).sum())
    else:
        brk_buy = brk_sell = 0.0
    brk, net = brk_buy + brk_sell, float(pnl.sum())
    gross = net + brk                                           # P&L before brokerage
    eq = capital + pnl.cumsum()
    peak = np.maximum(eq.cummax(), capital)
    dd = float((peak - eq).max())
    dd_pct = (float(np.nan_to_num(((peak - eq) / peak.where(peak > 0)).max() * 100)) if compound
              else (float(dd / capital * 100) if capital else 0.0))
    return {
        "net": net, "ret": float(net / capital * 100) if capital else 0.0,
        "n": int(len(pnl)), "wins": int(win.sum()), "losses": int((~win).sum()),
        "win_rate": float(win.mean() * 100), "pf": float(gp / gl) if gl > 0 else float("inf") if gp > 0 else 0.0,
        "avg": float(pnl.mean()), "best": float(pnl.max()), "worst": float(pnl.min()),
        "avg_win": float(pnl[win].mean()) if win.any() else 0.0,
        "avg_loss": float(pnl[~win].mean()) if (~win).any() else 0.0,
        "dd": dd, "dd_pct": dd_pct,
        "gross": gross, "brk": brk, "brk_buy": brk_buy, "brk_sell": brk_sell,
        "brk_pct": float(brk / gross * 100) if gross > 0 else 0.0, "end": float(capital + net),
    }


def _bt_group(trades, col):
    t = trades.copy()
    if col == "pattern":
        t["pattern"] = t["pattern"].str.split("+")
        t = t.explode("pattern")
    g = t.groupby(col)["pnl"].agg(trades="count", net_pnl="sum", avg_pnl="mean",
                                  win_rate=lambda s: (s > 0).mean() * 100).round(2)
    return g.sort_values("net_pnl", ascending=False).reset_index()


MIN_TRADES_BEST = 5          # a candle / stock needs at least this many trades to be crowned BEST / SAFEST (if any has)


def _hold_bars(entry, exit_, exchange, tf_min):
    """Holding time in candles (trading time only - nights / weekends are not counted)."""
    s_min, e_min = (9 * 60, 23 * 60 + 30) if exchange == "MCX" else (9 * 60 + 15, 15 * 60 + 30)
    entry, exit_ = pd.Timestamp(entry), pd.Timestamp(exit_)
    tod = lambda t: min(max(t.hour * 60 + t.minute, s_min), e_min)
    days = int(np.busday_count(entry.date(), exit_.date()))
    mins = days * (e_min - s_min) + tod(exit_) - tod(entry)
    unit = (e_min - s_min) if tf_min >= 1440 else tf_min          # 1-day candle = one whole session
    return max(mins, 0) / unit


def _bt_rank(trades, by, capital, tf_min):
    """One row per candle / stock / stock+candle: return, win rate, trades still open, avg hold  ->  ranked, with a BEST / SAFEST tag."""
    t = trades.copy()
    if "pattern" in by:                                      # a signal with 2 patterns counts for both
        t["pattern"] = t["pattern"].str.split("+")
        t = t.explode("pattern")
    t["_open"] = (t["reason"] == "Open at end of data").astype(int)
    t["_win"] = (t["pnl"] > 0).astype(int)
    t["_hold"] = [_hold_bars(a, b, e, tf_min) for a, b, e in zip(t["entry_time"], t["exit_time"], t["exchange"])]
    has_brk = "brokerage_total" in t.columns
    agg = dict(trades=("pnl", "count"), wins=("_win", "sum"), net_pnl=("pnl", "sum"),
               avg_pnl=("pnl", "mean"), still_open=("_open", "sum"), avg_hold_bars=("_hold", "mean"))
    if has_brk:
        agg["brokerage"] = ("brokerage_total", "sum")
    g = t.groupby(list(by)).agg(**agg).reset_index()
    g["win_rate"] = g["wins"] / g["trades"] * 100
    g["return_pct"] = g["net_pnl"] / capital * 100 if capital else 0.0
    g["open_pct"] = g["still_open"] / g["trades"] * 100
    enough = (g["trades"] >= MIN_TRADES_BEST).any()
    pool = g[g["trades"] >= MIN_TRADES_BEST] if enough else g
    best = pool.sort_values(["net_pnl", "trades"], ascending=[False, False]).index[0]
    safe = pool.sort_values(["open_pct", "avg_hold_bars", "net_pnl"], ascending=[True, True, False]).index[0]
    g["tag"] = ""
    g.loc[safe, "tag"] = "SAFEST"
    g.loc[best, "tag"] = "BEST + SAFEST" if best == safe else "BEST"
    g = g.sort_values(["net_pnl"], ascending=False).reset_index(drop=True)
    g.insert(0, "rank", range(1, len(g) + 1))
    cols = (["rank", *by, "tag", "trades", "win_rate", "net_pnl"] + (["brokerage"] if has_brk else [])
            + ["return_pct", "avg_pnl", "still_open", "open_pct", "avg_hold_bars"])
    g = g[cols].round({"win_rate": 1, "net_pnl": 2, "brokerage": 2, "return_pct": 2, "avg_pnl": 2, "open_pct": 1, "avg_hold_bars": 1})
    return g, g[g["tag"].str.contains("BEST")].iloc[0], g[g["tag"].str.contains("SAFEST")].iloc[0], enough


def _bt_table(g):
    """Ranked table: the BEST / SAFEST rows are highlighted green."""
    hi = lambda r: ["background-color: rgba(46,160,67,0.28)" if r["tag"] else ""] * len(r)
    fmt = {"win_rate": "{:.1f}%", "return_pct": "{:+.2f}%", "open_pct": "{:.1f}%", "net_pnl": "{:,.2f}",
           "brokerage": "{:,.2f}", "avg_pnl": "{:,.2f}", "avg_hold_bars": "{:.1f}"}
    return g.style.apply(hi, axis=1).format({k: v for k, v in fmt.items() if k in g.columns})


def _bt_quick(trades, capital, tf_min):
    """QUICK ANSWER card - the best / safest candle, stock and stock+candle, whether 1 or many were selected."""
    st.markdown("#### Quick answer")
    for title, by in (("Candle", ["pattern"]), ("Stock", ["symbol"]), ("Stock + candle", ["symbol", "pattern"])):
        if title == "Stock + candle" and trades["symbol"].nunique() < 2 and trades["pattern"].str.split("+").explode().nunique() < 2:
            continue                                         # 1 stock + 1 candle - nothing to compare
        g, best, safe, enough = _bt_rank(trades, by, capital, tf_min)
        name = lambda r: " · ".join(str(r[c]) for c in by)
        line = lambda r: (f"**{name(r)}** → **{r['return_pct']:+.2f}%** · {int(r['trades'])} trades, {int(r['still_open'])} still open "
                          f"· win {r['win_rate']:.0f}% · avg hold {r['avg_hold_bars']:.1f} bars")
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.success("🏆 BEST  " + line(best))
            if name(safe) == name(best):
                st.info("🛡 SAFEST  same as Best, so no trade-off here.")
            else:
                st.info("🛡 SAFEST  " + line(safe))
            st.caption(f"BEST = highest net return · SAFEST = fewest trades still open (no stop-loss, so open trades are the real risk), "
                       f"ties → shorter hold. " + (f"Only {title.lower()}s with at least {MIN_TRADES_BEST} trades are compared."
                                                    if enough else f"Every {title.lower()} has fewer than {MIN_TRADES_BEST} trades - treat with caution."))


def _bt_by_index(trades, imap, capital, tf_min, compound=False):
    """Result split per index. A stock that belongs to two chosen indices counts in both."""
    t = trades.copy()
    t["index"] = t["symbol"].map(imap)
    t = t.dropna(subset=["index"]).explode("index")
    if t.empty:
        st.info("No trades belong to the chosen indices.")
        return
    st.caption("Every index, best first. Numbers are slices of the SAME combined run (shared capital / max positions), "
               "so they are not a separate backtest of each index, and a stock in two indices is counted in both.")
    st.dataframe(_bt_table(_bt_rank(t, ["index"], capital, tf_min)[0]), use_container_width=True, hide_index=True)
    pick = st.selectbox("See one index in detail", sorted(t["index"].unique()), key="bt_index_view")
    one = t[t["index"] == pick].drop(columns=["index"]).reset_index(drop=True)
    s = bt_stats(one, capital, compound)
    c = st.columns(4)
    c[0].metric("Net P&L", f"{s['net']:,.0f}", f"{s['ret']:.2f}% of capital")
    c[1].metric("Trades", s["n"], f"{s['wins']}W / {s['losses']}L", delta_color="off")
    c[2].metric("Win rate", f"{s['win_rate']:.1f}%")
    c[3].metric("Profit factor", "inf" if math.isinf(s["pf"]) else f"{s['pf']:.2f}")
    st.line_chart(one.set_index("exit_time")["pnl"].cumsum().rename("Cumulative P&L"))
    st.markdown("**By stock**")
    st.dataframe(_bt_table(_bt_rank(one, ["symbol"], capital, tf_min)[0]), use_container_width=True, hide_index=True)
    st.markdown("**By pattern**")
    st.dataframe(_bt_table(_bt_rank(one, ["pattern"], capital, tf_min)[0]), use_container_width=True, hide_index=True)
    st.markdown("**Trades**")
    st.dataframe(one, use_container_width=True, hide_index=True)
    _csv_button(f"⬇ {pick} trades CSV", one, f"backtest_trades_{pick.replace(' ', '_')}", "bt_dl_index_trades")


# ============================== INDEX COMPARISON (benchmark = the index's own candles from Angel One) ==============================
POPULAR_INDICES = ("Nifty 50", "Nifty Bank", "SENSEX", "Nifty IT", "Nifty Next 50", "Nifty Midcap 100",
                   "Nifty Smallcap 250", "Nifty 500",
                   "Nifty Financial Services", "Nifty Auto", "Nifty Pharma", "Nifty FMCG", "Nifty Metal")
_INDEX_ALIAS = {"sensex30": "sensex", "niftyfinancialservices": "niftyfinservice",
                "niftysmallcap50": "niftysmlcap50", "niftysmallcap100": "niftysmlcap100", "niftysmallcap250": "niftysmlcap250",
                "niftymidsmallcap400": "niftymidsmlcap400", "niftytotalmarket": "niftytotalmkt"}     # setup name -> Angel One name (normalised)


def _norm_name(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


@st.cache_data(ttl=86400, show_spinner=False)
def angel_indices():
    """Index instruments of the Angel One scrip master (NSE / BSE) -> [{label, exchange, token, symbol, tradingsymbol}], popular first."""
    try:
        d = load_master()
    except Exception:
        return []
    d = d[d["exch_seg"].isin(["NSE", "BSE"])]
    it = d["instrumenttype"].fillna("").astype(str).str.upper() if "instrumenttype" in d else pd.Series("", index=d.index)
    d = d[it.str.contains("IDX") & ~it.str.startswith(("FUT", "OPT"))]
    rows, seen = [], set()
    for r in d.to_dict("records"):
        sym, nm = str(r.get("symbol") or "").strip(), str(r.get("name") or "").strip()
        base = sym or nm                                       # symbol is the readable one ("Nifty 50"), name is a code ("NIFTY")
        if not base:
            continue
        label = base if base not in seen else f"{base} ({r['exch_seg']})"
        if label in seen:
            continue
        seen.add(label)
        rows.append({"label": label, "exchange": r["exch_seg"], "token": str(r["token"]),
                     "symbol": base, "tradingsymbol": sym or base, "keys": {_norm_name(sym), _norm_name(nm)} - {""}})
    pop = {_norm_name(x): n for n, x in enumerate(POPULAR_INDICES)}
    rows.sort(key=lambda r: (min([pop[k] for k in r["keys"] if k in pop] or [999]), r["label"].lower()))
    return rows


def _bt_index_candles(smart, ix, d_from, d_to):
    """Daily candles of one index for From..To (session cache; _bt_load also caches finished chunks on disk)."""
    cache = st.session_state.setdefault("bt_idx_cache", {})
    key = (ix["exchange"], ix["token"], str(d_from), str(d_to))
    if key not in cache:
        start = pd.Timestamp(d_from)
        end = min(pd.Timestamp(d_to) + pd.Timedelta(days=1), now_ist() + pd.Timedelta(days=1))
        try:
            d = _bt_load(smart, ix, "ONE_DAY", start, end)
        except Exception:
            d = pd.DataFrame()
        if d is None or d.empty:
            return pd.DataFrame()                                  # not cached -> tried again next time
        cache[key] = d
    return cache[key]


def _bt_index_stats(d):
    """Buy & hold of the index: first day OPEN -> last day CLOSE, plus its max drawdown and a daily % curve."""
    d = d.sort_values("date")
    close = d["close"].astype(float)
    first_open, last_close = float(d["open"].iloc[0]), float(close.iloc[-1])
    px = pd.concat([pd.Series([first_open]), close], ignore_index=True)
    dd = float(((px.cummax() - px) / px.cummax()).max() * 100)
    curve = pd.Series((close.to_numpy() / first_open - 1) * 100, index=pd.to_datetime(d["date"]).dt.normalize().to_numpy())
    curve = curve[~curve.index.duplicated(keep="last")]
    return {"ret": (last_close / first_open - 1) * 100, "dd_pct": dd, "first": first_open, "last": last_close, "curve": curve}


def _bt_strategy_curve(trades, capital):
    """Cumulative backtest P&L as % of capital, one point per exit day."""
    x = pd.to_datetime(trades["exit_time"], errors="coerce")
    if getattr(x.dt, "tz", None) is not None:
        x = x.dt.tz_localize(None)
    return trades["pnl"].astype(float).groupby(x.dt.normalize()).sum().cumsum() / capital * 100


def _bt_index_compare(res, trades, capital, compound=False):
    """'Compare with an index' box on top of the result: index chosen from Angel One's index list vs this backtest."""
    ss = st.session_state
    setup = res.get("setup") or {}
    opts = angel_indices()
    with st.expander("📊 Compare with an index (Angel One)", expanded=True):
        if not opts:
            st.info("Index list could not be loaded from the Angel One scrip master.")
            return
        by_label = {o["label"]: o for o in opts}
        if "bt_cmp_idx" not in ss:                          # first time: the index chosen in the setup (if Angel One has it)
            want = {_INDEX_ALIAS.get(_norm_name(x), _norm_name(x)) for x in (setup.get("index_pick") or [])}
            ss["bt_cmp_idx"] = [o["label"] for o in opts if o["keys"] & want][:3]
        else:
            ss["bt_cmp_idx"] = [x for x in ss["bt_cmp_idx"] if x in by_label]
        pick = st.multiselect("Index to compare with", list(by_label), key="bt_cmp_idx",
                              help="Index list comes from Angel One. Its own candles for the same From - To dates are loaded and "
                                   "compared with this backtest.")
        if not pick:
            st.caption("Select an index (Nifty 50, Nifty Bank, SENSEX ...) to compare its result with this backtest.")
            return
        smart = ss.get("smart")
        if smart is None:
            st.warning("Angel One login is required to load index candles.")
            return
        t0, t1 = pd.to_datetime(trades["entry_time"]).min().date(), pd.to_datetime(trades["exit_time"]).max().date()
        d_from, d_to = _date_or(setup.get("d_from"), t0), _date_or(setup.get("d_to"), t1)
        stats, miss = {}, []
        with st.spinner("Loading index candles..."):
            for lb in pick:
                d = _bt_index_candles(smart, by_label[lb], d_from, d_to)
                if not d.empty:
                    d = d[(d["date"] >= pd.Timestamp(d_from)) & (d["date"] < pd.Timestamp(d_to) + pd.Timedelta(days=1))]
                if d.empty:
                    miss.append(lb)
                else:
                    stats[lb] = _bt_index_stats(d)
        if miss:
            st.warning("No candle data for: " + ", ".join(miss))
        if not stats:
            return
        s = bt_stats(trades, capital, compound)
        rows = [{"": "My backtest (compound)" if compound else "My backtest", "Return": f"{s['ret']:+.2f}%", "Max drawdown": f"-{s['dd_pct']:.2f}%",
                 "From → To": f"{capital:,.0f} → {capital + s['net']:,.0f}", "My backtest vs index": ""}]
        for lb, x in stats.items():
            diff = s["ret"] - x["ret"]
            rows.append({"": lb, "Return": f"{x['ret']:+.2f}%", "Max drawdown": f"-{x['dd_pct']:.2f}%",
                         "From → To": f"{x['first']:,.2f} → {x['last']:,.2f}",
                         "My backtest vs index": f"{diff:+.2f} pts ({'ahead' if diff >= 0 else 'behind'})"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        curves = {"My backtest (% of capital)": _bt_strategy_curve(trades, capital)}
        curves.update({f"{lb} (%)": x["curve"] for lb, x in stats.items()})
        st.line_chart(pd.concat(curves, axis=1, sort=True).sort_index().ffill().fillna(0.0))
        st.caption(f"Index = buy & hold from the first day's open to the last day's close of {d_from:%d %b %Y} → {d_to:%d %b %Y} "
                   "(daily candles from Angel One). Backtest return = net P&L ÷ total capital - the capital is not fully "
                   "invested all the time, so this is a rough benchmark, not a like-for-like return.")


ALL_INDICES = "All indices (combined)"


def _bt_brokerage_box(replay, uid):
    """Brokerage controls on top of the Results. Optional (switch) and editable - the whole Results re-calculates when a value
       changes, nothing is downloaded again. The setup form only gives the starting values."""
    c0 = replay.get("cfg") or {}
    b0, s0 = float(c0.get("brokerage_buy_pct") or 0), float(c0.get("brokerage_sell_pct") or 0)
    fb0, fs0 = float(c0.get("brokerage_buy_flat") or 0) or 20.0, float(c0.get("brokerage_sell_flat") or 0) or 20.0
    flat0 = str(c0.get("brokerage_mode") or "") == "flat"
    any0 = (float(c0.get("brokerage_buy_flat") or 0) > 0 or float(c0.get("brokerage_sell_flat") or 0) > 0) if flat0 else (b0 > 0 or s0 > 0)
    on0 = bool(c0.get("brokerage_on", True)) and any0
    modes = ["% of trade value", "₹ per order"]
    K = lambda n: f"bt_rb_{n}_{uid}"
    with st.container(border=True):
        _section("Brokerage", "optional - change it here and every number below updates")
        c1, c2, c3, c4 = st.columns([1.3, 1.5, 1.4, 1.4])
        on = c1.toggle("Apply brokerage", value=on0, key=K("on"),
                       help="Off = results without any brokerage (gross). On = brokerage is deducted from every trade.")
        mode = c2.selectbox("Charged as", modes, index=1 if flat0 else 0, key=K("mode"), disabled=not on,
                            help="% of trade value = buy value / sell value x %   |   ₹ per order = a fixed amount on each buy and each sell order")
        flat = mode == modes[1]
        if flat:
            bf = c3.number_input("Buy order (₹)", min_value=0.0, step=1.0, format="%.2f", value=fb0, key=K("bf"), disabled=not on)
            sf = c4.number_input("Sell order (₹)", min_value=0.0, step=1.0, format="%.2f", value=fs0, key=K("sf"), disabled=not on)
            bp, sp = b0, s0
        else:
            bp = c3.number_input("Buy leg (%)", min_value=0.0, max_value=5.0, step=0.01, format="%.2f", value=b0, key=K("bp"), disabled=not on)
            sp = c4.number_input("Sell leg (%)", min_value=0.0, max_value=5.0, step=0.01, format="%.2f", value=s0, key=K("sp"), disabled=not on)
            bf, sf = fb0, fs0
    return {"on": bool(on), "mode": "flat" if flat else "pct", "buy_pct": float(bp), "sell_pct": float(sp),
            "buy_flat": float(bf), "sell_flat": float(sf)}


def _bt_views(res, uid, brk):
    """{'Normal': (trades, signals), 'Compound': (trades, signals)} - replayed with the brokerage chosen in Results (cached)."""
    cache = st.session_state.setdefault("bt_replay_cache", {})
    key = (uid, brk["on"], brk["mode"], brk["buy_pct"], brk["sell_pct"], brk["buy_flat"], brk["sell_flat"])
    if key not in cache:
        if len(cache) > 16:
            cache.clear()
        d = dict(res["replay"]["cfg"])
        d.update(brokerage_on=brk["on"], brokerage_mode=brk["mode"], brokerage_buy_pct=brk["buy_pct"],
                 brokerage_sell_pct=brk["sell_pct"], brokerage_buy_flat=brk["buy_flat"], brokerage_sell_flat=brk["sell_flat"])
        with st.spinner("Recalculating..."):
            cache[key] = {name: bt_replay(res["replay"], _cfg_from_data({**d, "compound": comp}))
                          for name, comp in (("Normal", False), ("Compound", True))}
    return cache[key]


def _bt_compare_table(views, capital, show_brk):
    """Normal and Compound side by side."""
    rows = []
    for name, (t, _sg) in views.items():
        s = bt_stats(_bt_trades_view(t, show_brk), capital, name == "Compound")
        if not s:
            rows.append({"Result": name, "Net P&L": "0", "Return": "0.00%", "End capital": f"{capital:,.0f}", "Trades": 0})
            continue
        row = {"Result": name, "Net P&L": f"{s['net']:,.0f}", "Return": f"{s['ret']:+.2f}%", "End capital": f"{s['end']:,.0f}",
               "Trades": s["n"], "Win rate": f"{s['win_rate']:.1f}%",
               "Profit factor": "inf" if math.isinf(s["pf"]) else f"{s['pf']:.2f}",
               "Max drawdown": f"-{s['dd_pct']:.2f}%"}
        if show_brk:
            row["Gross P&L"], row["Brokerage"] = f"{s['gross']:,.0f}", f"{s['brk']:,.0f}"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Max drawdown: Normal = % of the starting capital, Compound = % from the highest equity reached "
               "(compound positions grow, so a fixed rupee amount would not be comparable).")


def _bt_trades_view(trades, show_brk):
    """Trades for display: brokerage columns only when brokerage is applied (older saved files get them worked out)."""
    t = trades.copy()
    if show_brk and "brokerage_total" not in t.columns and {"brokerage_buy", "brokerage_sell"} <= set(t.columns):
        t["brokerage_total"] = (t["brokerage_buy"] + t["brokerage_sell"]).round(2)
        t["gross_pnl"] = (t["pnl"] + t["brokerage_total"]).round(2)
    if not show_brk:
        t = t.drop(columns=[c for c in ("gross_pnl", "brokerage_buy", "brokerage_sell", "brokerage_total") if c in t.columns])
    return t


def _bt_show(res, capital):
    ss = st.session_state
    for e in res["errors"][:15]:
        st.warning(e)
    if len(res["errors"]) > 15:
        st.caption(f"... and {len(res['errors']) - 15} more data warnings")
    uid = res.setdefault("_uid", time.time_ns())            # new run / opened file -> fresh Results widgets
    replay = res.get("replay")
    compound, view_name = False, "normal"
    if replay and replay.get("cfg"):
        brk = _bt_brokerage_box(replay, uid)                 # optional + editable brokerage
        views = _bt_views(res, uid, brk)
        show_brk = brk["on"]
        mode = st.radio("Result type", ["Normal", "Compound"], horizontal=True, key=f"bt_mode_{uid}",
                        help="Normal = every position is sized from the STARTING capital.  Compound = profits / losses are "
                             "re-invested: every new position is sized from the CURRENT capital (starting capital + net P&L of "
                             "the closed trades). Qty mode: the qty is scaled the same way and rounded down.")
        compound, view_name = mode == "Compound", mode.lower()
        _bt_compare_table(views, capital, show_brk)
        trades, signals = views[mode]
        st.caption(("COMPOUND: every new position is sized from the current capital (starting capital + closed P&L)."
                    if compound else "NORMAL: every position is sized from the starting capital (profits are not re-invested)."))
    else:
        trades, signals = res["trades"], res["signals"]
        show_brk = bool({"brokerage_buy", "brokerage_sell"} <= set(trades.columns)
                        and (trades["brokerage_buy"].sum() + trades["brokerage_sell"].sum()) > 0)
        st.info("This result was saved by an older version, so Compound and editable brokerage are not available for it. "
                "Run the backtest again (Edit in Strategies) to get them.")
    trades = _bt_trades_view(trades, show_brk)
    imap = res.get("index_map") or {}
    if imap:                                                 # 2+ indices were chosen -> list them; default = full combined result
        ix_list = sorted({ix for v in imap.values() for ix in v})
        opts = [ALL_INDICES] + ix_list
        if st.session_state.get("bt_index_filter") not in opts:
            st.session_state.pop("bt_index_filter", None)
        pick_ix = st.selectbox("Show result for", opts, key="bt_index_filter",
                               help="Default = everything combined. Pick one index to see only its stocks' trades, "
                                    "stats, charts and tables.")
        if pick_ix != ALL_INDICES:
            syms = {sy for sy, v in imap.items() if pick_ix in v}
            trades = trades[trades["symbol"].isin(syms)].reset_index(drop=True)
            if signals is not None and not signals.empty:
                signals = signals[signals["symbol"].isin(syms)].reset_index(drop=True)
            res = {**res, "index_map": {}}                   # one index only -> no 'By index' comparison tab
            st.caption(f"Showing only **{pick_ix}** stocks ({len(syms)}). Numbers are a slice of the same combined run "
                       f"(shared capital / max positions); a stock in two indices shows under both.")
    if trades.empty:
        st.info("No trades in this period (no signal passed the filters). Try a longer period / more patterns / more stocks.")
        if signals is not None and not signals.empty:
            st.dataframe(signals, use_container_width=True, hide_index=True)
        return
    _bt_index_compare(res, trades, capital, compound)        # index (from Angel One) vs this backtest, on top of the result
    s = bt_stats(trades, capital, compound)
    r1 = st.columns(4)
    r1[0].metric("Net P&L" if show_brk else "P&L", f"{s['net']:,.0f}", f"{s['ret']:.2f}% of capital")
    r1[1].metric("Trades", s["n"], f"{s['wins']}W / {s['losses']}L", delta_color="off")
    r1[2].metric("Win rate", f"{s['win_rate']:.1f}%")
    r1[3].metric("Profit factor", "inf" if math.isinf(s["pf"]) else f"{s['pf']:.2f}")
    rb = st.columns(4)
    if show_brk:
        rb[0].metric("Gross P&L (before brokerage)", f"{s['gross']:,.0f}")
        rb[1].metric("Total brokerage", f"{s['brk']:,.0f}", f"{s['brk_pct']:.1f}% of gross profit" if s["gross"] > 0 else None,
                     delta_color="off")
        rb[2].metric("Buy / sell brokerage", f"{s['brk_buy']:,.0f} / {s['brk_sell']:,.0f}")
    rb[3].metric("End capital", f"{s['end']:,.0f}", f"{s['ret']:+.2f}%")
    r2 = st.columns(4)
    r2[0].metric("Avg P&L / trade", f"{s['avg']:,.0f}")
    r2[1].metric("Avg win / avg loss", f"{s['avg_win']:,.0f} / {s['avg_loss']:,.0f}")
    r2[2].metric("Best / worst trade", f"{s['best']:,.0f} / {s['worst']:,.0f}")
    r2[3].metric("Max drawdown", f"{s['dd']:,.0f}",
                 f"-{s['dd_pct']:.2f}% from equity peak" if compound else f"-{s['dd_pct']:.2f}% of capital", delta_color="off")

    tf_min = int(res.get("tf_min", 5))
    _bt_quick(trades, capital, tf_min)

    eq = trades.set_index("exit_time")["pnl"].cumsum().rename("Cumulative P&L")
    st.line_chart(eq)

    imap = res.get("index_map") or {}
    names = ["Trades", "Signals", "By stock", "By pattern", "Best combos", "By day"] + (["By index"] if imap else [])
    tabs = st.tabs(names)
    t_tr, t_sg, t_st, t_pt, t_bc, t_dy = tabs[:6]
    if imap:
        with tabs[6]:
            _bt_by_index(trades, imap, capital, tf_min, compound)
    with t_tr:
        st.dataframe(trades, use_container_width=True, hide_index=True)
        _csv_button("⬇ Trades CSV", trades, f"backtest_trades_{view_name}", "bt_dl_trades")
    with t_sg:
        if signals is not None and not signals.empty:
            st.caption(signals["status"].str.replace(r"\s*\(.*", "", regex=True).value_counts().to_dict())
            st.dataframe(signals, use_container_width=True, hide_index=True)
            _csv_button("⬇ Signals CSV", signals, "backtest_signals", "bt_dl_signals")
    with t_st:
        st.dataframe(_bt_table(_bt_rank(trades, ["symbol"], capital, tf_min)[0]), use_container_width=True, hide_index=True)
    with t_pt:
        st.dataframe(_bt_table(_bt_rank(trades, ["pattern"], capital, tf_min)[0]), use_container_width=True, hide_index=True)
    with t_bc:
        st.caption("Every stock + candle pair, best first.")
        st.dataframe(_bt_table(_bt_rank(trades, ["symbol", "pattern"], capital, tf_min)[0]), use_container_width=True, hide_index=True)
    with t_dy:
        d = trades.assign(day=pd.to_datetime(trades["exit_time"]).dt.date)
        dagg = dict(trades=("pnl", "count"), net_pnl=("pnl", "sum"))
        if "brokerage_total" in d.columns:
            dagg["brokerage"] = ("brokerage_total", "sum")
        day = d.groupby("day").agg(**dagg).round(2).reset_index()
        st.bar_chart(day.set_index("day")["net_pnl"])
        st.dataframe(day, use_container_width=True, hide_index=True)
    st.caption("Backtest = history replay with today's index members (survivorship bias), entry at next candle open, "
               "SL checked before target inside one candle. Live paper results can differ slightly. "
               "Brokerage changes made in Results are not stored in the saved file - the file keeps the brokerage used in the run.")


# ============================== SAVED STRATEGIES (folders -> files) ==============================
# STRATEGIES_ROOT/<folder>/<file>.json       = name, saved time, setup (every input) and a small results summary
# STRATEGIES_ROOT/<folder>/<file>.res.json   = the full result (trades, signals ...) - only read when a file is opened
CREATE_FOLDER = "+ Create new folder"
DEFAULT_FOLDER = "Unsorted"
_RES_SUFFIX = ".res.json"


def _safe_name(s, maxlen=60):
    """Folder / file name safe for the disk (no slashes, no dots -> no path tricks, no clash with '.res.json')."""
    s = re.sub(r'[\\/:*?"<>|\x00-\x1f.]+', " ", str(s or ""))
    return re.sub(r"\s+", " ", s).strip()[:maxlen].strip()


def _folder_dir(name):
    name = _safe_name(name)
    if not name:
        return None
    root = os.path.realpath(STRATEGIES_ROOT)
    p = os.path.realpath(os.path.join(root, name))
    return p if os.path.dirname(p) == root else None


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, default=_json_default)
    os.replace(tmp, path)


def _is_strategy_file(fn):
    return fn.endswith(".json") and not fn.endswith(_RES_SUFFIX)


def list_folders():
    """[(folder name, number of files)] sorted by name."""
    out = []
    try:
        for d in sorted(os.listdir(STRATEGIES_ROOT), key=str.lower):
            p = os.path.join(STRATEGIES_ROOT, d)
            if os.path.isdir(p):
                out.append((d, sum(1 for f in os.listdir(p) if _is_strategy_file(f))))
    except OSError:
        pass
    return out


def list_files(folder):
    """Saved files of one folder (newest first) - only the small header file of each is read."""
    d = _folder_dir(folder)
    rows = []
    if not d or not os.path.isdir(d):
        return rows
    for fn in os.listdir(d):
        if not _is_strategy_file(fn):
            continue
        try:
            with open(os.path.join(d, fn), encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        meta["file"] = fn[:-5]
        meta.setdefault("name", meta["file"])
        rows.append(meta)
    rows.sort(key=lambda m: str(m.get("saved_at", "")), reverse=True)
    return rows


def _df_pack(df):
    if df is None:
        return None
    df = df.copy()
    for c in df.columns:                                       # timestamps are stored as IST-naive text
        if pd.api.types.is_datetime64_any_dtype(df[c]) and getattr(df[c].dt, "tz", None) is not None:
            df[c] = df[c].dt.tz_convert(IST).dt.tz_localize(None)
    dt = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    return {"dt": dt, "df": json.loads(df.to_json(orient="split", date_format="iso", date_unit="ms",
                                                  index=False, default_handler=str))}


def _df_unpack(p):
    if not p:
        return pd.DataFrame()
    df = pd.DataFrame(p["df"]["data"], columns=p["df"]["columns"])
    for c in p.get("dt", []):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in df.columns:                                       # numbers that came back as object (None mixed in)
        if df[c].dtype == object:
            nn = df[c].dropna()
            if len(nn) and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in nn):
                df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _replay_pack(rp):
    """Replay data (signals + their exits + run settings) -> json-able. Lets a saved file still switch Normal / Compound and edit the brokerage."""
    if not rp:
        return None
    return {"cands": _df_pack(rp["cands"]), "sigs": _df_pack(rp["sigs"]), "cfg": rp["cfg"]}


def _replay_unpack(p):
    if not p:
        return None
    return {"cands": _df_unpack(p.get("cands")), "sigs": _df_unpack(p.get("sigs")), "cfg": p.get("cfg") or {}}


def save_strategy(folder, fname, setup, res):
    """Create / replace folder/file. Returns True when an existing file was replaced."""
    d, stem = _folder_dir(folder), _safe_name(fname)
    if not d or not stem:
        raise ValueError("Folder name and file name are needed.")
    os.makedirs(d, exist_ok=True)
    main, resf = os.path.join(d, stem + ".json"), os.path.join(d, stem + _RES_SUFFIX)
    existed = os.path.exists(main)
    summary = None
    if res is not None:
        trades = res["trades"]
        if trades is not None and not trades.empty:
            s = bt_stats(trades, res["capital"])
            summary = {k: s[k] for k in ("net", "ret", "n", "win_rate", "pf")}
        else:
            summary = {"n": 0}
        _write_json(resf, {"trades": _df_pack(trades), "signals": _df_pack(res.get("signals")),
                           "errors": list(res.get("errors") or []), "tf_min": res.get("tf_min"),
                           "capital": res.get("capital"), "label": res.get("label"),
                           "index_map": res.get("index_map") or {}, "replay": _replay_pack(res.get("replay"))})
    elif os.path.exists(resf):
        os.remove(resf)                                        # replaced by a setup-only save -> no stale result
    _write_json(main, {"name": stem, "saved_at": now_ist().strftime("%Y-%m-%d %H:%M:%S"),
                       "setup": setup, "summary": summary, "has_result": res is not None})
    return existed


def load_strategy(folder, stem):
    """-> (header dict, result dict or None)"""
    d = _folder_dir(folder)
    if not d or os.path.basename(stem) != stem:
        raise ValueError("Bad file name.")
    with open(os.path.join(d, stem + ".json"), encoding="utf-8") as f:
        meta = json.load(f)
    res, rp = None, os.path.join(d, stem + _RES_SUFFIX)
    if meta.get("has_result") and os.path.exists(rp):
        with open(rp, encoding="utf-8") as f:
            r = json.load(f)
        res = {"trades": _df_unpack(r.get("trades")), "signals": _df_unpack(r.get("signals")),
               "errors": r.get("errors") or [], "tf_min": r.get("tf_min") or 5, "capital": r.get("capital"),
               "label": r.get("label") or "", "index_map": r.get("index_map") or {}, "setup": meta.get("setup"),
               "replay": _replay_unpack(r.get("replay"))}
    return meta, res


def _cb_open_strategy(folder, stem, with_result):
    """Open = setup + saved result back on the Backtest page.  Edit = only the setup (change it, run again, save over it)."""
    ss = st.session_state
    try:
        meta, res = load_strategy(folder, stem)
    except Exception as ex:
        ss["sf_msg"] = ("error", f"Could not open the file: {ex}")
        return
    ss["bt_restore"] = {**(meta.get("setup") or {}), "_folder": folder, "_file": meta.get("name", stem)}
    ss["bt_gen"] = ss.get("bt_gen", 0) + 1                    # new widget keys -> the form starts from the saved values
    ss.pop("bt_index_filter", None)
    ss.pop("bt_cmp_idx", None)
    if with_result and res is not None:
        ss["bt_result"] = res
    else:
        ss.pop("bt_result", None)
        if with_result:
            ss["sf_msg"] = ("info", "This file was saved without a result - opened for editing.")
    ss["bt_view"] = "backtest"


def _cb_delete_strategy(folder, stem):
    d = _folder_dir(folder)
    if d and os.path.basename(stem) == stem:
        for suf in (".json", _RES_SUFFIX):
            try:
                os.remove(os.path.join(d, stem + suf))
            except OSError:
                pass
    st.session_state["sf_msg"] = ("success", f"Deleted {stem}.")


def _cb_new_folder():
    ss = st.session_state
    nm = _safe_name(ss.get("sf_new_name"))
    d = _folder_dir(nm)
    if not d:
        ss["sf_msg"] = ("error", "Enter a folder name.")
    elif os.path.exists(d):
        ss["sf_msg"] = ("error", f"Folder '{nm}' already exists.")
    else:
        os.makedirs(d)
        ss["sf_sel"] = nm
        ss["sf_new_name"] = ""


def _cb_rename_folder(old):
    ss = st.session_state
    nm = _safe_name(ss.get(f"sf_ren_{old}"))
    a, b = _folder_dir(old), _folder_dir(nm)
    if not b:
        ss["sf_msg"] = ("error", "Enter a folder name.")
    elif nm == old:
        return
    elif os.path.exists(b) and os.path.normcase(a) != os.path.normcase(b):
        ss["sf_msg"] = ("error", f"Folder '{nm}' already exists.")
    else:
        os.rename(a, b)
        ss["sf_sel"] = nm


def _cb_delete_folder(name):
    d = _folder_dir(name)
    if d and os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
    st.session_state.pop("sf_sel", None)
    st.session_state["sf_msg"] = ("success", f"Deleted folder {name}.")


# ============================== LOGIN STORAGE (several saved Angel One logins) ==============================
SAVED_LOGINS_FILE = os.path.join(_DATA_DIR, "angel_logins.json")
OLD_LOGIN_FILE = os.path.join(_DATA_DIR, "angel_login.json")          # single login of the older version (migrated once)
_LOGIN_KEYS = ("api_key", "client_id", "mpin", "totp_secret")


def _write_logins(rows):
    tmp = SAVED_LOGINS_FILE + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)      # readable by the current user only
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(rows, f)
    os.replace(tmp, SAVED_LOGINS_FILE)
    os.chmod(SAVED_LOGINS_FILE, 0o600)


def _migrate_old_login():
    try:
        with open(OLD_LOGIN_FILE, encoding="utf-8") as f:
            d = json.load(f)
        row = {k: str(d.get(k, "")).strip() for k in _LOGIN_KEYS}
        if not all(row.values()):
            return []
        row["last_used"] = datetime.fromtimestamp(os.path.getmtime(OLD_LOGIN_FILE)).strftime("%Y-%m-%d %H:%M:%S")
        _write_logins([row])
        os.remove(OLD_LOGIN_FILE)
        return [row]
    except Exception:
        return []


def list_saved_logins():
    """Saved logins, most recently used first."""
    try:
        with open(SAVED_LOGINS_FILE, encoding="utf-8") as f:
            rows = json.load(f)
    except FileNotFoundError:
        rows = _migrate_old_login()
    except Exception:
        rows = []
    rows = [r for r in rows if isinstance(r, dict) and all(str(r.get(k, "")).strip() for k in _LOGIN_KEYS)]
    return sorted(rows, key=lambda r: str(r.get("last_used", "")), reverse=True)


def save_login(creds):
    """Add (or refresh) one login and stamp it as used now."""
    try:
        row = {k: str(v).strip() for k, v in zip(_LOGIN_KEYS, creds)}
        row["client_id"] = row["client_id"].upper()
        row["last_used"] = now_ist().strftime("%Y-%m-%d %H:%M:%S")
        rows = [r for r in list_saved_logins() if str(r["client_id"]).upper() != row["client_id"].upper()]
        _write_logins(rows + [row])
    except Exception:
        pass


def remove_saved_login(client_id):
    try:
        _write_logins([r for r in list_saved_logins() if str(r["client_id"]).upper() != str(client_id).upper()])
    except Exception:
        pass


def env_creds():
    c = tuple(os.getenv(k, "").strip() for k in ("ANGEL_API_KEY", "ANGEL_CLIENT_ID", "ANGEL_MPIN", "ANGEL_TOTP_SECRET"))
    if all(c):
        return c
    try:
        a = st.secrets["angel"]
        c = tuple(str(a.get(k, "")).strip() for k in ("api_key", "client_id", "mpin", "totp_secret"))
        if all(c):
            return c
    except Exception:
        pass
    return None


# ---- DEMO ID: one-click login on the login page, nobody has to type any key ----
# Put the demo account's 4 details in ONE of these places (the first one found is used):
#   1) Streamlit secrets  (.streamlit/secrets.toml  or  the Streamlit Cloud "Secrets" box):
#        [demo]
#        api_key = "..."
#        client_id = "..."
#        mpin = "..."
#        totp_secret = "..."
#   2) environment variables DEMO_API_KEY, DEMO_CLIENT_ID, DEMO_MPIN, DEMO_TOTP_SECRET
#   3) the DEMO_FALLBACK line below (only if you accept the keys being written inside this file)
DEMO_LABEL = "Demo ID"
DEMO_FALLBACK = ("", "", "", "")                               # (api_key, client_id, mpin, totp_secret)


def demo_creds():
    try:
        a = st.secrets["demo"]
        c = tuple(str(a.get(k, "")).strip() for k in _LOGIN_KEYS)
        if all(c):
            return c
    except Exception:
        pass
    c = tuple(os.getenv(k, "").strip() for k in ("DEMO_API_KEY", "DEMO_CLIENT_ID", "DEMO_MPIN", "DEMO_TOTP_SECRET"))
    if all(c):
        return c
    c = tuple(str(x).strip() for x in DEMO_FALLBACK)
    return c if all(c) else None


def _demo_card():
    """Demo ID row - one click logs in with the demo details, the visitor never sees or types a key."""
    creds = demo_creds()
    if not creds:
        return
    with st.container(border=True):
        c_av, c_info, c_go = st.columns([0.6, 5, 1.6], vertical_alignment="center")
        c_av.markdown("<div style='width:46px;height:46px;border-radius:50%;background:#3a2410;color:#f0a020;"
                      "display:flex;align-items:center;justify-content:center;font-weight:700'>DM</div>",
                      unsafe_allow_html=True)
        c_info.markdown(f"<div style='font-weight:700;font-size:1.05rem;line-height:1.3'>{html.escape(DEMO_LABEL)}</div>"
                        "<div style='opacity:.65;font-size:.92rem'>Click to login automatically - no keys needed</div>",
                        unsafe_allow_html=True)
        if c_go.button("▶ Login", key="demo_go", type="primary", use_container_width=True):
            _connect(creds, False)                             # False = the demo details are never written to the saved-logins file


def _last_used_text(stamp):
    try:
        d = datetime.strptime(str(stamp)[:10], "%Y-%m-%d").date()
    except ValueError:
        return "earlier"
    return "today" if d == now_ist().date() else f"{d.day} {d:%b}"


AVATARS = (("#7db4ff", "#0b2a52"), ("#22b23a", "#12301a"), ("#f0a020", "#3a2410"))      # (text, background)


def _connect(creds, remember):
    ss = st.session_state
    with st.spinner("Angel One login..."):
        sm, err = angel_login(*creds)
    if sm is None:
        st.error(err)
        return
    ss.smart, ss.client_id, ss.creds = sm, str(creds[1]).strip().upper(), tuple(creds)
    ss.pop("no_autologin", None)
    ss.pop("login_new", None)
    if remember:
        save_login(creds)                                      # new login is added / a saved one gets "last used: today"
    st.rerun()


def login_page():
    no_dim()
    ss = st.session_state
    st.title("Angel One login")
    st.caption("Login is used only to download historical candles. No orders are placed.")
    _demo_card()
    saved = list_saved_logins()
    if not saved:
        ss["login_new"] = True                                 # nobody saved yet -> normal login form directly
    if not ss.get("login_new"):
        with st.container(border=True):
            st.markdown("<div style='opacity:.85;margin-bottom:.2rem'>Saved logins</div>", unsafe_allow_html=True)
            for i, r in enumerate(saved):
                cid = str(r["client_id"]).strip()
                fg, bg = AVATARS[int(hashlib.md5(cid.upper().encode()).hexdigest(), 16) % len(AVATARS)]
                ini = html.escape(re.sub(r"[^A-Za-z0-9]", "", cid)[:2].upper() or "?")
                st.divider()
                c_av, c_info, c_go, c_del = st.columns([0.6, 5, 1.6, 0.6], vertical_alignment="center")
                c_av.markdown(f"<div style='width:46px;height:46px;border-radius:50%;background:{bg};color:{fg};"
                              f"display:flex;align-items:center;justify-content:center;font-weight:700'>{ini}</div>",
                              unsafe_allow_html=True)
                c_info.markdown(f"<div style='font-weight:700;font-size:1.05rem;line-height:1.3'>{html.escape(cid)}</div>"
                                f"<div style='opacity:.65;font-size:.92rem'>Last used {_last_used_text(r.get('last_used'))}</div>",
                                unsafe_allow_html=True)
                if c_go.button("🔑 Login", key=f"sl_go_{cid}", use_container_width=True):
                    _connect(tuple(str(r[k]).strip() for k in _LOGIN_KEYS), True)
                if c_del.button("🗑", key=f"sl_del_{cid}", type="tertiary", help="Remove this saved login"):
                    remove_saved_login(cid)
                    st.rerun()
            st.divider()
            if st.button("➕ Login with new account", key="sl_new", use_container_width=True):
                ss["login_new"] = True
                st.rerun()
        return
    with st.container(border=True):
        if saved and st.button("← Back to saved logins", key="sl_back", type="tertiary"):
            ss.pop("login_new", None)
            st.rerun()
        with st.form("angel_login", border=False):
            f1, f2, f3, f4 = st.columns(4)
            api_key = f1.text_input("API key", type="password")
            client_id = f2.text_input("Client ID", placeholder="S12345")
            mpin = f3.text_input("MPIN", type="password")
            totp = f4.text_input("TOTP secret", type="password")
            remember = st.checkbox("Remember this login on this device", value=True)
            go = st.form_submit_button("🔗 Connect", type="primary")
    if go:
        if not all(x.strip() for x in (api_key, client_id, mpin, totp)):
            st.error("Fill all 4 fields.")
            return
        _connect((api_key, client_id, mpin, totp), remember)


# ============================== APP HEADER (title + Backtest / Strategies tabs + Logout) ==============================
def _app_header():
    ss = st.session_state
    view = ss.get("bt_view", "backtest")
    c0, c1, c2, _sp, c3 = st.columns([2.2, 1.5, 1.7, 3.6, 1.1], vertical_alignment="center")
    c0.markdown("<div style='font-size:1.55rem;font-weight:700;white-space:nowrap'>Backtest app</div>", unsafe_allow_html=True)
    if c1.button("🧪 Backtest", key="nav_backtest", type="primary" if view == "backtest" else "tertiary",
                 use_container_width=True) and view != "backtest":
        ss["bt_view"] = "backtest"
        st.rerun()
    if c2.button("📁 Strategies", key="nav_strategies", type="primary" if view == "strategies" else "tertiary",
                 use_container_width=True) and view != "strategies":
        ss["bt_view"] = "strategies"
        st.rerun()
    if c3.button("Logout", key="bt_logout", type="tertiary", use_container_width=True):
        for k_ in ("smart", "client_id", "creds"):
            ss.pop(k_, None)
        ss["no_autologin"] = True                              # back to the login page - saved logins are kept
        ss.pop("login_new", None)
        st.rerun()
    st.divider()


# ============================== BACKTEST PAGE ==============================
# ============================== BULK STOCK NAMES FROM A FILE (Excel / CSV / TXT) ==============================
_NAME_HEADERS = ("symbol", "stock", "name", "script", "scrip", "company", "instrument", "ticker")


def read_stock_names(upload):
    """Stock names from an uploaded Excel / CSV / TXT file.
       Excel: first sheet; the column whose header says symbol / stock / name ... (else the first column).
       CSV / TXT: every value split by new line, comma, semicolon or tab."""
    raw, fname = upload.getvalue(), upload.name.lower()
    if fname.endswith((".xlsx", ".xlsm", ".xls")):
        try:
            df = pd.read_excel(io.BytesIO(raw), header=None, dtype=str)
        except ImportError:
            raise RuntimeError("Excel reading needs openpyxl - run: pip install openpyxl")
        if df.empty:
            return []
        col, start = 0, 0
        for j in range(df.shape[1]):
            h = str(df.iat[0, j]).strip().lower()
            if any(k in h for k in _NAME_HEADERS):
                col, start = j, 1
                break
        vals = df.iloc[start:, col].tolist()
    else:
        text = raw.decode("utf-8-sig", errors="ignore")
        vals = re.split(r"[\n\r,;\t]+", text)
    out, seen = [], set()
    for v in vals:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        v = str(v).strip()
        if v and v.lower() != "nan" and v.upper() not in seen:
            seen.add(v.upper())
            out.append(v)
    return out[:3000]


@st.cache_data(ttl=3600, show_spinner=False)
def exchange_universe(exchange):
    """EVERY tradable name of the exchange (not only the chosen index) - used to match uploaded names."""
    d = load_master()
    d = d[d["exch_seg"] == exchange].copy()
    nm = d["name"].fillna("").astype(str)
    sy = d["symbol"].fillna("").astype(str)
    if exchange == "NSE":
        d = d[sy.str.upper().str.endswith("-EQ")]
        d["label"] = d["name"]
    elif exchange == "BSE":
        d["label"] = d["symbol"]
    else:
        itype = d["instrumenttype"].fillna("") if "instrumenttype" in d else pd.Series("", index=d.index)
        d = d[itype.str.startswith("FUT")].copy()
        d["exp"] = pd.to_datetime(d["expiry"], format="%d%b%Y", errors="coerce")
        d = d[d["exp"] >= pd.Timestamp.now().normalize()].sort_values("exp").drop_duplicates("name")
        d["label"] = d["name"]
    return d.drop_duplicates("label").reset_index(drop=True)


def match_stock_names(names, uni):
    """-> (matched master rows, names not found). Matches name, symbol or symbol without -EQ (case-insensitive)."""
    lk = {}
    for i, (lb, sy, nm) in enumerate(zip(uni["label"], uni["symbol"], uni["name"])):
        for k in (lb, sy, nm, re.sub(r"-EQ$", "", str(sy).upper())):
            k = str(k).upper().strip()
            if k and k not in lk:
                lk[k] = i
    rows, miss, seen = [], [], set()
    for raw in names:
        key = re.sub(r"^(NSE|BSE|NFO|MCX)\s*:\s*", "", raw.upper().strip())
        key = re.sub(r"\.(NS|BO|BSE|NSE)$", "", key)
        i = lk.get(key)
        if i is None:
            i = lk.get(key + "-EQ")
        if i is None:
            miss.append(raw)
        elif i not in seen:
            seen.add(i)
            rows.append(uni.iloc[i].to_dict())
    return rows, miss


def _date_or(s, default):
    """Saved ISO date -> date (never later than today); falls back to the default."""
    try:
        return min(datetime.strptime(str(s)[:10], "%Y-%m-%d").date(), now_ist().date())
    except (TypeError, ValueError):
        return default


def backtest_page():
    """BACKTEST: same conditions as the Strategy Builder + From date / To date -> replay on historical candles."""
    ss = st.session_state
    g = ss.get("bt_gen", 0)                                   # +1 every time a saved file is opened / edited -> fresh widgets
    R = ss.get("bt_restore") or {}                            # saved setup (empty = normal blank form)
    K = lambda name: f"{name}_g{g}"
    smart = ss.get("smart")
    try:
        master = load_master()
    except Exception as ex:
        master = None
        st.warning(f"Stock list load aagala ({ex}).")

    if master is not None:
        _registry_sync()
    save_box = st.container(border=True)                      # SAVE TO FOLDER sits at the very top; filled in below
    today = now_ist().date()
    EXCH = ["NSE", "BSE", "NFO", "MCX"]
    TF_OPTS = list(TIMEFRAMES)
    picked, typed, sel_times, time_sel, up_labels = [], "", [], [], []
    with st.container(border=True):
        e1, e2, e3 = st.columns(3)
        exchange = e1.selectbox("Exchange", EXCH, index=EXCH.index(R["exchange"]) if R.get("exchange") in EXCH else 0,
                                key=K("bt_exchange"))
        capital = e2.number_input("Capital", min_value=10000.0, value=float(R["capital"]) if R.get("capital") else None,
                                  step=10000.0, placeholder="Capital", key=K("bt_capital"))
        tf_label = e3.selectbox("Candle timeframe", TF_OPTS, index=TF_OPTS.index(R["tf_label"]) if R.get("tf_label") in TF_OPTS else None,
                                placeholder="Select timeframe", key=K("bt_tf"))
        rx = R if R.get("exchange") == exchange else {}       # saved values that belong to the chosen exchange
        d1, d2, d3 = st.columns(3)
        d_from = d1.date_input("From date", value=_date_or(R.get("d_from"), today - pd.Timedelta(days=30)), max_value=today, key=K("bt_from"))
        d_to = d2.date_input("To date", value=_date_or(R.get("d_to"), today), max_value=today, key=K("bt_to"))
        precise = d3.checkbox("Accurate exits (1-minute candles - slower)", value=bool(R.get("precise")), key=K("bt_precise"),
                              help="Off = target / SL checked on the chosen timeframe candles (fast). "
                                   "On = checked on 1-minute candles like the live engine (more API calls).")

        with st.container(border=True):
            _section("Entry condition", "index, stocks, candle patterns, position size")
            ix1, ix2, _b = st.columns(3)
            with ix1:
                index_pick = _stock_picker("Index", index_options(exchange), K(f"bt_index_{exchange}"),
                                           default=rx.get("index_pick"), button_label="Select index")
            tf_min_ui = TIMEFRAMES[tf_label][1] if tf_label else None
            g_starts, g_ends = tf_time_grid(exchange, tf_min_ui) if tf_label else ([], [])
            entry_times = exit_at = ""
            rt = rx if rx.get("tf_label") == tf_label else {}
            with ix2:
                time_sel = _stock_picker("Time condition (optional)", ["Entry time", "Exit time"], K("bt_timecond"),
                                         default=R.get("time_sel"), button_label="Select entry / exit time")
            if time_sel:                                    # entry / exit time boxes appear ONLY after a time condition is chosen
                tc1, tc2, _tc3 = st.columns(3)
                if not tf_label:
                    st.caption("⏱ Select the candle timeframe first - the times follow the timeframe.")
                if "Entry time" in time_sel and tf_label:
                    with tc1:
                        if tf_min_ui < 1440:
                            sel_times = _stock_picker("Entry time", g_starts, K(f"bt_etimes_{exchange}_{tf_min_ui}"),
                                                      default=rt.get("entry_sel"), button_label="Select entry time")
                            entry_times = ",".join(sel_times)
                        else:
                            st.caption("⏱ 1-day candles always enter at the session open.")
                if "Exit time" in time_sel and tf_label:
                    default_x = exit_time(exchange).strftime("%H:%M")
                    x_opts = sorted(set(g_ends) | {default_x})
                    x_def = rt.get("exit_at") if rt.get("exit_at") in x_opts else default_x
                    exit_at = tc2.selectbox("Exit time (square-off)", x_opts, index=x_opts.index(x_def),
                                            key=K(f"bt_exit_{exchange}_{tf_min_ui}"),
                                            help="Open trades still running are closed at this time (on the last day of Max hold).")
            items, index_map = [], {}
            if master is not None:
                inst = exchange_list_cached(exchange, index_pick)
                labels = inst["label"].tolist()
                by_label = inst.set_index("label")
                ix_name = " + ".join(index_pick) if len(index_pick) <= 2 else f"{len(index_pick)} indices"
                r_picked = rx.get("picked") if (rx and rx.get("index_pick", []) == list(index_pick)) else None
                picked = _stock_picker(f"{ix_name or exchange} stocks ({len(labels)})", labels,
                                       K(f"bt_stocks_{exchange}_{'+'.join(index_pick) or 'all'}"),
                                       default=r_picked if r_picked is not None else (labels if index_pick else None))
                typed = st.text_input("Or type names (comma separated)", value=rx.get("typed", ""), key=K("bt_typed"))
                up = st.file_uploader("Or upload an Excel / CSV file with stock names", type=["xlsx", "xls", "csv", "txt"],
                                      key=K("bt_upload"),
                                      help="One stock name or symbol per row in the first column (a header row is optional). "
                                           "Any stock of the chosen exchange works - it does not have to be in the selected index. "
                                           "Excel: first sheet is read.")
                uni_ = exchange_universe(exchange)
                by_lb_ = {r["label"]: r for r in uni_.to_dict("records")}
                up_rows, up_miss, base_up, up_src = [], [], [], ""
                if up is not None:
                    try:
                        _m_rows, up_miss = match_stock_names(read_stock_names(up), uni_)
                        base_up = [r["label"] for r in _m_rows]
                        up_src = up.name
                    except Exception as ex:
                        st.error(f"Could not read the file: {ex}")
                elif rx.get("upload_names"):                        # a saved strategy brings its file stocks back into the box
                    base_up = [x for x in rx["upload_names"] if x in by_lb_]
                    up_src = "the saved strategy"
                if up is not None or base_up:
                    # STOCKS BOX: the stocks of the file as chips. x on a chip removes it; type in the box to ADD any other stock.
                    sig = hashlib.md5(up.getvalue()).hexdigest()[:8] if up is not None else "saved"
                    ub_key = K(f"bt_upbox_{exchange}_{sig}")        # new file -> new box (starts from that file's stocks)
                    if ub_key not in ss:
                        ss[ub_key] = list(base_up)
                    ss[ub_key] = [x for x in ss[ub_key] if x in by_lb_]
                    cur_up = ss[ub_key]
                    n_file = sum(1 for x in cur_up if x in set(base_up))
                    n_new = len(cur_up) - n_file
                    n_gone = len(base_up) - n_file
                    st.markdown("<style>.st-key-bt_upbox_wrap [data-baseweb='select'] > div:first-child"
                                "{max-height:150px;overflow-y:auto}</style>", unsafe_allow_html=True)
                    with st.container(border=True, key="bt_upbox_wrap"):
                        hb1, hb2 = st.columns([5, 1.6], vertical_alignment="center")
                        hb1.markdown(f"✅ **Stocks added from {html.escape(up_src)}** · {len(cur_up)}")
                        hb2.button("Reset to file", key=f"{ub_key}__reset", type="tertiary", use_container_width=True,
                                   on_click=lambda k_=ub_key, v_=list(base_up): ss.__setitem__(k_, v_))
                        st.multiselect("Stocks from the file", list(by_lb_), key=ub_key, label_visibility="collapsed",
                                       placeholder="Type a stock name to add more stocks here")
                        bits = [f"{n_file} from the file"]
                        if n_new:
                            bits.append(f"{n_new} added by you")
                        if n_gone:
                            bits.append(f"{n_gone} removed")
                        st.caption(" · ".join(bits) + "   (✕ on a stock removes it, type in the box to add a new one)")
                    if up_miss:
                        st.warning(f"{len(up_miss)} names not found in {exchange}: " + ", ".join(up_miss[:200])
                                   + (" ..." if len(up_miss) > 200 else ""), icon="⚠️")
                    up_rows = [by_lb_[x] for x in cur_up]
                up_labels = [r["label"] for r in up_rows]
                names, miss = list(picked), []
                for t in [x.strip().upper() for x in typed.split(",") if x.strip()]:
                    hit = inst[(inst["label"].str.upper() == t) | (inst["symbol"].str.upper() == t)
                               | (inst["name"].str.upper() == t)]
                    if hit.empty:
                        miss.append(t)
                    elif hit.iloc[0]["label"] not in names:
                        names.append(hit.iloc[0]["label"])
                if miss:
                    st.warning("Not found in " + exchange + ": " + ", ".join(miss))
                items = [to_item({**by_label.loc[n].to_dict(), "label": n}, exchange) for n in names]
                have_ = {it["symbol"] for it in items}
                for r_ in up_rows:                                  # stocks from the uploaded file (added to the picked / typed ones)
                    if r_["label"] not in have_:
                        have_.add(r_["label"])
                        items.append(to_item(r_, exchange))
                index_map = build_index_map(items, exchange, index_pick) if len(index_pick) >= 2 else {}
                st.caption(f"✅ Total {len(items)} stocks selected (of {len(labels)} in the chosen index list)"
                           + (f" · {len(up_labels)} from the file box" if up_labels else ""))
            picked_pats = _stock_picker(f"Candle patterns ({len(PATTERNS)})", list(PATTERNS), K("bt_patterns"),
                                        default=R.get("patterns") or list(PATTERNS), button_label="Select candle patterns")
            cl1, _c2, _c3 = st.columns(3)
            with cl1:
                cloud_sel = _stock_picker("Ichimoku cloud (optional)", [CLOUD_LABEL[k] for k in CLOUD_OPTIONS],
                                          K("bt_cloud"), default=R.get("cloud_sel"), button_label="Select cloud position (optional)")
            cloud_pick = cloud_text([CLOUD_BY_LABEL[x] for x in cloud_sel])

            NO_LIMIT = "Select max (no limit)"
            MAX_OPTS = [NO_LIMIT] + list(range(1, 26)) + [30, 40, 50]
            m1, m2, m3 = st.columns(3)
            max_pick = m1.selectbox("Max open positions", MAX_OPTS,
                                    index=MAX_OPTS.index(R["max_pick"]) if R.get("max_pick") in MAX_OPTS else 0, key=K("bt_max"))
            no_limit = max_pick == NO_LIMIT
            max_open = 10 ** 6 if no_limit else int(max_pick)
            def_amt = None if (no_limit or not capital) else round(capital / max_open, 2)
            if (R.get("size_mode") == "amount" and R.get("position_amount") and capital
                    and float(capital) == float(R.get("capital") or 0) and max_pick == R.get("max_pick")):
                def_amt = float(R["position_amount"])         # saved amount stays until capital / max positions change
            size_mode, position_amount, position_qty = _sizing_inputs(
                m2, m3, capital, max_open, no_limit, len(items), K("bt"),
                cur_mode=R.get("size_mode") or "amount", default_amt=def_amt, cur_qty=R.get("position_qty"))

        with st.container(border=True):
            _section("Exit condition", "target, optional stop loss / holding days")
            x1, x2, x3 = st.columns(3)
            target_pct = x1.number_input("Target (%)", min_value=0.01, value=float(R["target_pct"]) if R.get("target_pct") else None,
                                         step=0.01, format="%.2f", placeholder="Target %", key=K("bt_target"))
            sl_pct = x2.number_input("Stop loss (%) - optional", min_value=0.0, value=float(R.get("sl_pct") or 0.0), step=0.1,
                                     format="%.2f", key=K("bt_sl"))
            max_hold = x3.number_input("Max hold (days)", min_value=1, value=int(R.get("max_hold") or MAX_HOLD_DAYS), step=1,
                                       key=K("bt_hold"),
                                       help="1 = same-day square-off at 15:15 (NSE) / 23:15 (MCX) - or the Exit time chosen in Time condition.")
            cx1, cx2, _cx3 = st.columns(3)
            with cx1:
                cloud_exit_sel = _stock_picker("Ichimoku cloud exit (optional)", [CLOUD_LABEL[k] for k in CLOUD_OPTIONS],
                                               K("bt_cloud_exit"), default=R.get("cloud_exit_sel"),
                                               button_label="Select cloud exit position (optional)")
            cloud_exit_pick = cloud_text([CLOUD_BY_LABEL[x] for x in cloud_exit_sel])
            cloud_exit_mode = "OR"
            if cloud_exit_pick:                             # the AND | OR choice only matters when BOTH Target and Cloud exit are given
                with cx2:
                    cloud_exit_mode = st.radio("Target + Cloud exit", ["OR", "AND"], horizontal=True,
                                               index=1 if str(R.get("cloud_exit_mode") or "").upper() == "AND" else 0,
                                               key=K("bt_cloud_exit_mode"),
                                               help="OR = exit when EITHER the target or the cloud exit is met (whichever comes first). "
                                                    "AND = exit only when BOTH are met.")
                _pos = " / ".join(CLOUD_LABEL[k].lower() for k in cloud_list(cloud_exit_pick))
                if cloud_exit_mode == "AND":
                    st.caption(f"AND: once the target is touched, exit at the CLOSE of the first candle that closes {_pos}. "
                               "Stop loss and Max hold still exit the trade at any time.")
                else:
                    st.caption(f"OR: exit at the target, or at the CLOSE of the first candle that closes {_pos} - whichever comes first. "
                               "Stop loss and Max hold still apply.")
        with st.container(border=True):
            _section("Brokerage", "starting values (0 = no brokerage) - you can switch it off or change it in Results")
            b1, b2, _b3 = st.columns(3)
            brokerage_buy_pct = b1.number_input("Buy leg brokerage (%)", min_value=0.0, max_value=5.0, step=0.01, format="%.2f",
                                                value=float(R.get("brokerage_buy_pct", BROKERAGE_BUY_PCT)), key=K("bt_brk_buy"),
                                                help="Buy order brokerage = Buy value x this %   (Buy value = entry price x qty)")
            brokerage_sell_pct = b2.number_input("Sell leg brokerage (%)", min_value=0.0, max_value=5.0, step=0.01, format="%.2f",
                                                 value=float(R.get("brokerage_sell_pct", BROKERAGE_SELL_PCT)), key=K("bt_brk_sell"),
                                                 help="Sell order brokerage = Sell value x this %   (Sell value = exit price x qty)")
        run = st.button("▶ Run backtest", type="primary", key=K("bt_run"))

    # everything the form holds right now - saved with the file so Open / Edit can bring it back
    setup = {"exchange": exchange, "capital": float(capital) if capital else None, "tf_label": tf_label,
             "d_from": d_from.isoformat(), "d_to": d_to.isoformat(), "precise": bool(precise),
             "index_pick": list(index_pick), "time_sel": list(time_sel), "entry_sel": list(sel_times), "exit_at": exit_at,
             "picked": list(picked), "typed": typed, "upload_names": list(up_labels), "patterns": list(picked_pats), "cloud_sel": list(cloud_sel),
             "max_pick": max_pick, "size_mode": size_mode,
             "position_amount": float(position_amount) if position_amount else None,
             "position_qty": int(position_qty) if position_qty else None,
             "target_pct": float(target_pct) if target_pct else None, "sl_pct": float(sl_pct or 0),
             "max_hold": int(max_hold), "cloud_exit_sel": list(cloud_exit_sel),
             "cloud_exit_mode": cloud_exit_mode,
             "brokerage_buy_pct": float(brokerage_buy_pct), "brokerage_sell_pct": float(brokerage_sell_pct)}

    # ---------------- WHERE THE RESULT IS SAVED (top of the page) - the save itself happens after Run backtest ----------------
    with save_box:
        st.markdown("<span style='font-size:1.05rem'>📁 <b>Folder and file</b></span>"
                    "<span style='opacity:.7'> &nbsp;(optional, the setup and its result are saved to Strategies after Run backtest)</span>",
                    unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)
        folder_names = [n for n, _ in list_folders()]
        f_opts = [CREATE_FOLDER] + folder_names
        folder_pick = f1.selectbox("Folder", f_opts, index=f_opts.index(R["_folder"]) if R.get("_folder") in f_opts else 0,
                                   key=K("sv_folder"))
        creating = folder_pick == CREATE_FOLDER
        new_folder = f2.text_input("New folder name (only when creating)", placeholder="Bank stocks",
                                   disabled=not creating, key=K("sv_newfolder"))
        file_name = f3.text_input("File name", value=R.get("_file", ""), placeholder="Engulfing cloud above", key=K("sv_file"))

    if run:
        missing = [n for n, v in (("Capital", capital), ("Candle timeframe", tf_label),
                                  (SIZE_LABELS[size_mode], position_qty if size_mode == "qty" else position_amount),
                                  ("Target %", target_pct)) if not v]
        if missing:
            st.error("Fill: " + ", ".join(missing))
        elif d_from > d_to:
            st.error("From date must be on or before To date.")
        elif entry_times and exit_at and exit_at <= min(entry_times.split(",")):
            st.error("Exit time must be after the earliest Entry time.")
        elif not items:
            st.error("Stocks not selected.")
        elif not picked_pats:
            st.error("Candle patterns not selected.")
        elif smart is None:
            st.error("Angel One login is required (historical data login required).")
        else:
            interval, tf_min = TIMEFRAMES[tf_label]
            data = {"exchange": exchange, "items": items, "capital": float(capital), "interval": interval, "tf_min": tf_min,
                    "max_open": int(max_open), "no_limit": no_limit, "position_amount": float(position_amount or 0),
                    "size_mode": size_mode, "position_qty": int(position_qty or 0), "target_pct": float(target_pct),
                    "sl_pct": float(sl_pct or 0), "max_hold": int(max_hold), "patterns": list(picked_pats),
                    "cloud_filter": cloud_pick, "cloud_exit": cloud_exit_pick, "cloud_exit_mode": cloud_exit_mode,
                    "brokerage_buy_pct": float(brokerage_buy_pct), "brokerage_sell_pct": float(brokerage_sell_pct),
                    "entry_times": entry_times, "exit_at": exit_at}
            cfg = _cfg_from_data(data)
            cfg.slippage_pct, cfg.brokerage_pct = 0.0, 0.0          # slippage stays off; brokerage = buy / sell leg % above
            bar = st.progress(0.0, text="Loading history...")

            def _prog(n, total, sym):
                bar.progress(n / total, text=f"Loading history {n}/{total}  ({sym})")

            res = run_backtest(smart, cfg, items, exchange, d_from, d_to, precise, _prog)
            bar.empty()
            res["capital"] = float(capital)
            res["index_map"] = index_map
            res["setup"] = setup                                     # the exact inputs of this run (saved with the result)
            res["label"] = (f"{exchange} | {tf_label} | {d_from:%d %b %Y} → {d_to:%d %b %Y} | "
                            f"{len(items)} stocks | target {target_pct}%"
                            + (f" | entry at {entry_times.replace(',', ', ')}" if entry_times else "")
                            + (f" | exit {exit_at}" if exit_at else "")
                            + (f" | cloud exit {cloud_exit_pick.replace(',', '/')} ({cloud_exit_mode})" if cloud_exit_pick else ""))
            ss.pop("bt_cmp_idx", None)                               # index comparison starts from this run's index
            ss["bt_result"] = res
            # AUTO SAVE: the finished run goes to Strategies (folder / file chosen at the top)
            if not _safe_name(file_name):
                st.info("Result not saved to Strategies - enter a File name at the top and run again to save it.")
            else:
                target = (_safe_name(new_folder) or DEFAULT_FOLDER) if creating else folder_pick
                try:
                    replaced = save_strategy(target, file_name, setup, res)
                    ss["sf_sel"] = target
                    ss["bt_restore"] = {**R, "_folder": target, "_file": _safe_name(file_name)}   # next run saves to the same file
                    st.success(f"{'Replaced' if replaced else 'Saved'} {_safe_name(file_name)} in Strategies / {target}.")
                except Exception as ex:
                    st.error(f"Backtest finished, but saving to Strategies failed: {ex}")

    res = ss.get("bt_result")
    if res:
        st.subheader("Result")
        st.caption(res["label"])
        _bt_show(res, res["capital"])


# ============================== STRATEGIES PAGE (folders on the side, files with Results) ==============================
def _results_cell(m):
    """(coloured headline, small line) of one saved file."""
    if not m.get("has_result"):
        return None, "No result saved"
    s = m.get("summary") or {}
    if not s.get("n"):
        return None, "0 trades"
    net, ret, pf = float(s["net"]), float(s["ret"]), float(s["pf"])
    pf_txt = "∞" if math.isinf(pf) else f"{pf:.1f}"
    head = f":{'green' if net >= 0 else 'red'}[**{net:+,.0f} ({ret:+.2f}%)**]"
    return head, f"{int(s['n'])} trades · win {float(s['win_rate']):.0f}% · PF {pf_txt}"


def _saved_on(m):
    try:
        return datetime.strptime(str(m.get("saved_at"))[:10], "%Y-%m-%d").strftime("%d %b")
    except ValueError:
        return "-"


def strategies_page():
    ss = st.session_state
    msg = ss.pop("sf_msg", None)
    if msg:
        getattr(st, msg[0])(msg[1])
    folders = list_folders()
    names = [n for n, _ in folders]
    if ss.get("sf_sel") not in names:
        ss["sf_sel"] = names[0] if names else None
    sel = ss["sf_sel"]

    left, right = st.columns([1, 2.7], gap="medium")
    with left.container(border=True):
        a, b = st.columns([1, 1], vertical_alignment="center")
        a.markdown("<div style='opacity:.85'>Folders</div>", unsafe_allow_html=True)
        with b.popover("➕ New", use_container_width=True):
            st.text_input("Folder name", key="sf_new_name")
            st.button("Create folder", key="sf_new_go", on_click=_cb_new_folder)
        for n, c in folders:
            if st.button(f"📁 {n}  ·  {c}", key=f"sf_pick_{n}", type="primary" if n == sel else "tertiary",
                         use_container_width=True) and n != sel:
                ss["sf_sel"] = n
                st.rerun()
        if not folders:
            st.caption("No folders yet.")

    with right.container(border=True):
        if not sel:
            st.info("Nothing saved yet. Pick a folder and file name on the Backtest tab and run a backtest, or create a folder with New.")
            return
        files = list_files(sel)
        h1, h2, h3 = st.columns([2.6, 1.5, 1.5], vertical_alignment="center")
        h1.markdown(f"<div style='font-size:1.25rem;font-weight:700'>{html.escape(sel)}</div>"
                    f"<div style='opacity:.65'>{len(files)} file{'s' if len(files) != 1 else ''}</div>", unsafe_allow_html=True)
        with h2.popover("✏️ Rename folder", use_container_width=True):
            st.text_input("New name", value=sel, key=f"sf_ren_{sel}")
            st.button("Rename", key=f"sf_ren_go_{sel}", on_click=_cb_rename_folder, args=(sel,))
        with h3.popover(":red[🗑 Delete folder]", use_container_width=True):
            st.warning(f"Delete **{sel}** and its {len(files)} file{'s' if len(files) != 1 else ''}? This cannot be undone.")
            st.button("Yes, delete folder", key=f"sf_del_go_{sel}", type="primary", on_click=_cb_delete_folder, args=(sel,))

        W = [2.6, 3.1, 1.1, 2.0]
        with st.container(border=True):
            hc = st.columns(W)
            for col, t in zip(hc, ("File name", "Results", "Saved", "Actions")):
                col.markdown(f"<div style='opacity:.8;{'text-align:right' if t == 'Actions' else ''}'>{t}</div>", unsafe_allow_html=True)
            if not files:
                st.caption("This folder is empty.")
            for m in files:
                st.divider()
                c1, c2, c3, c4 = st.columns(W, vertical_alignment="center")
                c1.markdown(f"📄 **{m['name']}**")
                head, sub = _results_cell(m)
                if head:
                    c2.markdown(head)
                c2.caption(sub)
                c3.markdown(_saved_on(m))
                with c4:
                    b1, b2, b3 = st.columns(3)
                    key = f"{sel}_{m['file']}"
                    b1.button("📂", key=f"sf_open_{key}", help="Open - setup and saved result on the Backtest tab",
                              on_click=_cb_open_strategy, args=(sel, m["file"], True), use_container_width=True)
                    b2.button("✏️", key=f"sf_edit_{key}", help="Edit - load the setup, change it, run again and save over it",
                              on_click=_cb_open_strategy, args=(sel, m["file"], False), use_container_width=True)
                    with b3.popover(":red[🗑]", help="Delete this file", use_container_width=True):
                        st.write(f"Delete **{m['name']}**?")
                        st.button("Yes, delete", key=f"sf_del_{key}", type="primary", on_click=_cb_delete_strategy, args=(sel, m["file"]))


# ============================== ENTRY ==============================
st.set_page_config(page_title="Backtest", layout="wide", initial_sidebar_state="collapsed")
_ss = st.session_state
if "smart" not in _ss and not _ss.get("no_autologin"):
    _creds = env_creds()                                      # ANGEL_* env / secrets -> auto login (saved logins use the login page)
    if _creds:
        with st.spinner("Angel One login..."):
            _sm, _err = angel_login(*_creds)
        if _sm is not None:
            _ss.smart, _ss.client_id, _ss.creds = _sm, _creds[1].strip().upper(), _creds
        else:
            st.warning(f"Auto login failed: {_err}")

if "smart" not in _ss:
    login_page()
else:
    # re-login every 8 hours (Angel session expiry)
    if _ss.get("creds") and time.time() - _ss.setdefault("login_time", time.time()) > 8 * 3600:
        _sm, _err = angel_login(*_ss.creds)
        if _sm is not None:
            _ss.smart, _ss.login_time = _sm, time.time()
    no_dim()
    _ui_css()
    _app_header()
    if _ss.get("bt_view") == "strategies":
        strategies_page()
    else:
        backtest_page()
