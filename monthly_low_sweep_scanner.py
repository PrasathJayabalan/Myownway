import math
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="Monthly Low Sweep Scanner", page_icon="📅", layout="wide")

# ============================================================
# MONTHLY LOW SWEEP RULES
# ============================================================
# 1. Mother candle = a completed MONTHLY candle. Its LOW is the liquidity level.
# 2. Later monthly candles fully inside the mother range = INSIDE candles (ignored).
# 3. Valid sweep: monthly low < mother low AND monthly close > mother low.
# 4. Confirmation (Weekly or Daily, MOST RECENT BOS within max age): close > mother low AND close > highest
#    high of the previous N candles (BOS).
# 5. Entry = BOS candle high + buffer.
# 6. SL    = monthly sweep low - buffer (or BOS candle low, tighter option).
# 7. Target = fixed % / R-multiple / structural (12-month high).
# 8. Each setup is tracked AFTER the BOS: PENDING -> ACTIVE -> TARGET/SL HIT.
#
# Rules-based educational scanner. Score is a checklist, NOT a probability.
# ============================================================

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.title {font-size: 36px; font-weight: 800; margin-bottom: 0;}
.sub {color:#6b7280; font-size:15px; margin-bottom:20px;}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="title">📅 Monthly Low Sweep Scanner</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub">Monthly sweep → reclaim → Weekly/Daily BOS → entry → SL → target '
    '• inside candles ignored • trade state tracked</div>', unsafe_allow_html=True)

OHLC = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
_pv = tuple(int(x) for x in pd.__version__.split(".")[:2])
MONTH_RULE = "ME" if _pv >= (2, 2) else "M"
TFS = ("Weekly", "Daily")


def stretch(fn, *args, **kwargs):
    """Works across Streamlit versions (width='stretch' vs use_container_width)."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


# ============================================================
# DATA
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def get_daily(symbol: str, period: str = "15y") -> pd.DataFrame:
    df = yf.download(symbol.strip(), period=period, interval="1d", auto_adjust=False,
                     progress=False, threads=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    need = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in need):
        return pd.DataFrame()
    df = df[need].dropna().copy()
    df.index = pd.to_datetime(df.index)
    return df[~df.index.duplicated(keep="last")]


def resample(daily, rule):
    return daily.resample(rule).agg(OHLC).dropna()


# ============================================================
# STRUCTURE
# ============================================================

def find_anchor(monthly, lookback):
    """Latest mother candle -> (inside candles) -> external sweep + reclaim.
    The previous month CAN be the mother (sweep may be the latest month)."""
    n = len(monthly)
    if n < 4:
        return None
    best = None
    for m in range(max(0, n - lookback - 2), n - 1):
        mh, ml = float(monthly.High.iloc[m]), float(monthly.Low.iloc[m])
        inside = []
        for s in range(m + 1, n):
            ch, cl, cc = (float(monthly[c].iloc[s]) for c in ("High", "Low", "Close"))
            if ch <= mh and cl >= ml:
                inside.append(s)
                continue
            if cl < ml and cc > ml:
                if s >= n - lookback and (best is None or s >= best["s"]):
                    best = dict(m=m, s=s, inside=inside.copy(), mh=mh, ml=ml,
                                sweep_low=cl, sweep_high=ch, sweep_close=cc)
                break
            if cl < ml or ch > mh:   # closed below (no reclaim) or range invalidated
                break
    if not best:
        return None
    s = best["s"]
    best["start"] = pd.Timestamp(monthly.index[s]).to_period("M").start_time
    best["failed"] = bool((monthly.Close.iloc[s + 1:] < best["sweep_low"]).any())
    return best


def find_bos(conf, start, level, n_back):
    """Most RECENT BOS since the sweep month (scans newest -> oldest)."""
    for i in range(len(conf) - 1, n_back - 1, -1):
        if conf.index[i] < start:
            break
        r = conf.iloc[i]
        prev_high = float(conf.High.iloc[i - n_back:i].max())
        if float(r.Close) > level and float(r.Close) > prev_high:
            return dict(i=i, date=conf.index[i], high=float(r.High), low=float(r.Low),
                        close=float(r.Close), volume=float(r.Volume), prior_high=prev_high,
                        avg_vol=float(conf.Volume.iloc[max(0, i - 20):i].mean()))
    return None


def trade_state(daily, after_date, entry, stop, target):
    state, trig = "PENDING ENTRY", None
    for dt, r in daily[daily.index > after_date].iterrows():
        if state == "PENDING ENTRY":
            if r.Low <= stop:
                return "INVALIDATED", None
            if r.High >= entry:
                state, trig = "ACTIVE", dt     # same-day exits not evaluated
        elif state == "ACTIVE":
            if r.Low <= stop:
                return "SL HIT", trig
            if r.High >= target:
                return "TARGET HIT", trig
    return state, trig


def plan_for_tf(daily, monthly, anchor, tf, p):
    conf = resample(daily, "W-FRI") if tf == "Weekly" else daily.copy()
    if tf == "Weekly" and len(conf) and conf.index[-1] > pd.Timestamp.today().normalize():
        conf = conf.iloc[:-1]                   # drop the unfinished week
    bos = find_bos(conf, anchor["start"], anchor["ml"], p["n_back"])
    if not bos:
        return dict(tf=tf, conf=conf, bos=None, status="WAIT (no BOS yet)")

    age = len(conf) - 1 - bos["i"]
    bos["age"] = age
    if age > p["max_age"]:
        return dict(tf=tf, conf=conf, bos=bos,
                    status=f"WAIT (BOS stale: {age} candles old)")

    buf = p["buf"] / 100
    entry = bos["high"] * (1 + buf)
    base = anchor["sweep_low"] if p["sl_mode"].startswith("Monthly") else bos["low"]
    stop = base * (1 - buf)
    risk = entry - stop
    note = ""
    if risk <= 0:
        return dict(tf=tf, conf=conf, bos=bos, status="INVALID RISK")

    if p["tgt_mode"] == "Fixed %":
        target = entry * (1 + p["tgt_pct"] / 100)
    elif p["tgt_mode"] == "R-multiple":
        target = entry + p["tgt_r"] * risk
    else:  # structural: highest high of last 12 monthly candles
        target = float(monthly.High.tail(12).max())
        if target < entry + risk:
            target = entry + 2 * risk
            note = "Structural high too close; used 2R"
    rr = (target - entry) / risk
    risk_pct = risk / entry * 100

    state, trig = trade_state(daily, bos["date"], entry, stop, target)
    cmp_ = float(daily.Close.iloc[-1])
    extended = state == "ACTIVE" and cmp_ > entry * (1 + p["chase"] / 100)

    vol_ok = bos["avg_vol"] > 0 and bos["volume"] > bos["avg_vol"]
    rng = anchor["sweep_high"] - anchor["sweep_low"]
    rej = (anchor["sweep_close"] - anchor["sweep_low"]) / rng if rng > 0 else 0
    score = (25 + (15 if rej >= 0.5 else 0) + 25 + (10 if vol_ok else 0)
             + (10 if risk_pct <= 10 else 5 if risk_pct <= 15 else 0)
             + (15 if rr >= 2 else 10 if rr >= 1.5 else 5 if rr >= 1.2 else 0))

    filters_ok = rr >= p["min_rr"] and risk_pct <= p["max_sl"]
    if state in ("PENDING ENTRY", "ACTIVE") and not extended and filters_ok:
        status = "QUALIFIED"
    elif extended:
        status = "EXTENDED"
    elif state in ("PENDING ENTRY", "ACTIVE"):
        status = "WAIT (R:R / SL filter)"
    else:
        status = state

    max_risk = p["capital"] * p["risk_pct"] / 100
    qty = max(0, min(math.floor(max_risk / risk), math.floor(p["capital"] / entry)))
    return dict(tf=tf, conf=conf, bos=bos, status=status, state=state, trig=trig, entry=entry,
                stop=stop, target=target, rr=rr, risk_pct=risk_pct, score=score, qty=qty,
                capital_used=qty * entry, max_loss=qty * risk, cmp=cmp_, note=note,
                vol_ok=vol_ok)


def analyze(symbol, p):
    r = dict(symbol=symbol, daily=pd.DataFrame(), monthly=pd.DataFrame(), anchor=None,
             plans={}, status="DATA ERROR", forming=False)
    daily = get_daily(symbol)
    if daily.empty or len(daily) < 300:
        return r
    r["daily"] = daily
    monthly = resample(daily, MONTH_RULE)
    forming = bool(monthly.index[-1] > pd.Timestamp.today().normalize())
    if forming and not p["include_forming"]:
        monthly = monthly.iloc[:-1]
    r["monthly"], r["forming"] = monthly, forming and p["include_forming"]

    anchor = find_anchor(monthly, p["lookback"])
    r["anchor"] = anchor
    if not anchor:
        r["status"] = "NO SETUP"
        return r
    if anchor["failed"]:
        r["status"] = "FAILED (closed below sweep low)"
        return r
    r["plans"] = {tf: plan_for_tf(daily, monthly, anchor, tf, p) for tf in TFS}
    r["status"] = r["plans"][p["tf"]]["status"]
    return r


# ============================================================
# CHARTS
# ============================================================

def candles(df, name):
    return go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close,
                          name=name, increasing_line_color="#16a34a",
                          decreasing_line_color="#dc2626")


def monthly_chart(r):
    a, mo = r["anchor"], r["monthly"]
    w = mo.iloc[max(0, a["m"] - 6): a["s"] + 5]
    fig = go.Figure(candles(w, "Monthly"))
    fig.add_hrect(y0=a["ml"], y1=a["mh"], fillcolor="rgba(59,130,246,0.08)", line_width=0,
                  annotation_text="MOTHER RANGE", annotation_position="top left")
    fig.add_hline(y=a["ml"], line_dash="dash", line_width=2, annotation_position="bottom left",
                  annotation_text=f"MOTHER LOW ₹{a['ml']:.2f}")
    fig.add_annotation(x=mo.index[a["m"]], y=a["mh"], text="MOTHER", showarrow=True, ay=-40)
    for i in a["inside"]:
        fig.add_annotation(x=mo.index[i], y=float(mo.High.iloc[i]), text="INSIDE",
                           showarrow=True, ay=-25, font=dict(size=10))
    tag = "SWEEP (forming)" if r["forming"] and a["s"] == len(mo) - 1 else "SWEEP"
    fig.add_annotation(x=mo.index[a["s"]], y=a["sweep_low"], text=f"{tag}<br>₹{a['sweep_low']:.2f}",
                       showarrow=True, ax=55, ay=45)
    fig.add_annotation(x=mo.index[a["s"]], y=a["sweep_close"], text="RECLAIM",
                       showarrow=True, ax=-60, ay=-45)
    fig.update_layout(title=f"{r['symbol']} — Monthly Liquidity Structure", height=520,
                      xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def confirm_chart(r, plan):
    a, conf = r["anchor"], plan["conf"]
    back = 120 if plan["tf"] == "Weekly" else 45
    d = conf[conf.index >= a["start"] - pd.Timedelta(days=back)]
    fig = go.Figure(candles(d, plan["tf"]))
    fig.add_hline(y=a["ml"], line_dash="dash", annotation_text=f"MOTHER LOW ₹{a['ml']:.2f}",
                  annotation_position="top left")
    fig.add_hline(y=a["sweep_low"], line_dash="dot", annotation_text=f"SWEEP LOW ₹{a['sweep_low']:.2f}",
                  annotation_position="bottom left")
    if plan.get("entry"):
        fig.add_annotation(x=plan["bos"]["date"], y=plan["bos"]["close"], text="BOS",
                           showarrow=True, ax=40, ay=-50)
        for y, lbl, dash in ((plan["entry"], "ENTRY", "solid"), (plan["stop"], "SL", "dot"),
                             (plan["target"], "TARGET", "dash")):
            fig.add_hline(y=y, line_dash=dash, line_width=2, annotation_text=f"{lbl} ₹{y:.2f}",
                          annotation_position="top right")
    fig.update_layout(title=f"{r['symbol']} — {plan['tf']} confirmation & trade plan", height=520,
                      xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=60, b=20))
    return fig


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Monthly Scanner")
    symbols_text = st.text_area(
        "Tickers (full Yahoo symbol)", height=200,
        value="HUDCO.NS\nBANKINDIA.NS\nTATASTEEL.NS\nRELIANCE.NS\nINFY.NS\nSBIN.NS\nHDFCBANK.NS\nITC.NS\nWIPRO.NS\nLT.NS",
        help="Enter the full Yahoo Finance ticker, e.g. RELIANCE.NS, TCS.BO, AAPL. One per line or comma-separated.")
    tf = st.selectbox("Primary confirmation timeframe", TFS, index=0,
                      help="Weekly = fewer, cleaner signals. Daily = earlier but noisier.")
    n_back = st.number_input("BOS lookback (candles)", 2, 10, 3)
    max_age = st.number_input("Max BOS age (candles)", 0, 30, 5,
                              help="BOS must be within this many confirmation candles "
                                   "(weeks for Weekly, days for Daily) of the latest candle.")
    lookback = st.number_input("Monthly lookback (months)", 3, 24, 6)
    include_forming = st.checkbox("Include current (forming) month", value=True)
    st.subheader("Trade plan")
    sl_mode = st.selectbox("Stop-loss basis", ["Monthly sweep low (structural)",
                                               "Confirmation candle low (tighter)"])
    tgt_mode = st.selectbox("Target method", ["Fixed %", "R-multiple",
                                              "Structural (12-month high)"])
    tgt_pct = st.number_input("Fixed target (%)", 2.0, 100.0, 15.0, 1.0)
    tgt_r = st.number_input("R-multiple", 1.0, 10.0, 2.0, 0.5)
    buf = st.number_input("Entry / SL buffer (%)", 0.0, 2.0, 0.10, 0.05)
    st.subheader("Filters & sizing")
    min_rr = st.number_input("Min R:R", 0.5, 10.0, 1.5, 0.1)
    max_sl = st.number_input("Max SL distance (%)", 1.0, 50.0, 15.0, 1.0)
    chase = st.number_input("Max chase above entry (%)", 0.0, 20.0, 3.0, 0.5)
    capital = st.number_input("Capital (₹)", 1000.0, value=100000.0, step=5000.0)
    risk_pct = st.number_input("Max risk / trade (%)", 0.1, 10.0, 1.0, 0.1)
    scan = st.button("🔎 SCAN & DRAW CHARTS", type="primary", use_container_width=True)

params = dict(tf=tf, n_back=int(n_back), max_age=int(max_age), lookback=int(lookback), include_forming=include_forming,
              sl_mode=sl_mode, tgt_mode=tgt_mode, tgt_pct=tgt_pct, tgt_r=tgt_r, buf=buf,
              min_rr=min_rr, max_sl=max_sl, chase=chase, capital=capital, risk_pct=risk_pct)

if scan:
    symbols = []
    for x in symbols_text.replace(",", "\n").splitlines():
        x = x.strip().upper()
        if x and x not in symbols:
            symbols.append(x)
    if not symbols:
        st.error("Enter at least one symbol.")
        st.stop()
    symbols = symbols[:30]
    results, bar = [], st.progress(0)
    for i, sym in enumerate(symbols):
        try:
            results.append(analyze(sym, params))
        except Exception as exc:
            results.append(dict(symbol=sym, daily=pd.DataFrame(), monthly=pd.DataFrame(),
                                anchor=None, plans={}, forming=False,
                                status=f"ERROR: {str(exc)[:60]}"))
        bar.progress((i + 1) / len(symbols))
    bar.empty()
    st.session_state["m_results"] = results
    st.session_state["m_tf"] = tf

# ============================================================
# RESULTS
# ============================================================

if "m_results" not in st.session_state:
    st.info("Enter symbols on the left and click **SCAN & DRAW CHARTS**.")
    st.markdown("""
```text
MOTHER MONTH LOW → (inside months ignored) → SWEEP BELOW → CLOSE BACK ABOVE
   → WEEKLY / DAILY BOS → ENTRY → SL (sweep low) → TARGET → STATE TRACKING
```
**Which timeframe?** Monthly gives the level and the stop. Weekly confirmation is the
balanced entry trigger. Daily is earlier but noisier. The scan shows both side by side.
""")
    st.stop()

results, ptf = st.session_state["m_results"], st.session_state["m_tf"]
sweeps = [r for r in results if r["anchor"]]
c1, c2, c3 = st.columns(3)
c1.metric("Stocks", len(results))
c2.metric("Monthly sweeps", len(sweeps))
c3.metric("Qualified", sum(r["status"] == "QUALIFIED" for r in results))

rows, cmp_rows = [], []
for r in results:
    pl = r["plans"].get(ptf)
    a = r["anchor"]
    rows.append({
        "Stock": r["symbol"], "Status": r["status"],
        "Mother Low": a["ml"] if a else None, "Sweep Low": a["sweep_low"] if a else None,
        "CMP": pl.get("cmp") if pl else None, "Entry": pl.get("entry") if pl else None,
        "SL": pl.get("stop") if pl else None, "Target": pl.get("target") if pl else None,
        "SL %": pl.get("risk_pct") if pl else None, "R:R": pl.get("rr") if pl else None,
        "Qty": pl.get("qty") if pl else None, "Score": pl.get("score") if pl else None})
    for t, x in r["plans"].items():
        cmp_rows.append({
            "Stock": r["symbol"], "TF": t, "Status": x["status"],
            "BOS date": x["bos"]["date"].strftime("%Y-%m-%d") if x.get("bos") else None,
            "BOS age": x["bos"]["age"] if x.get("bos") and "age" in x["bos"] else None,
            "Entry": x.get("entry"), "SL": x.get("stop"), "Target": x.get("target"),
            "SL %": x.get("risk_pct"), "R:R": x.get("rr"), "Score": x.get("score")})

summary = pd.DataFrame(rows)
st.markdown(f"### 🔎 Scanner results ({ptf} confirmation)")
stretch(st.dataframe, summary.round(2), hide_index=True)
st.download_button("⬇️ Download CSV", summary.to_csv(index=False).encode(),
                   "monthly_low_sweep_results.csv", "text/csv")

if cmp_rows:
    st.markdown("### ⚖️ Weekly vs Daily confirmation")
    stretch(st.dataframe, pd.DataFrame(cmp_rows).round(2), hide_index=True)
    st.caption("Compare entry, SL distance and R:R per timeframe. Daily usually triggers "
               "earlier; Weekly usually filters more false breakouts.")

st.markdown("---")
st.markdown("## 📈 Setup charts")
for r in results:
    a = r["anchor"]
    with st.expander(f"{r['symbol']} — {r['status']}", expanded=(r["status"] == "QUALIFIED")):
        if not a:
            st.warning("No valid external monthly low sweep/reclaim found (or data error).")
            continue
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mother low", f"₹{a['ml']:.2f}")
        m2.metric("Sweep low", f"₹{a['sweep_low']:.2f}")
        m3.metric("Inside months", len(a["inside"]))
        m4.metric("Sweep month", "Forming" if r["forming"] and a["s"] == len(r["monthly"]) - 1
                  else "Closed")
        stretch(st.plotly_chart, monthly_chart(r), key=f"mo_{r['symbol']}")
        if a["failed"]:
            st.error("A later monthly close is below the sweep low. Setup failed.")
            continue
        tabs = st.tabs([f"{t} confirmation" for t in TFS])
        for tab, t in zip(tabs, TFS):
            plan = r["plans"][t]
            with tab:
                if not plan.get("bos"):
                    st.warning(f"No {t} BOS yet. No entry generated.")
                    continue
                if "entry" not in plan:
                    st.warning(plan["status"])
                    continue
                k = st.columns(6)
                k[0].metric("BOS", plan["bos"]["date"].strftime("%Y-%m-%d"),
                            f"{plan['bos']['age']} candles ago", delta_color="off")
                k[1].metric("Entry", f"₹{plan['entry']:.2f}")
                k[2].metric("SL", f"₹{plan['stop']:.2f}", f"-{plan['risk_pct']:.1f}%",
                            delta_color="off")
                k[3].metric("Target", f"₹{plan['target']:.2f}")
                k[4].metric("R:R", f"{plan['rr']:.2f}")
                k[5].metric("Score", f"{plan['score']}/100")
                st.caption(f"Trade state: **{plan['state']}** • Status: **{plan['status']}**"
                           + (f" • {plan['note']}" if plan["note"] else ""))
                stretch(st.plotly_chart, confirm_chart(r, plan), key=f"cf_{r['symbol']}_{t}")
                q = st.columns(3)
                q[0].metric("Quantity", plan["qty"])
                q[1].metric("Capital used", f"₹{plan['capital_used']:.0f}")
                q[2].metric("Max planned loss", f"₹{plan['max_loss']:.0f}")