import math
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(
    page_title="Weekly Low Sweep Scanner V2",
    page_icon="📊",
    layout="wide",
)

# ============================================================
# V2 RULES
# ============================================================
# 1. Primary liquidity = previous completed weekly LOW.
# 2. The previous completed weekly candle is the "mother/anchor candle".
# 3. Later weekly candles that remain inside the mother candle range are
#    labelled INSIDE CANDLES and are NOT treated as independent sweep levels.
# 4. A valid weekly sweep requires:
#       sweep low < mother/previous weekly low
#       AND sweep close > mother/previous weekly low
# 5. Daily confirmation:
#       close > weekly low
#       AND close > highest high of previous 3 daily candles
# 6. Entry = confirmation candle high + buffer.
# 7. SL = weekly sweep low - buffer.
# 8. Target = entry * (1 + target %).
#
# This is a rules-based educational scanner. Score != probability of success.
# ============================================================

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.title {font-size: 36px; font-weight: 800; margin-bottom: 0;}
.sub {color:#6b7280; font-size:15px; margin-bottom:20px;}
.section {font-size:22px; font-weight:750; margin-top:15px;}
.small {color:#6b7280; font-size:13px;}
.badge {
    display:inline-block; padding:5px 10px; border-radius:999px;
    font-weight:700; font-size:12px; margin-right:5px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">📊 Weekly Low Sweep Scanner V2</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub">External liquidity only • inside candles ignored • '
    'weekly sweep → reclaim → daily BOS → entry → SL → +5% target</div>',
    unsafe_allow_html=True
)

# ============================================================
# DATA
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def get_daily(symbol: str, period: str = "2y") -> pd.DataFrame:
    ticker = symbol.upper().replace(".NS", "") + ".NS"
    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in needed):
        return pd.DataFrame()

    df = df[needed].dropna().copy()
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")]
    return df


def make_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    return daily.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()


def find_latest_external_sweep(weekly: pd.DataFrame, lookback: int = 12):
    """
    External weekly sweep:
    - mother candle = immediately previous completed weekly candle
    - later candle breaks mother low
    - later candle closes back above mother low

    Inside candles are explicitly ignored as independent liquidity levels.
    """
    if len(weekly) < lookback + 2:
        return None

    start = max(1, len(weekly) - lookback - 1)

    for i in range(len(weekly) - 1, start - 1, -1):
        mother = weekly.iloc[i - 1]
        sweep = weekly.iloc[i]

        mother_low = float(mother["Low"])
        mother_high = float(mother["High"])
        sweep_low = float(sweep["Low"])
        sweep_close = float(sweep["Close"])

        if sweep_low < mother_low and sweep_close > mother_low:
            # Find consecutive inside weekly candles between mother and sweep.
            inside = []
            for j in range(i + 1, i):
                pass

            return {
                "mother_idx": i - 1,
                "sweep_idx": i,
                "mother_date": weekly.index[i - 1],
                "sweep_date": weekly.index[i],
                "mother_low": mother_low,
                "mother_high": mother_high,
                "sweep_low": sweep_low,
                "sweep_close": sweep_close,
            }

    return None


def find_latest_anchor_and_sweep(weekly: pd.DataFrame, lookback: int = 12):
    """
    More useful V2 model:
    Search a mother candle and then later candles.
    A candle is considered an INSIDE candle when:
        high <= mother high AND low >= mother low
    We do not allow an inside candle to become a new liquidity anchor.

    A sweep is the first later candle that:
        low < mother low AND close > mother low
    """
    if len(weekly) < 5:
        return None

    start = max(0, len(weekly) - lookback - 3)

    candidates = []

    for m in range(start, len(weekly) - 2):
        mother = weekly.iloc[m]
        mh = float(mother["High"])
        ml = float(mother["Low"])

        inside_indices = []
        broke_without_reclaim = False

        for s in range(m + 1, len(weekly)):
            c = weekly.iloc[s]
            ch = float(c["High"])
            cl = float(c["Low"])
            cc = float(c["Close"])

            is_inside = ch <= mh and cl >= ml

            if is_inside:
                inside_indices.append(s)
                continue

            # A valid external sweep/reclaim.
            if cl < ml and cc > ml:
                candidates.append({
                    "mother_idx": m,
                    "sweep_idx": s,
                    "inside_indices": inside_indices.copy(),
                    "mother_date": weekly.index[m],
                    "sweep_date": weekly.index[s],
                    "mother_high": mh,
                    "mother_low": ml,
                    "sweep_low": cl,
                    "sweep_close": cc,
                })
                break

            # If price broke below the mother low and closed below it,
            # this is not a reclaim sweep. Do not manufacture a signal.
            if cl < ml and cc <= ml:
                broke_without_reclaim = True
                break

            # If price breaks above mother high, the old range has been
            # invalidated for this candidate.
            if ch > mh:
                break

    if not candidates:
        return None

    return candidates[-1]


def daily_bos(daily: pd.DataFrame, sweep_date, weekly_low: float):
    """
    Daily bullish confirmation:
    close > weekly low AND close > highest high of previous 3 daily candles.
    """
    x = daily[daily.index >= pd.Timestamp(sweep_date)].copy()
    x = x.tail(30)

    if len(x) < 4:
        return None

    for i in range(3, len(x)):
        row = x.iloc[i]
        prev_high = float(x.iloc[i - 3:i]["High"].max())

        if float(row["Close"]) > weekly_low and float(row["Close"]) > prev_high:
            avg20 = float(x["Volume"].tail(20).mean())
            return {
                "date": x.index[i],
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"]),
                "avg20": avg20,
                "prior_high": prev_high,
            }

    return None


def setup_score(entry, weekly_low, rr, volume_ok):
    score = 15 + 15 + 10 + 15 + 15  # sweep, reclaim, rejection, daily reclaim, BOS

    distance = abs(entry - weekly_low) / weekly_low * 100
    score += 10 if distance <= 3 else 5 if distance <= 6 else 0
    score += 5 if volume_ok else 0
    score += 5 if rr >= 1.2 else 0
    score += 5
    return min(score, 100)


def analyze(symbol, capital, risk_pct, target_pct, lookback, buffer_pct):
    daily = get_daily(symbol)

    if daily.empty or len(daily) < 100:
        return {
            "symbol": symbol,
            "daily": daily,
            "weekly": pd.DataFrame(),
            "setup": None,
            "status": "DATA ERROR",
        }

    weekly = make_weekly(daily)
    anchor = find_latest_anchor_and_sweep(weekly, lookback)

    if not anchor:
        return {
            "symbol": symbol,
            "daily": daily,
            "weekly": weekly,
            "setup": None,
            "status": "REMOVE",
        }

    weekly_low = anchor["mother_low"]
    sweep_low = anchor["sweep_low"]

    bos = daily_bos(daily, anchor["sweep_date"], weekly_low)

    if not bos:
        return {
            "symbol": symbol,
            "daily": daily,
            "weekly": weekly,
            "setup": {
                "anchor": anchor,
                "bos": None,
            },
            "status": "WAIT",
        }

    entry = bos["high"] * (1 + buffer_pct / 100)
    stop = sweep_low * (1 - buffer_pct / 100)
    target = entry * (1 + target_pct / 100)

    risk_share = entry - stop
    reward_share = target - entry
    rr = reward_share / risk_share if risk_share > 0 else 0

    volume_ok = bos["volume"] > bos["avg20"] if bos["avg20"] > 0 else False
    score = setup_score(entry, weekly_low, rr, volume_ok)

    max_risk_money = capital * risk_pct / 100
    qty_by_risk = math.floor(max_risk_money / risk_share) if risk_share > 0 else 0
    qty_by_capital = math.floor(capital / entry)
    qty = max(0, min(qty_by_risk, qty_by_capital))

    setup = {
        "anchor": anchor,
        "bos": bos,
        "entry": entry,
        "stop": stop,
        "target": target,
        "risk_share": risk_share,
        "reward_share": reward_share,
        "rr": rr,
        "score": score,
        "qty": qty,
        "capital_used": qty * entry,
        "max_loss": qty * risk_share,
        "volume_ok": volume_ok,
        "cmp": float(daily.iloc[-1]["Close"]),
    }

    return {
        "symbol": symbol,
        "daily": daily,
        "weekly": weekly,
        "setup": setup,
        "status": "QUALIFIED" if rr >= 1.2 else "WAIT",
    }


# ============================================================
# CHARTS
# ============================================================

def weekly_chart(result, show_inside=True):
    weekly = result["weekly"].copy()
    setup = result["setup"]

    if weekly.empty or not setup:
        return None

    a = setup["anchor"]
    m = a["mother_idx"]
    s = a["sweep_idx"]

    left = max(0, m - 8)
    right = min(len(weekly), s + 5)
    w = weekly.iloc[left:right].copy()

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=w.index,
        open=w["Open"],
        high=w["High"],
        low=w["Low"],
        close=w["Close"],
        name="Weekly candles",
        increasing_line_color="#16a34a",
        decreasing_line_color="#dc2626",
    ))

    # Mother candle range
    mother_date = weekly.index[m]
    fig.add_hrect(
        y0=a["mother_low"],
        y1=a["mother_high"],
        fillcolor="rgba(59,130,246,0.08)",
        line_width=0,
        annotation_text="MOTHER RANGE",
        annotation_position="top left",
    )

    # Previous / mother low
    fig.add_hline(
        y=a["mother_low"],
        line_dash="dash",
        line_width=2,
        annotation_text=f"PREVIOUS / MOTHER LOW ₹{a['mother_low']:.2f}",
        annotation_position="bottom left",
    )

    # Mother candle marker
    fig.add_annotation(
        x=mother_date,
        y=a["mother_high"],
        text="MOTHER CANDLE",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-45,
    )

    # Inside candles
    if show_inside:
        for idx in a["inside_indices"]:
            if idx < len(weekly):
                dt = weekly.index[idx]
                row = weekly.iloc[idx]
                fig.add_annotation(
                    x=dt,
                    y=float(row["High"]),
                    text="INSIDE",
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-28,
                    font=dict(size=10),
                )

    # Sweep
    sweep_date = weekly.index[s]
    fig.add_annotation(
        x=sweep_date,
        y=a["sweep_low"],
        text=f"SWEEP<br>₹{a['sweep_low']:.2f}",
        showarrow=True,
        arrowhead=2,
        ax=55,
        ay=45,
        font=dict(size=11),
    )

    # Reclaim
    fig.add_annotation(
        x=sweep_date,
        y=a["sweep_close"],
        text="RECLAIM<br>CLOSE > LOW",
        showarrow=True,
        arrowhead=2,
        ax=-65,
        ay=-50,
        font=dict(size=10),
    )

    fig.update_layout(
        title=f"{result['symbol']} — Weekly Liquidity Structure",
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=65, b=20),
        legend=dict(orientation="h"),
    )
    return fig


def daily_chart(result):
    daily = result["daily"].copy()
    setup = result["setup"]

    if daily.empty or not setup:
        return None

    a = setup["anchor"]
    bos = setup.get("bos")

    # Show a useful window around sweep and confirmation.
    start_date = pd.Timestamp(a["sweep_date"]) - pd.Timedelta(days=20)
    if bos:
        end_date = pd.Timestamp(bos["date"]) + pd.Timedelta(days=10)
    else:
        end_date = daily.index[-1]

    d = daily[(daily.index >= start_date) & (daily.index <= end_date)].copy()
    if d.empty:
        d = daily.tail(60)

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=d.index,
        open=d["Open"],
        high=d["High"],
        low=d["Low"],
        close=d["Close"],
        name="Daily candles",
        increasing_line_color="#16a34a",
        decreasing_line_color="#dc2626",
    ))

    # Weekly liquidity line
    fig.add_hline(
        y=a["mother_low"],
        line_dash="dash",
        line_width=2,
        annotation_text=f"PREVIOUS WEEKLY LOW ₹{a['mother_low']:.2f}",
        annotation_position="top left",
    )

    # Sweep low
    fig.add_hline(
        y=a["sweep_low"],
        line_dash="dot",
        line_width=1.5,
        annotation_text=f"SWEEP LOW ₹{a['sweep_low']:.2f}",
        annotation_position="bottom left",
    )

    # BOS
    if bos:
        fig.add_annotation(
            x=bos["date"],
            y=bos["close"],
            text="BOS / CHOCH<br>CONFIRMATION",
            showarrow=True,
            arrowhead=2,
            ax=55,
            ay=-55,
            font=dict(size=11),
        )

        # Entry, SL and target
        fig.add_hline(
            y=setup["entry"],
            line_width=2,
            annotation_text=f"ENTRY ₹{setup['entry']:.2f}",
            annotation_position="top right",
        )

        fig.add_hline(
            y=setup["stop"],
            line_dash="dot",
            line_width=2,
            annotation_text=f"SL ₹{setup['stop']:.2f}",
            annotation_position="bottom right",
        )

        fig.add_hline(
            y=setup["target"],
            line_dash="dash",
            line_width=2,
            annotation_text=f"+TARGET ₹{setup['target']:.2f}",
            annotation_position="top right",
        )

        # Entry zone arrow
        fig.add_annotation(
            x=bos["date"],
            y=setup["entry"],
            text="ENTRY TRIGGER",
            showarrow=True,
            arrowhead=2,
            ax=-60,
            ay=35,
            font=dict(size=10),
        )

    fig.update_layout(
        title=f"{result['symbol']} — Daily Confirmation & Trade Plan",
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=65, b=20),
    )
    return fig




# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ V2 Scanner")

    symbols_text = st.text_area(
        "NSE Symbols",
        value="HUDCO\nBANKINDIA\nTATASTEEL\nRELIANCE\nINFY\nSBIN\nHDFCBANK\nITC\nWIPRO\nLT",
        height=230,
        help="Enter up to 100 NSE symbols, one per line or comma-separated. Example: HUDCO,SBIN,INFY",
    )

    capital = st.number_input(
        "Capital (₹)",
        min_value=1000.0,
        value=10000.0,
        step=1000.0,
    )

    risk_pct = st.number_input(
        "Max risk / trade (%)",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
    )

    target_pct = st.number_input(
        "Target (%)",
        min_value=1.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
    )

    lookback = st.number_input(
        "Weekly lookback",
        min_value=3,
        max_value=30,
        value=12,
        step=1,
    )

    buffer_pct = st.number_input(
        "Entry / SL buffer (%)",
        min_value=0.0,
        max_value=1.0,
        value=0.10,
        step=0.05,
    )

    show_inside = st.checkbox("Show INSIDE candles", value=True)

    scan = st.button(
        "🔎 SCAN & DRAW CHARTS",
        type="primary",
        use_container_width=True,
    )

# ============================================================
# SCAN
# ============================================================

if scan:
    symbols = []
    for x in symbols_text.replace(",", "\n").splitlines():
        x = x.strip().upper()
        if x and x not in symbols:
            symbols.append(x)

    if not symbols:
        st.error("Enter at least one symbol.")
        st.stop()

    if len(symbols) > 5000:
        symbols = symbols[:5000]
        st.warning("Maximum 5,000 symbols per scan. Only the first 5,000 unique symbols will be scanned.")

    if len(symbols) > 500:
        st.warning("Large scan: up to 5,000 symbols are supported, but Yahoo Finance may throttle requests. A full scan can take a long time; scan smaller batches if data errors or rate limits occur.")

    results = []
    progress = st.progress(0)
    msg = st.empty()

    for i, symbol in enumerate(symbols):
        msg.write(f"Scanning **{symbol}** — stock {i + 1} of {len(symbols)}...")
        try:
            results.append(
                analyze(
                    symbol,
                    capital,
                    risk_pct,
                    target_pct,
                    lookback,
                    buffer_pct,
                )
            )
        except Exception as exc:
            results.append({
                "symbol": symbol,
                "daily": pd.DataFrame(),
                "weekly": pd.DataFrame(),
                "setup": None,
                "status": f"ERROR: {str(exc)[:80]}",
            })
        progress.progress((i + 1) / len(symbols))

    progress.empty()
    msg.empty()

    st.session_state["v2_results"] = results
    st.session_state["v2_target"] = target_pct


# ============================================================
# RESULTS
# ============================================================

if "v2_results" not in st.session_state:
    st.info(
        "Enter your stock symbols on the left and click "
        "**SCAN & DRAW CHARTS**."
    )

    st.markdown("""
### V2 logic

```text
PREVIOUS / MOTHER LOW
        ↓
INSIDE CANDLES
        ↓
IGNORE INSIDE LOWS
        ↓
EXTERNAL LOW SWEEP
        ↓
CLOSE BACK ABOVE LOW
        ↓
RECLAIM
        ↓
DAILY BOS / CHOCH
        ↓
ENTRY
        ↓
STRUCTURAL SL
        ↓
+5% TARGET
```
""")
else:
    results = st.session_state["v2_results"]

    qualified = [r for r in results if r["status"] == "QUALIFIED"]
    sweeps = [
        r for r in results
        if r["setup"] is not None
    ]
    confirmed = [
        r for r in results
        if r["setup"] is not None and r["setup"].get("bos") is not None
    ]

    st.markdown("### 📊 Scan Summary")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Stocks", len(results))
    c2.metric("External Sweeps", len(sweeps))
    c3.metric("Daily BOS", len(confirmed))
    c4.metric("Qualified", len(qualified))

    # Summary table
    rows = []

    for r in results:
        s = r["setup"]
        if not s:
            rows.append({
                "Stock": r["symbol"],
                "Status": r["status"],
                "Previous Low": None,
                "Sweep Low": None,
                "BOS": "❌",
                "Entry": None,
                "SL": None,
                "Target": None,
                "R:R": None,
                "Qty": 0,
                "Score": 0,
            })
            continue

        a = s["anchor"]
        rows.append({
            "Stock": r["symbol"],
            "Status": r["status"],
            "Previous Low": a["mother_low"],
            "Sweep Low": a["sweep_low"],
            "BOS": "✅" if s.get("bos") else "⏳",
            "Entry": s.get("entry"),
            "SL": s.get("stop"),
            "Target": s.get("target"),
            "R:R": s.get("rr"),
            "Qty": s.get("qty", 0),
            "Score": s.get("score", 0),
        })

    summary = pd.DataFrame(rows)

    for col in [
        "Previous Low", "Sweep Low", "Entry",
        "SL", "Target", "R:R"
    ]:
        summary[col] = summary[col].apply(
            lambda x: None if pd.isna(x) else round(float(x), 2)
        )

    st.markdown("### 🔎 Scanner Results")
    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
    )

    # Separate qualified-only table for quick trade-plan review.
    st.markdown("### ✅ Qualified Stocks Only")
    if qualified:
        qualified_rows = []
        copy_lines = []
        for r in qualified:
            setup = r["setup"]
            entry = float(setup["entry"])
            stop = float(setup["stop"])
            target = float(setup["target"])
            qualified_rows.append({
                "Ticker": r["symbol"],
                "Entry": round(entry, 2),
                "SL": round(stop, 2),
                "Target": round(target, 2),
                "R:R": round(float(setup["rr"]), 2),
                "Score": int(setup.get("score", 0)),
                "BOS Date": setup["bos"]["date"].strftime("%Y-%m-%d"),
            })
            # Exact requested format: TICKER:Entry|SL|Target
            copy_lines.append(
                f"{r['symbol']}:{entry:.2f}|{stop:.2f}|{target:.2f}"
            )

        qualified_df = pd.DataFrame(qualified_rows)
        st.dataframe(qualified_df, use_container_width=True, hide_index=True)
        st.caption("Copy-ready format: TICKER:Entry|SL|Target (prices rounded to 2 decimals).")
        st.code("\n".join(copy_lines), language=None)
        st.download_button(
            "⬇️ Download Qualified Stocks CSV",
            qualified_df.to_csv(index=False).encode("utf-8"),
            "weekly_low_sweep_qualified_stocks.csv",
            "text/csv",
        )
    else:
        st.info("No qualified stocks in this scan. The table and copy list will appear when one or more stocks qualify.")

    csv = summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Results CSV",
        csv,
        "weekly_low_sweep_v2_results.csv",
        "text/csv",
    )

    st.markdown("---")
    st.markdown("## 📈 Visual Setup Charts")

    st.caption(
        "Each stock gets a weekly structure chart and a daily confirmation "
        "chart. Inside candles are shown but are NOT used as separate "
        "liquidity levels."
    )

    # Every selected stock gets its own expandable chart section.
    for r in results:
        symbol = r["symbol"]
        status = r["status"]
        s = r["setup"]

        label = f"{symbol} — {status}"

        with st.expander(label, expanded=(status == "QUALIFIED")):
            if status in ["REMOVE", "DATA ERROR"] or not s:
                st.warning(
                    "No valid external weekly low sweep/reclaim was found "
                    "using the V2 rules."
                )
                continue

            a = s["anchor"]

            b1, b2, b3, b4, b5 = st.columns(5)
            b1.metric("Previous/Mother Low", f"₹{a['mother_low']:.2f}")
            b2.metric("Sweep Low", f"₹{a['sweep_low']:.2f}")
            b3.metric("Inside Candles", len(a["inside_indices"]))
            b4.metric("Status", status)
            b5.metric("Score", f"{s.get('score', 0)}/100")

            st.plotly_chart(
                weekly_chart(r, show_inside=show_inside),
                use_container_width=True,
                key=f"weekly_{symbol}",
            )

            st.markdown("#### Daily confirmation")

            if not s.get("bos"):
                st.warning(
                    "Weekly sweep + reclaim detected, but daily BOS/CHOCH "
                    "confirmation has not occurred yet. No entry is generated."
                )
            else:
                bos = s["bos"]

                d1, d2, d3, d4, d5 = st.columns(5)
                d1.metric("BOS Date", bos["date"].strftime("%Y-%m-%d"))
                d2.metric("Entry", f"₹{s['entry']:.2f}")
                d3.metric("SL", f"₹{s['stop']:.2f}")
                d4.metric("Target", f"₹{s['target']:.2f}")
                d5.metric("R:R", f"{s['rr']:.2f}")

                st.plotly_chart(
                    daily_chart(r),
                    use_container_width=True,
                    key=f"daily_{symbol}",
                )

                st.markdown("#### 💰 Position sizing")

                p1, p2, p3 = st.columns(3)
                p1.metric("Quantity", int(s["qty"]))
                p2.metric("Capital Used", f"₹{s['capital_used']:.2f}")
                p3.metric("Maximum Planned Risk", f"₹{s['max_loss']:.2f}")

                if status == "QUALIFIED":
                    st.success(
                        f"{symbol}: all mechanical V2 filters passed. "
                        f"This is a rules-based setup, not a guaranteed outcome."
                    )
                else:
                    st.warning(
                        f"{symbol}: confirmation exists, but the R:R "
                        f"filter did not qualify the setup."
                    )

    st.markdown("---")
    st.markdown("""
### 🧠 How V2 differs from V1

**V1 problem**

```text
Candle 1 low → liquidity
Candle 2 low → liquidity
Inside candle low → liquidity
Inside candle low → liquidity
```

**V2**

```text
Mother / anchor candle
        │
        ├── Inside candle → IGNORE
        ├── Inside candle → IGNORE
        ├── Inside candle → IGNORE
        │
        ↓
Break below mother LOW
        ↓
Close back above mother LOW
        ↓
REAL EXTERNAL SWEEP
```

The scanner therefore does **not** create a new liquidity level from every
inside candle.
""")
