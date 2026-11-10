import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="StockSense", page_icon="📈", layout="wide")

TICKERS = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
}
EPS = 1e-6
COLS = ["Vol5", "Vol21", "Vol63"]


# ---------------------------------------------------------------- data
@st.cache_data(ttl=3600, show_spinner="Downloading price data...")
def load_data(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start="2015-01-01", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.reset_index().ffill().drop_duplicates()
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Return"] = d["Close"].pct_change()
    d["Vol5"] = d["Return"].rolling(5).std()
    d["Vol21"] = d["Return"].rolling(21).std()
    d["Vol63"] = d["Return"].rolling(63).std()
    d["MA21"] = d["Close"].rolling(21).mean()
    d["MA50"] = d["Close"].rolling(50).mean()

    delta = d["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["RSI14"] = 100 - 100 / (1 + gain / loss)

    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"] = d["MACD"] - d["MACD_signal"]
    return d


# ---------------------------------------------------------------- model
@st.cache_data(show_spinner="Fitting volatility model...")
def volatility_model(df_ind: pd.DataFrame):
    """HAR-style model: log(next 5-day avg |return|) ~ log(Vol5, Vol21, Vol63)."""
    d = df_ind.copy()
    d["VolTarget"] = d["Return"].abs().rolling(5).mean().shift(-5)
    train = d.dropna(subset=COLS + ["VolTarget"]).reset_index(drop=True)
    s = int(len(train) * 0.8)

    X, y = np.log(train[COLS] + EPS), np.log(train["VolTarget"] + EPS)
    har = LinearRegression().fit(X[:s], y[:s])

    y_test = train["VolTarget"][s:].values
    preds = {
        "Training average": np.full(len(y_test), train["VolTarget"][:s].mean()),
        "Persistence (Vol21)": train["Vol21"][s:].values * 0.8,
        "HAR-style (log)": np.exp(har.predict(X[s:])),
    }
    table = pd.DataFrame(
        [
            {
                "Model": name,
                "MAE": round(mean_absolute_error(y_test, p), 5),
                "R2": round(r2_score(y_test, p), 4),
            }
            for name, p in preds.items()
        ]
    )

    # final model uses ALL history to forecast the next 5 days
    final = LinearRegression().fit(X, y)
    latest = d[COLS].dropna().iloc[[-1]]
    forecast = float(np.exp(final.predict(np.log(latest + EPS)))[0])
    history = train["VolTarget"]
    return table, forecast, history, len(train) - s


def risk_label(forecast: float, history: pd.Series):
    pct = (history < forecast).mean() * 100
    if pct < 33:
        return "LOW", pct, "green"
    if pct < 67:
        return "NORMAL", pct, "orange"
    return "HIGH", pct, "red"


# ---------------------------------------------------------------- sidebar
st.sidebar.title("📈 StockSense")
name = st.sidebar.selectbox("Stock", list(TICKERS))
ticker = TICKERS[name]
page = st.sidebar.radio(
    "Page", ["Market view", "Risk forecast", "Model comparison", "Direction reality check", "About"]
)
st.sidebar.caption("Educational project. Not financial advice.")

raw = load_data(ticker)
if raw.empty:
    st.error("Could not download data right now. Wait a minute and refresh.")
    st.stop()
data = add_indicators(raw)
last = data.iloc[-1]

# ---------------------------------------------------------------- pages
if page == "Market view":
    st.title(f"{name} ({ticker})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last close", f"₹{last['Close']:.2f}", f"{last['Return'] * 100:.2f}%")
    c2.metric("RSI (14)", f"{last['RSI14']:.1f}")
    c3.metric("Close vs MA50", f"{(last['Close'] / last['MA50'] - 1) * 100:.1f}%")
    c4.metric("Data up to", str(last["Date"].date()))

    left, right = st.columns([1, 3])
    with left:
        days = st.select_slider("Window (trading days)", [60, 120, 250, 500, 1000], value=250)
        show_ma = st.checkbox("Moving averages (21, 50)", True)
        show_rsi = st.checkbox("RSI", True)
        show_macd = st.checkbox("MACD", True)

    d = data.tail(days)
    rows = 1 + int(show_rsi) + int(show_macd)
    heights = [0.6] + [0.2] * (rows - 1)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=heights)
    fig.add_trace(
        go.Candlestick(x=d["Date"], open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="Price"),
        row=1, col=1,
    )
    if show_ma:
        fig.add_trace(go.Scatter(x=d["Date"], y=d["MA21"], name="MA21", line=dict(width=1.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=d["Date"], y=d["MA50"], name="MA50", line=dict(width=1.2)), row=1, col=1)
    r = 2
    if show_rsi:
        fig.add_trace(go.Scatter(x=d["Date"], y=d["RSI14"], name="RSI14"), row=r, col=1)
        fig.add_hline(y=70, line_dash="dot", row=r, col=1)
        fig.add_hline(y=30, line_dash="dot", row=r, col=1)
        r += 1
    if show_macd:
        fig.add_trace(go.Bar(x=d["Date"], y=d["MACD_hist"], name="MACD hist"), row=r, col=1)
        fig.add_trace(go.Scatter(x=d["Date"], y=d["MACD"], name="MACD"), row=r, col=1)
        fig.add_trace(go.Scatter(x=d["Date"], y=d["MACD_signal"], name="Signal"), row=r, col=1)
    fig.update_layout(height=700, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
    right.plotly_chart(fig, use_container_width=True)

elif page == "Risk forecast":
    st.title("5-day risk forecast")
    table, forecast, history, n_test = volatility_model(data)
    label, pct, color = risk_label(forecast, history)
    price = float(last["Close"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Expected avg daily move (next 5 days)", f"{forecast * 100:.2f}%")
    c2.metric("In rupees (approx.)", f"±₹{forecast * price:.1f} per day")
    c3.markdown(f"### Risk level: :{color}[{label}]")
    st.caption(
        f"Today's forecast is higher than {pct:.0f}% of past 5-day periods for this stock. "
        "'Expected daily move' means the average size of daily price change, up or down, "
        "not a prediction of direction."
    )

    recent = data.tail(250).copy()
    recent["Realised 5-day avg |return|"] = recent["Return"].abs().rolling(5).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent["Date"], y=recent["Realised 5-day avg |return|"], name="Realised"))
    next_day = recent["Date"].iloc[-1] + pd.Timedelta(days=7)
    fig.add_trace(
        go.Scatter(x=[recent["Date"].iloc[-1], next_day], y=[recent["Realised 5-day avg |return|"].iloc[-1], forecast],
                   name="Forecast", mode="lines+markers", line=dict(dash="dot"))
    )
    fig.update_layout(height=420, yaxis_tickformat=".2%", margin=dict(l=10, r=10, t=30, b=10),
                      title="Realised volatility (last 250 days) and forecast")
    st.plotly_chart(fig, use_container_width=True)
    st.info("Model: HAR-style linear regression on log of 5, 21 and 63-day volatility. "
            "See the Model comparison page for how it was tested.")

elif page == "Model comparison":
    st.title("Model comparison (volatility forecast)")
    table, forecast, history, n_test = volatility_model(data)
    st.write(f"Chronological 80/20 split. Test set: {n_test} trading days. Lower MAE is better; "
             "R² above 0 means it beats predicting the test average.")
    st.dataframe(table, use_container_width=True, hide_index=True)
    best = table.sort_values("MAE").iloc[0]
    st.success(f"Lowest error on {name}: **{best['Model']}** (MAE {best['MAE']}, R² {best['R2']}).")
    st.caption(
        "In the original Reliance test, HAR-style gave MAE 0.00332 and R² 0.040 vs persistence 0.00345 and "
        "training average 0.00418. The edge is modest. Other stocks may rank the models differently, "
        "and that is a result to report, not hide."
    )

elif page == "Direction reality check":
    st.title("Can we predict up or down? Honest answer: barely")
    up = (data["Return"].dropna() > 0).mean() * 100
    up_recent = (data["Return"].tail(250).dropna() > 0).mean() * 100
    c1, c2 = st.columns(2)
    c1.metric("Up days (all history)", f"{up:.1f}%")
    c2.metric("Up days (last 250)", f"{up_recent:.1f}%")
    st.markdown(
        """
**What the project found on Reliance (test period Jul 2024 to Nov 2026):**

| Model | Direction accuracy |
|---|---|
| Always predict UP | 49.5% |
| Ridge | 48.8% |
| Random Forest (tuned) | 50.2% |
| XGBoost (tuned) | 49.3% |

All models are within noise of a coin flip, and none beat "predict the average" on price error.
This fits the efficient market hypothesis: public price and volume history carries almost no
information about tomorrow's direction. That is why this dashboard forecasts **risk**, not direction.
"""
    )

else:
    st.title("About StockSense")
    st.markdown(
        """
**Problem.** An earlier model that predicted raw next-day prices (Random Forest, price MAE ₹30.88)
did worse than the "tomorrow = today" baseline (₹13.44).

**Fix.** Predicting daily returns removed the trend problem (Random Forest MAE fell to ₹13.55),
but no model beat chance on up/down direction.

**Pivot.** Volatility clusters, so how large tomorrow's moves will be is forecastable.
A simple HAR-style model gave the best result.

**Data.** Yahoo Finance via `yfinance`, daily prices from 2015, split chronologically (no shuffling).

**Limits.** One main stock tested in depth, one train/test split, modest R², no transaction costs or news.

*Educational project. Not financial advice.*
"""
    )
