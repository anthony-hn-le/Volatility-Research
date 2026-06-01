"""
Data collection and cleaning pipeline.
Primary source: yfinance. Backup: Alpha Vantage (free unadjusted endpoint).
"""
import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from dotenv import load_dotenv

load_dotenv()

ASSETS = [
    'SPY', 'QQQ', 'IWM',
    'AAPL', 'MSFT', 'NVDA',
    'JPM', 'GS', 'BAC',
    'JNJ', 'UNH', 'PFE',
    'AMZN', 'WMT', 'HD',
    'XOM', 'CVX', 'CAT',
]

START_DATE = '2010-01-01'
END_DATE = '2025-12-31'
TRAIN_END = '2017-12-31'
TEST_START = '2018-01-01'

# Implied-volatility proxies for the three index tickers (via yfinance).
# ^RVX (Russell 2000 VIX) is not carried by yfinance — IWM IV is left as NaN.
INDEX_VIX_MAP = {'SPY': '^VIX', 'QQQ': '^VXN'}   # IWM omitted: ^RVX unavailable
INDEX_TICKERS = ['SPY', 'QQQ', 'IWM']             # full list; IWM col → NaN
IV_CACHE_PATH = os.path.join('data', 'processed', 'iv_monthly.csv')


def download_prices(assets=ASSETS, start=START_DATE, end=END_DATE, raw_dir='data/raw'):
    """Download daily OHLCV from yfinance and save one CSV per asset."""
    os.makedirs(raw_dir, exist_ok=True)
    results = {}
    for ticker in assets:
        path = os.path.join(raw_dir, f'{ticker}_daily.csv')
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            print(f"WARNING: no data for {ticker}")
            continue
        df.columns = df.columns.get_level_values(0)
        df.to_csv(path)
        results[ticker] = df
        print(f"{ticker}: {len(df)} rows ({df.index[0].date()} – {df.index[-1].date()})")
    return results


def compute_realized_vol(df, window=21):
    """
    Realized volatility: annualized rolling std of daily log returns.
    Returns series in percent (e.g., 15.0 = 15%).
    """
    log_returns = np.log(df['Close'] / df['Close'].shift(1))
    rv = log_returns.rolling(window).std() * np.sqrt(252) * 100
    return rv.rename(f'rv_{window}d')


def load_and_clean(ticker, raw_dir='data/raw'):
    """Load raw CSV, compute returns and RV, return cleaned DataFrame."""
    path = os.path.join(raw_dir, f'{ticker}_daily.csv')
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = 'Date'
    df = df.sort_index()
    df['return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['rv_21d'] = compute_realized_vol(df, window=21)
    df['regime'] = pd.cut(
        df['rv_21d'],
        bins=[0, 15, 25, np.inf],
        labels=['low', 'medium', 'high']
    )
    return df


def build_master_prices(assets=ASSETS, raw_dir='data/raw', processed_dir='data/processed'):
    """Merge close prices for all assets into a single wide DataFrame."""
    os.makedirs(processed_dir, exist_ok=True)
    closes = {}
    for ticker in assets:
        df = load_and_clean(ticker, raw_dir)
        closes[ticker] = df['Close']
    prices = pd.DataFrame(closes)
    prices.to_csv(os.path.join(processed_dir, 'prices_clean.csv'))
    return prices


def build_realized_vol_matrix(assets=ASSETS, raw_dir='data/raw', processed_dir='data/processed'):
    """Build asset × date matrix of realized volatility."""
    os.makedirs(processed_dir, exist_ok=True)
    rvs = {}
    for ticker in assets:
        df = load_and_clean(ticker, raw_dir)
        rvs[ticker] = df['rv_21d']
    rv_matrix = pd.DataFrame(rvs)
    rv_matrix.to_csv(os.path.join(processed_dir, 'realized_vol.csv'))
    return rv_matrix


def fetch_index_iv(
    start: str = TEST_START,
    end: str = END_DATE,
    processed_dir: str = 'data/processed',
) -> pd.DataFrame:
    """
    Download VIX-family implied-volatility proxies for the index tickers.

    Mapping (yfinance availability as of 2026):
        SPY → ^VIX   (CBOE S&P 500 Volatility Index)
        QQQ → ^VXN   (CBOE Nasdaq-100 Volatility Index)
        IWM → NaN    (^RVX not carried by yfinance)

    Returns a DataFrame of month-end IV values with columns ['SPY','QQQ','IWM']
    in percent — the same units as rv_21d (no further scaling needed).
    Missing tickers produce NaN columns; callers should use .dropna(axis=1).

    Cache-first: if data/processed/iv_monthly.csv already exists the function
    loads and returns it immediately without hitting the network.
    """
    os.makedirs(processed_dir, exist_ok=True)
    cache = os.path.join(processed_dir, 'iv_monthly.csv')

    if os.path.exists(cache):
        print(f"Loading IV matrix from cache: {cache}")
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    series: dict[str, pd.Series] = {}
    for equity_ticker, vix_ticker in INDEX_VIX_MAP.items():
        raw = yf.download(vix_ticker, start=start, end=end,
                          progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            print(f"  WARNING: no data for {vix_ticker} ({equity_ticker}) — column will be NaN")
            continue
        # Flatten MultiIndex columns produced by yfinance ≥1.0
        raw.columns = raw.columns.get_level_values(0)  # type: ignore[assignment]
        # VIX-family tickers are already in % (e.g. 18.5 = 18.5% annualised)
        monthly = raw['Close'].resample('ME').last()
        series[equity_ticker] = monthly
        print(f"  {equity_ticker} ({vix_ticker}): "
              f"{len(monthly)} month-ends, "
              f"range {monthly.min():.1f}–{monthly.max():.1f}%")

    # Build DataFrame with all three columns; missing tickers → NaN
    iv = pd.DataFrame(series).reindex(columns=INDEX_TICKERS)
    iv.index.name = 'Date'
    iv.to_csv(cache)

    available = [c for c in INDEX_TICKERS if iv[c].notna().any()]
    print(f"Saved → {cache}  |  Available: {available}  |  NaN cols: "
          f"{[c for c in INDEX_TICKERS if iv[c].isna().all()]}")
    return iv
