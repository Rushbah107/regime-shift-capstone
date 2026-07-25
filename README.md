# Regime-Shift: Macro-Aware Tactical Asset Allocation Engine

A regime-detection and backtesting pipeline that uses a Hidden Markov Model (HMM) to
classify market conditions (Bull / Bear / Crisis) across four asset classes — equities,
bonds, gold, and crypto — and trades a long/short strategy conditioned on those regimes,
validated with strict walk-forward testing to avoid lookahead bias.

## Assets covered

| Ticker | Asset class |
|---|---|
| `SPY` | US Equities |
| `TLT` | Long-Term US Treasury Bonds |
| `GLD` | Gold (commodity / inflation hedge) |
| `BTC-USD` | Crypto (alternative / risk asset) |

Data period: 2018-01-01 to 2024-01-01, sourced via `yfinance`.

## Key decisions and why

### Why 3 regimes (Bull / Bear / Crisis) instead of 2 or 4+
Two regimes (e.g. just "up" vs "down") collapse two very different failure modes into one
state: a slow grinding drawdown and a sharp volatility spike are not the same thing for
position sizing or risk management. Three regimes is the smallest model that separates
them — Bull (positive momentum, normal vol), Bear (negative momentum, normal vol), and
Crisis (elevated volatility, regardless of momentum direction). More than 3 states
(4–5+) was tried informally and tends to fragment the data into states that are hard to
interpret and unstable across walk-forward folds, given the relatively short training
windows (a few hundred trading days) available in an expanding-window backtest. 3 keeps
the model identifiable and each state statistically well-populated.

### Why these features (21-day momentum, 21-day volatility)
- **Momentum (`mom_21d`)**: cumulative log return over the trailing ~1 trading month.
  Captures trend direction, which is the primary signal that separates Bull from Bear.
- **Volatility (`vol_21d`)**: rolling standard deviation of daily log returns over the
  same window. Captures regime *stress*, which is largely orthogonal to trend direction —
  a market can trend down calmly (Bear) or fall apart violently (Crisis), and volatility
  is what distinguishes the two.

Both are expanding-window z-scored (`min_periods=63`, ~1 trading quarter) rather than
using raw values or a fixed lookback z-score. This does two things:
1. Puts momentum and volatility on comparable scales so the HMM doesn't implicitly
   weight one feature more heavily just because of its raw units.
2. `min_periods=63` prevents unstable z-scores from the first few weeks of data (a
   z-score computed from a handful of points is dominated by noise, not signal) from
   leaking a spurious "Crisis" signal into the very start of the backtest.

Both features are deliberately simple and interpretable — this keeps the regime labels
auditable (you can look at `model.means_` and immediately understand what each state
represents) rather than treating the HMM as an opaque black box.

### Why log returns instead of simple returns
Log returns are additive across time (`sum of log returns over N days == log return of
the N-day holding period`), which makes rolling/expanding aggregation (momentum, monthly
resampling) mathematically clean — no compounding formula needed, just `.sum()`.

### Why walk-forward validation instead of a single train/test split or full-sample fit
A single HMM fit on the entire history "knows" about future crashes and rallies when
assigning early-period regime labels — this is lookahead bias, and it makes a backtest's
performance numbers meaningless (you're implicitly trading with hindsight). The
walk-forward approach refits the HMM from scratch on an **expanding training window**
and only predicts forward into the next unseen block, so at any point in the backtest
the model has only ever seen data available as of that date. This is slower and the
model may be noisier early on (small training samples), but it's the only way the
resulting equity curve reflects what would have been realistically tradeable.

### Why long Bull / short Bear / flat Crisis
This is a deliberately simple, interpretable position rule to test whether the regime
signal has any value at all before layering on more complexity (position sizing,
transaction costs, risk overlays). Flat in Crisis reflects the idea that Crisis is
defined by elevated *volatility* rather than a clear directional signal — taking no
position when the model is least confident about direction avoids whipsaw losses.

### Why positions are lagged by one day (`positions.shift(1)`)
A regime label observed at the close of day *t* can only be acted on starting day
*t+1* — you can't trade on information before it exists. Skipping this shift is one of
the most common ways a backtest silently cheats.

## Repository structure

```
.
├── Project Guide Final.ipynb   # Learning notebook: numpy/pandas/feature-engineering/
│                                 walk-forward mechanics/HMM & PyTorch primers
├── Final Code.ipynb            # Capstone: HMM regime detection, walk-forward backtest,
│                                 performance summary
└── README.md
```

## How to run

### 1. Set up the environment
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install numpy pandas matplotlib yfinance scipy hmmlearn scikit-learn torch cvxpy
```

### 2. Launch the notebook
```bash
jupyter lab   # or open Final Code.ipynb in VS Code / Jupyter
```

### 3. Run cells top to bottom
`Final Code.ipynb` is organized as:
1. Data download + feature engineering (all 4 tickers)
2. Full-sample HMM fit — regime overlay plots + transition matrices (diagnostic view)
3. Walk-forward backtest — HMM refit per fold, regime charts, equity curves
4. Performance summary table (Sharpe, Sortino, max drawdown, Calmar, turnover) vs.
   buy-and-hold benchmark, per asset

## How to reproduce results exactly

- All random elements (`np.random.seed`, `random_state` in `GaussianHMM`) are fixed to
  `42` throughout, so re-running the notebook end-to-end on the same data window should
  reproduce identical regime labels, transition matrices, and performance metrics.
- Results depend on the data pulled from `yfinance` at run time. Since `auto_adjust=True`
  is used, prices are dividend/split-adjusted — if you re-run this months or years later,
  Yahoo Finance's adjusted price history for the same historical dates should still match
  (adjustments are back-applied consistently), but it's worth confirming row counts match
  if you're trying to reproduce a specific past run exactly.
- Walk-forward parameters (`min_train_size=252`, `test_size=21`, `step=21`) are fixed in
  the backtest function signature — changing these will change fold boundaries and
  therefore the exact equity curve, so keep them constant for reproducibility.

## Known limitations

- No transaction costs or slippage modeled — turnover is reported but not yet priced in.
- Test windows (21 trading days) are short, so per-fold state counts can be thin,
  especially for Crisis regimes, which are rare by construction.
- This is a baseline strategy (binary long/short/flat) intended to test whether the
  regime signal carries information — not a production-ready allocation engine.
