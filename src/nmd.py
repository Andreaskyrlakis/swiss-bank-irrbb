"""Non-maturity deposits: pass-through of market rates into client rates, stable share."""

import numpy as np
import pandas as pd


def estimate_pass_through(client_rate: pd.Series, market_rate: pd.Series, start: str, end: str) -> dict:
    """OLS of the client rate on the market rate (monthly levels, both in %) over [start, end].

    Returns beta (pass-through), intercept, R^2 and the number of months.
    """
    data = pd.concat([client_rate.rename("client"), market_rate.rename("market")], axis=1).dropna()
    data = data.loc[start:end]
    x = np.column_stack([np.ones(len(data)), data["market"].to_numpy()])
    y = data["client"].to_numpy()
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ coef
    r2 = 1 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2)
    return {"beta": float(coef[1]), "intercept_pct": float(coef[0]), "r2": float(r2), "months": int(len(data))}


def cycle_pass_through(client_rate: pd.Series, market_rate: pd.Series, trough: str, peak: str) -> float:
    """Change in client rate divided by change in market rate between two dates (cumulative beta)."""
    peak, trough = pd.Timestamp(peak), pd.Timestamp(trough)
    return float((client_rate.loc[peak] - client_rate.loc[trough]) /
                 (market_rate.loc[peak] - market_rate.loc[trough]))


def stable_share_from_disclosure(avg_repricing_y: float, max_repricing_y: float) -> float:
    """Stable share s such that a volatile part at 0 and a stable part spread uniformly over
    (0, max] have the published average repricing date: s * max / 2 = avg."""
    return min(1.0, 2 * avg_repricing_y / max_repricing_y)
