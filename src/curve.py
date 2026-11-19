"""Nelson-Siegel-Svensson zero curve fitted to Swiss Confederation bond yields plus SARON.

Rates are continuously compounded, in decimals, as functions of maturity t in years.
"""

from dataclasses import dataclass

import numpy as np


def _loadings(t, tau):
    x = np.maximum(np.asarray(t, dtype=float), 1e-6) / tau
    slope = (1 - np.exp(-x)) / x
    return slope, slope - np.exp(-x)


def svensson(t, beta0, beta1, beta2, beta3, tau1, tau2):
    """Svensson (1994) extension of Nelson-Siegel (1987): a second hump with its own decay."""
    slope, hump1 = _loadings(t, tau1)
    _, hump2 = _loadings(t, tau2)
    return beta0 + beta1 * slope + beta2 * hump1 + beta3 * hump2


@dataclass
class ZeroCurve:
    beta0: float
    beta1: float
    beta2: float
    beta3: float
    tau1: float
    tau2: float

    def zero(self, t):
        """Continuously compounded zero rate (decimal)."""
        return svensson(t, self.beta0, self.beta1, self.beta2, self.beta3, self.tau1, self.tau2)

    def shifted(self, bp: float) -> "ZeroCurve":
        """Same curve moved in parallel by `bp` basis points."""
        return ZeroCurve(self.beta0 + bp / 1e4, self.beta1, self.beta2, self.beta3, self.tau1, self.tau2)

    def discount(self, t, shift=None):
        """Discount factor exp(-(r(t) + shift(t)) t); shift is a function of t in decimals."""
        t = np.asarray(t, dtype=float)
        rate = self.zero(t)
        if shift is not None:
            rate = rate + shift(t)
        return np.exp(-rate * t)

    def par_swap_rate(self, tenor, shift=None):
        """Fixed rate of a par swap with annual fixed payments against an overnight (SARON) leg."""
        times = np.arange(1, int(round(tenor)) + 1, dtype=float)
        dfs = self.discount(times, shift)
        return (1 - dfs[-1]) / dfs.sum()


def fit_svensson(maturities, rates, weights=None, tau1_grid=np.linspace(0.3, 10, 98),
                 tau2_grid=np.linspace(2, 40, 77)) -> ZeroCurve:
    """Weighted least squares. For each (tau1, tau2) on a grid the betas are linear; keep the best."""
    maturities = np.asarray(maturities, dtype=float)
    rates = np.asarray(rates, dtype=float)
    root_w = np.sqrt(np.ones_like(rates) if weights is None else np.asarray(weights, dtype=float))
    best = None
    for tau1 in tau1_grid:
        slope, hump1 = _loadings(maturities, tau1)
        for tau2 in tau2_grid:
            if tau2 <= tau1 + 0.5:
                continue
            _, hump2 = _loadings(maturities, tau2)
            design = np.column_stack([np.ones_like(slope), slope, hump1, hump2])
            betas, *_ = np.linalg.lstsq(design * root_w[:, None], rates * root_w, rcond=None)
            error = float(np.sum(root_w ** 2 * (design @ betas - rates) ** 2))
            if best is None or error < best[0]:
                best = (error, betas, tau1, tau2)
    assert best is not None
    _, betas, tau1, tau2 = best
    b0, b1, b2, b3 = (float(b) for b in betas)
    return ZeroCurve(b0, b1, b2, b3, float(tau1), float(tau2))


def build_chf_curve(bond_maturities, bond_ytm_pct, saron_pct) -> ZeroCurve:
    """CHF risk-free curve: SARON at one day plus Confederation bonds.

    Confederation coupons are close to the yields (0-1.5%), so yields to maturity are used as
    zero rates (the coupon effect is a few basis points). Annual yields -> continuous: ln(1 + y).
    The SARON point gets the weight of ten bonds so that the short end starts at the fixing.
    """
    maturities = np.concatenate([[1 / 365], np.asarray(bond_maturities, dtype=float)])
    yields = np.concatenate([[saron_pct], np.asarray(bond_ytm_pct, dtype=float)]) / 100
    weights = np.concatenate([[10.0], np.ones(len(maturities) - 1)])
    return fit_svensson(maturities, np.log1p(yields), weights)
