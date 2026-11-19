"""Banking-book cash flows, Delta EVE, Delta NII, outlier test and swap hedge.

Conventions
-----------
* Amounts in CHF millions. Assets are positive, liabilities negative (as in IRRBBA1).
* A *ladder* position is split into monthly tranches whose maturities are spread uniformly
  between 0 and ``max_y`` (so the average repricing date is ``max_y / 2``). Each tranche is a
  bullet loan paying its coupon monthly and its principal at maturity (or earlier if it prepays).
* An *overnight* position reprices tomorrow.
* A *non-maturity deposit* (NMD) is split into a volatile part (overnight) and a stable part that
  is replicated by a uniform ladder up to ``max_y``.
"""

from dataclasses import dataclass, field

from functools import lru_cache

import numpy as np

ONE_DAY = 1 / 365


@dataclass
class Position:
    name: str
    volume: float                 # signed, CHF m
    kind: str                     # "ladder", "overnight", "nmd"
    max_y: float = 0.0            # longest repricing date of the ladder (years)
    coupon: float = 0.0           # annual rate (decimal) of the existing stock
    stable_share: float = 1.0     # NMD only: share replicated by the ladder
    volatile_tenor: float = ONE_DAY  # NMD only: where the volatile part reprices
    nii_mode: str = "contract"    # "contract" (reprices at maturity) or "administered" (client rate)
    beta: float = 1.0             # pass-through of market-rate changes into the position's rate
    rate_floor: float | None = None  # floor on the administered client rate (decimal)
    cpr: float = 0.0              # constant annual prepayment rate (ladder only)
    prepay_incentive: float = 0.0  # extra CPR per 100 bp fall of the 5-year rate (ladder only)
    meta: dict = field(default_factory=dict)

    @property
    def average_repricing(self) -> float:
        if self.kind == "overnight":
            return 0.0
        if self.kind == "nmd":
            return self.stable_share * self.max_y / 2
        return self.max_y / 2


def tranche_maturities(max_y: float) -> np.ndarray:
    """Monthly tranches with maturities at the midpoints of (0, max_y]; mean = max_y / 2."""
    count = max(1, int(round(12 * max_y)))
    return (np.arange(count) + 0.5) * max_y / count


def _ladder_cash_flows(volume, max_y, coupon, cpr=0.0):
    """Cash flows (times, amounts) of a uniform ladder of bullet tranches with monthly coupons
    and a constant prepayment rate ``cpr`` (survival exp(-lambda t), lambda = -ln(1 - cpr))."""
    times, unit_amounts = _unit_ladder_cash_flows(float(max_y), float(coupon), float(cpr))
    return times, volume * unit_amounts


@lru_cache(maxsize=None)
def _unit_ladder_cash_flows(max_y, coupon, cpr):
    """Cash flows of a ladder with total notional 1 (cached: the sensitivity grids reuse them)."""
    maturities = tranche_maturities(max_y)
    notional = 1.0 / len(maturities)
    hazard = -np.log(1 - cpr) if cpr > 0 else 0.0
    times, amounts = [], []
    for maturity in maturities:
        # payment dates: maturity, maturity - 1m, ... (> 0)
        dates = maturity - np.arange(0, int(np.ceil(maturity * 12))) / 12
        dates = np.sort(dates[dates > 1e-9])
        previous = np.concatenate([[0.0], dates[:-1]])
        survival_prev = np.exp(-hazard * previous)
        survival = np.exp(-hazard * dates)
        interest = notional * survival_prev * coupon * (dates - previous)
        principal = notional * (survival_prev - survival)
        principal[-1] += notional * survival[-1]
        times.append(dates)
        amounts.append(interest + principal)
    return np.concatenate(times), np.concatenate(amounts)


def position_cash_flows(position: Position, cpr_override: float | None = None):
    """(times, amounts) of one position."""
    if position.kind == "bullet_fixed":
        times = np.arange(1, int(round(position.max_y)) + 1, dtype=float)
        amounts = np.full(len(times), position.volume * position.coupon)
        amounts[-1] += position.volume
        return times, amounts
    if position.kind == "overnight":
        return np.array([ONE_DAY]), np.array([position.volume])
    if position.kind == "ladder":
        cpr = position.cpr if cpr_override is None else cpr_override
        return _ladder_cash_flows(position.volume, position.max_y, position.coupon, cpr)
    if position.kind == "nmd":
        stable = position.volume * position.stable_share
        volatile = position.volume - stable
        t_stable, a_stable = _ladder_cash_flows(stable, position.max_y, position.coupon)
        return (np.concatenate([[position.volatile_tenor], t_stable]),
                np.concatenate([[volatile], a_stable]))
    raise ValueError(position.kind)


def economic_value(positions, curve, shift=None) -> dict:
    """Present value of every position on the (shifted) curve.

    Prepayment-sensitive ladders use CPR = cpr + prepay_incentive * max(0, -shift(5y) in %).
    """
    values = {}
    for position in positions:
        cpr = None
        if position.kind == "ladder" and position.prepay_incentive > 0 and shift is not None:
            fall_pct = max(0.0, -float(shift(5.0)) * 100)
            cpr = min(0.99, position.cpr + position.prepay_incentive * fall_pct)
        times, amounts = position_cash_flows(position, cpr_override=cpr)
        values[position.name] = float(np.sum(amounts * curve.discount(times, shift)))
    return values


def delta_eve(positions, curve, shift) -> tuple[float, dict]:
    """Change in economic value of equity (shocked minus base), total and by position."""
    base = economic_value(positions, curve)
    shocked = economic_value(positions, curve, shift)
    by_position = {name: shocked[name] - base[name] for name in base}
    return float(sum(by_position.values())), by_position


def delta_nii(positions, shift, horizon=1.0) -> tuple[float, dict]:
    """12-month change in net interest income, constant balance sheet, instantaneous shock.

    * contract positions: each tranche that matures within the horizon is rolled into the same
      original term at a rate moved by beta * shock(original term), for the rest of the horizon;
    * administered positions (deposits, variable mortgages): the client rate moves at once by
      beta * shock(overnight), subject to an optional floor; the whole volume is affected.
    """
    contributions = {}
    for position in positions:
        if position.nii_mode == "administered":
            move = position.beta * float(shift(ONE_DAY))
            new_rate = position.coupon + move
            if position.rate_floor is not None:
                new_rate = max(new_rate, position.rate_floor)
            contributions[position.name] = position.volume * (new_rate - position.coupon) * horizon
            continue
        if position.kind == "bullet_fixed":
            reprice_times = np.array([position.max_y])
            notionals = np.array([position.volume])
            roll_tenor = position.max_y
        elif position.kind == "overnight":
            reprice_times = np.array([ONE_DAY])
            notionals = np.array([position.volume])
            roll_tenor = ONE_DAY
        else:
            reprice_times = tranche_maturities(position.max_y)
            notionals = np.full(len(reprice_times), position.volume / len(reprice_times))
            roll_tenor = max(position.max_y, ONE_DAY)
        inside = reprice_times < horizon
        move = position.beta * float(shift(roll_tenor))
        contributions[position.name] = float(
            np.sum(notionals[inside] * move * (horizon - reprice_times[inside])))
    return float(sum(contributions.values())), contributions


def payer_swap(notional: float, tenor: float, fixed_rate: float, name="hedge_swap"):
    """A pay-fixed / receive-SARON swap as two positions: an overnight floating asset and a
    fixed-rate bullet liability with annual coupons (value ~0 at the par rate)."""
    floating = Position(f"{name}_float", notional, "overnight")
    fixed = Position(f"{name}_fixed", -notional, "bullet_fixed", max_y=tenor, coupon=fixed_rate)
    return [floating, fixed]


def outlier_ratio(delta_eves: dict, tier1: float) -> tuple[str, float]:
    """Worst (most negative) Delta EVE across scenarios as a share of Tier 1."""
    worst = min(delta_eves, key=lambda name: delta_eves[name])
    return worst, -delta_eves[worst] / tier1


def repricing_profile(position: Position):
    """(times, notionals) at which the principal of a position reprices (no prepayment)."""
    if position.kind == "overnight":
        return np.array([ONE_DAY]), np.array([position.volume])
    if position.kind == "bullet_fixed":
        return np.array([position.max_y]), np.array([position.volume])
    maturities = tranche_maturities(position.max_y)
    if position.kind == "ladder":
        return maturities, np.full(len(maturities), position.volume / len(maturities))
    stable = position.volume * position.stable_share
    return (np.concatenate([[position.volatile_tenor], maturities]),
            np.concatenate([[position.volume - stable], np.full(len(maturities), stable / len(maturities))]))
