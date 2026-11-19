"""Turn the published BCV repricing table (IRRBBA1) and the documented assumptions into positions.

All assumptions are written in ``results/assumptions.json`` by ``scripts/02_curve_and_inputs.py``
(coupons from SNB history, pass-through betas from SNB client rates). This module only applies them.
"""

import json

from .data import ROOT, bcv_positions
from .model import Position
from .nmd import stable_share_from_disclosure

ASSUMPTIONS_FILE = ROOT / "results" / "assumptions.json"

# Items the model treats as overnight although they sit in the "defined" block
OVERNIGHT_DEFINED = {"mortgages_money_market"}   # money-market (SARON) mortgages, avg 0.0 y
# Items whose client rate is administered by the bank (NII uses a pass-through beta)
ADMINISTERED = {"sight_deposits": "S0", "savings_deposits": "S1", "mortgages_variable": "MV"}


def load_assumptions() -> dict:
    return json.loads(ASSUMPTIONS_FILE.read_text(encoding="utf-8"))


def build_positions(assumptions: dict | None = None, stable_shares: dict | None = None,
                    max_tenors: dict | None = None, betas: dict | None = None,
                    deposit_floor: float | None = 0.0, include_swaps: bool = True,
                    mortgage_cpr: float = 0.0, mortgage_incentive: float = 0.0) -> list[Position]:
    """Positions of BCV's banking book at 31.12.2025.

    Optional overrides (used by the sensitivity analysis): NMD stable shares and ladder lengths,
    pass-through betas, deposit-rate floor, removal of the existing hedging swaps, mortgage
    prepayment (constant CPR plus a refinancing incentive per 100 bp fall in rates).
    """
    assumptions = assumptions or load_assumptions()
    coupons = assumptions["coupons"]
    client = assumptions["client_rates"]
    beta_est = {k: v["beta"] for k, v in assumptions["pass_through"].items()}
    if betas:
        beta_est.update(betas)
    positions = []
    for row in bcv_positions().itertuples(index=False):
        if not include_swaps and row.item.startswith("swap_"):
            continue
        coupon = coupons.get(row.item, 0.0)
        if row.repricing == "overnight" or row.item in OVERNIGHT_DEFINED:
            positions.append(Position(row.item, row.volume_chf_m, "overnight", coupon=coupon))
        elif row.repricing == "defined":
            position = Position(row.item, row.volume_chf_m, "ladder", max_y=2 * row.avg_repricing_y,
                                coupon=coupon)
            if row.item == "mortgages_fixed":
                position.cpr = mortgage_cpr
                position.prepay_incentive = mortgage_incentive
            positions.append(position)
        else:  # non-maturity positions
            max_y = (max_tenors or {}).get(row.item, row.max_repricing_y)
            share = (stable_shares or {}).get(
                row.item, stable_share_from_disclosure(row.avg_repricing_y, row.max_repricing_y))
            position = Position(row.item, row.volume_chf_m, "nmd", max_y=max_y, stable_share=share,
                                coupon=coupon)
            if row.item in ADMINISTERED:
                position.nii_mode = "administered"
                position.coupon = client[row.item]
                position.beta = beta_est[row.item]
                position.rate_floor = deposit_floor if row.side == "liability" else None
            positions.append(position)
    return positions
