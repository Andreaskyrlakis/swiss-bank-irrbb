import numpy as np
import pandas as pd
import pytest

from src import shocks
from src.curve import ZeroCurve, fit_svensson, svensson
from src.data import bcv_positions
from src.model import (Position, delta_eve, delta_nii, economic_value, payer_swap,
                       position_cash_flows, tranche_maturities)
from src.nmd import estimate_pass_through, stable_share_from_disclosure

FLAT = ZeroCurve(0.01, 0.0, 0.0, 0.0, 1.0, 5.0)   # flat 1% continuously compounded


# ---------------------------------------------------------------- shocks
def test_shocks_match_finma_annex_2_chf_table():
    # values printed in FINMA Circ. 2019/2 Annex 2 for CHF (bp), buckets 1, 2 and 4
    table = shocks.bucket_table(parallel_bp=150, short_bp=150, long_bp=100)
    assert table["parallel_up"][0] == 150 and table["parallel_down"][-1] == -150
    assert table["short_up"][0] == 150 and table["short_up"][1] == 148 and table["short_up"][3] == 137
    assert table["steepener"][0] == -97
    assert table["flattener"][0] == 120


def test_no_floor_on_down_shock():
    down = shocks.shock_function("parallel_down")
    assert down(np.array([0.01, 5.0, 30.0])) == pytest.approx([-0.015] * 3)


# ---------------------------------------------------------------- curve
def test_svensson_fit_recovers_known_curve():
    maturities = np.array([0.003, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])
    true = svensson(maturities, 0.02, -0.015, 0.01, 0.005, 1.5, 10.0)
    curve = fit_svensson(maturities, true)
    assert np.max(np.abs(curve.zero(maturities) - true)) < 2e-5


def test_par_payer_swap_has_zero_value():
    curve = ZeroCurve(0.004, -0.004, 0.0, 0.0, 2.0, 8.0)
    rate = curve.par_swap_rate(5)
    value = sum(economic_value(payer_swap(1000.0, 5, rate), curve).values())
    assert abs(value) < 0.05      # CHF m on a CHF 1 bn swap (one day of SARON accrual)


# ---------------------------------------------------------------- cash flows
def test_ladder_average_repricing_is_half_the_maximum():
    assert tranche_maturities(8.0).mean() == pytest.approx(4.0)
    assert Position("x", 100, "nmd", max_y=10, stable_share=0.36).average_repricing == pytest.approx(1.8)


def test_ladder_principal_sums_to_volume():
    position = Position("x", 1000.0, "ladder", max_y=6.0, coupon=0.0)
    _, amounts = position_cash_flows(position)
    assert amounts.sum() == pytest.approx(1000.0)


def test_delta_eve_matches_duration_for_a_zero_coupon_bullet():
    bond = Position("bond", 1000.0, "bullet_fixed", max_y=5, coupon=0.0)
    small = lambda t: 0.0001 + 0 * np.asarray(t, dtype=float)  # noqa: E731
    change, _ = delta_eve([bond], FLAT, small)
    value = sum(economic_value([bond], FLAT).values())
    assert change == pytest.approx(-5 * value * 0.0001, rel=1e-3)


def test_prepayment_lowers_value_of_above_market_loan():
    slow = Position("m", 1000.0, "ladder", max_y=8.0, coupon=0.03, cpr=0.0)
    fast = Position("m", 1000.0, "ladder", max_y=8.0, coupon=0.03, cpr=0.2)
    assert economic_value([fast], FLAT)["m"] < economic_value([slow], FLAT)["m"]


# ---------------------------------------------------------------- NII
def test_nii_of_overnight_asset():
    up = shocks.shock_function("parallel_up", parallel_bp=100)
    nii, _ = delta_nii([Position("cash", 1000.0, "overnight")], up)
    assert nii == pytest.approx(1000.0 * 0.01 * (1 - 1 / 365))


def test_administered_deposit_rate_is_floored():
    deposit = Position("dep", -1000.0, "nmd", max_y=5, coupon=0.001, nii_mode="administered",
                       beta=1.0, rate_floor=0.0)
    down = shocks.shock_function("parallel_down")
    nii, _ = delta_nii([deposit], down)
    assert nii == pytest.approx(-1000.0 * (0.0 - 0.001))   # client rate can only fall to 0


# ---------------------------------------------------------------- NMD
def test_stable_share_from_disclosure():
    assert stable_share_from_disclosure(1.8, 10.0) == pytest.approx(0.36)
    assert stable_share_from_disclosure(1.8, 5.0) == pytest.approx(0.72)


def test_pass_through_regression_recovers_beta():
    index = pd.date_range("2020-01-01", periods=48, freq="MS")
    market = pd.Series(np.linspace(-0.7, 1.7, 48), index=index)
    client = 0.05 + 0.4 * market
    estimate = estimate_pass_through(client, market, "2020-01", "2023-12")
    assert estimate["beta"] == pytest.approx(0.4) and estimate["r2"] == pytest.approx(1.0)


# ---------------------------------------------------------------- data
def test_transcribed_irrbba1_adds_up_to_published_total():
    # the published lines are rounded to CHF 1 m; their sum (3,516) is within rounding of the total 3,517
    assert abs(bcv_positions().volume_chf_m.sum() - 3517) <= 2
