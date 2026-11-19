"""Step 2: CHF curve at 31.12.2025, the six FINMA shocks, coupons of the stock and deposit
pass-through, all from SNB data.  -> results/assumptions.json, results/curve.json, figures 1-2
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data, shocks  # noqa: E402
from src.curve import build_chf_curve  # noqa: E402
from src.nmd import cycle_pass_through, estimate_pass_through  # noqa: E402
from src.plotting import COLORS, save, style  # noqa: E402

DATE = "2025-12-30"   # last day of 2025 with Confederation bond quotes in the SNB cube
END_MONTH = pd.Timestamp("2025-12-01")
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
style()

# ---------------------------------------------------------------- 1. curve
bonds = data.confederation_bond_yields(DATE)
saron = data.saron_fixing(DATE)
curve = build_chf_curve(bonds["maturity_y"], bonds["ytm_pct"], saron)
fitted = curve.zero(bonds["maturity_y"].to_numpy())
fit_error_bp = (np.log1p(bonds["ytm_pct"].to_numpy() / 100) - fitted) * 1e4
print(f"Confederation bonds on {DATE}: {len(bonds)}, maturities {bonds.maturity_y.min():.1f}-"
      f"{bonds.maturity_y.max():.1f} y; SARON {saron:.2f}%")
print(f"Svensson: b0={curve.beta0:.4f} b1={curve.beta1:.4f} b2={curve.beta2:.4f} b3={curve.beta3:.4f} "
      f"tau1={curve.tau1:.2f} tau2={curve.tau2:.2f}; RMSE {np.sqrt(np.mean(fit_error_bp**2)):.1f} bp")
key_tenors = [1 / 365, 1, 2, 5, 10, 20, 30]
curve_summary = {
    "date": DATE, "saron_pct": saron, "n_bonds": int(len(bonds)),
    "svensson": {"beta0": curve.beta0, "beta1": curve.beta1, "beta2": curve.beta2, "beta3": curve.beta3, "tau1": curve.tau1, "tau2": curve.tau2},
    "rmse_bp": float(np.sqrt(np.mean(fit_error_bp ** 2))),
    "max_abs_error_bp": float(np.max(np.abs(fit_error_bp))),
    "zero_rates_pct": {f"{t:.3g}y": float(curve.zero(t) * 100) for t in key_tenors},
    "par_swap_rate_5y_pct": float(curve.par_swap_rate(5) * 100),
}
(RESULTS / "curve.json").write_text(json.dumps(curve_summary, indent=2))

finma = data.finma_shocks()
sizes = dict(parallel_bp=finma["parallel_bp"], short_bp=finma["short_bp"], long_bp=finma["long_bp"])
table = shocks.bucket_table(**sizes)
pd.DataFrame(table, index=[b[0] for b in shocks.BUCKETS]).to_csv(RESULTS / "shock_table_bp.csv")

# ---------------------------------------------------------------- 2. coupons of the stock
def window_mean(series: pd.Series, years: float) -> float:
    """Average of a monthly/daily series (in %) over the `years` years ending in December 2025."""
    start = END_MONTH - pd.DateOffset(months=int(round(12 * years)) - 1)
    return float(series.loc[start:END_MONTH + pd.offsets.MonthEnd(0)].mean())


new_business = lambda code: data.snb_series("zikrepro", code, column="D1", reference="M")  # noqa: E731
rendoblid = data.read_snb_cube("rendoblid")
issuer_8y = lambda code: (rendoblid[rendoblid.D0 == code].assign(Date=lambda d: pd.to_datetime(d.Date))  # noqa: E731
                          .set_index("Date")["Value"].dropna().resample("MS").mean())
confed_7y = data.snb_series("rendoblim", "7J")

positions = data.bcv_positions().set_index("item")
coupon_sources = {
    "mortgages_fixed": ("SNB new fixed-rate mortgages, 8-year, mean (zikrepro 80)", new_business("80"), 8.0),
    "customer_loans_fixed": ("SNB new fixed-rate mortgages, 7-year, mean (zikrepro 70), proxy", new_business("70"), 6.6),
    "financial_investments": ("SNB yields of cantonal bonds, 8-year (rendoblid K)", issuer_8y("K"), 12.0),
    "pfandbrief_loans": ("SNB yields of Pfandbrief institutions, 8-year (rendoblid P)", issuer_8y("P"), 13.6),
    "swap_pay_legs": ("SNB Confederation 7-year yield (rendoblim 7J), swap-rate proxy", confed_7y, 7.0),
}
coupons, coupon_notes = {}, {}
for item, (label, series, years) in coupon_sources.items():
    coupons[item] = window_mean(series, years) / 100
    last = min(series.dropna().index.max(), END_MONTH).strftime("%Y-%m")
    coupon_notes[item] = f"{label}; average over the last {years:g} years (data to {last}): {coupons[item]:.2%}"
# short positions: the 31.12.2025 market rate at their ladder length
for item, row in positions.iterrows():
    if item not in coupons and item not in ("sight_deposits", "savings_deposits", "mortgages_variable"):
        tenor = max(2 * row.avg_repricing_y, 1 / 365)
        coupons[item] = float(curve.zero(tenor))
        coupon_notes[item] = f"zero rate at {tenor:.2f} y on 30.12.2025: {coupons[item]:.2%}"

# ---------------------------------------------------------------- 3. client rates and pass-through
saron_monthly = data.snb_series("zimoma", "SARON")
client_series = {"sight_deposits": new_business("S0"), "savings_deposits": new_business("S1"),
                 "mortgages_variable": new_business("MV")}
labels = {"sight_deposits": "Private payment accounts (S0)", "savings_deposits": "Savings deposits (S1)",
          "mortgages_variable": "Variable mortgages (MV)"}
client_rates, pass_through = {}, {}
for item, series in client_series.items():
    client_rates[item] = float(series.loc[END_MONTH]) / 100
    estimate = estimate_pass_through(series, saron_monthly, "2022-06", "2025-12")
    estimate["cycle_beta_up_2022-06_to_2023-12"] = cycle_pass_through(series, saron_monthly, "2022-06", "2023-12")
    estimate["cycle_beta_down_2023-12_to_2025-12"] = cycle_pass_through(series, saron_monthly, "2023-12", "2025-12")
    estimate["snb_series"] = labels[item]
    pass_through[item] = estimate
    print(f"{item:18s} client rate Dec-2025 {client_rates[item]:.3%}  beta {estimate['beta']:.2f} "
          f"(R2 {estimate['r2']:.2f}), up-cycle {estimate['cycle_beta_up_2022-06_to_2023-12']:.2f}, "
          f"down-cycle {estimate['cycle_beta_down_2023-12_to_2025-12']:.2f}")

assumptions = {
    "date": DATE,
    "coupons": coupons, "coupon_notes": coupon_notes,
    "client_rates": client_rates, "pass_through": pass_through,
    "nmd_rule": "volatile part overnight; stable part uniform ladder up to the published maximum; "
                "stable share = 2 x average / maximum (matches IRRBBA1 averages)",
    "ladder_rule": "defined-repricing items: uniform monthly ladder on (0, 2 x published average]",
}
(RESULTS / "assumptions.json").write_text(json.dumps(assumptions, indent=2))
for item, note in coupon_notes.items():
    print(f"  coupon {item:22s} {note}")

# ---------------------------------------------------------------- figures
grid = np.linspace(1 / 365, 40, 400)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
ax.scatter(bonds["maturity_y"], np.log1p(bonds["ytm_pct"] / 100) * 100, s=18, color=COLORS[0],
           label="Confederation bonds (yield)", zorder=3)
ax.scatter([0], [saron], marker="s", s=24, color=COLORS[3], label="SARON fixing", zorder=3)
ax.plot(grid, curve.zero(grid) * 100, color=COLORS[1], lw=2, label="Svensson zero curve")
ax.set(xlabel="Maturity (years)", ylabel="Rate (%, cont. comp.)", title="CHF curve, 30 Dec 2025")
ax.legend(fontsize=8)
ax = axes[1]
for i, (name, f) in enumerate(shocks.all_shocks(**sizes).items()):
    ax.plot(grid, (curve.zero(grid) + f(grid)) * 100, lw=1.6, color=COLORS[i], label=shocks.LABELS[name])
ax.plot(grid, curve.zero(grid) * 100, color="black", lw=1, ls="--", label="Base")
ax.set(xlabel="Maturity (years)", ylabel="Rate (%)", title="Six FINMA shocks (CHF: 150 / 150 / 100 bp)")
ax.legend(fontsize=8, ncol=2)
save(fig, "fig1_curve_and_shocks")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
window = slice("2015-01", "2026-12")
ax.plot(saron_monthly.loc[window], color="black", lw=1.6, label="SARON")
for i, (item, series) in enumerate(client_series.items()):
    ax.plot(series.loc[window], color=COLORS[i], lw=1.4, label=labels[item])
ax.axhline(0, color="grey", lw=0.6)
ax.set(ylabel="%", title="Client rates on new business vs SARON (SNB)")
ax.legend(fontsize=8)
ax = axes[1]
sub = pd.concat([saron_monthly.rename("saron"), client_series["savings_deposits"].rename("sav"),
                 client_series["sight_deposits"].rename("sight")], axis=1, sort=True).loc["2022-06":"2025-12"].dropna()
ax.scatter(sub.saron, sub.sav, s=14, color=COLORS[1], label=f"Savings: beta {pass_through['savings_deposits']['beta']:.2f}")
ax.scatter(sub.saron, sub.sight, s=14, color=COLORS[0], label=f"Payment accounts: beta {pass_through['sight_deposits']['beta']:.2f}")
xs = np.linspace(sub.saron.min(), sub.saron.max(), 10)
for item, color in (("savings_deposits", COLORS[1]), ("sight_deposits", COLORS[0])):
    p = pass_through[item]
    ax.plot(xs, p["intercept_pct"] + p["beta"] * xs, color=color, lw=1)
ax.set(xlabel="SARON (%)", ylabel="Client rate (%)", title="Pass-through, Jun 2022 - Dec 2025 (monthly)")
ax.legend(fontsize=8)
save(fig, "fig2_deposit_pass_through")
