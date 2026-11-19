"""Step 3: Delta EVE and Delta NII of BCV's banking book under the six FINMA shocks, comparison
with the published IRRBB1 table, explanation of the gaps, outlier test.
-> results/irrbb_results.json, results/model_vs_published.csv, figures 3-6
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data, shocks  # noqa: E402
from src.bcv import build_positions, load_assumptions  # noqa: E402
from src.curve import ZeroCurve  # noqa: E402
from src.model import (delta_eve, delta_nii, economic_value, outlier_ratio,  # noqa: E402
                       repricing_profile)
from src.plotting import COLORS, save, style  # noqa: E402

RESULTS = ROOT / "results"
style()
curve_info = json.loads((RESULTS / "curve.json").read_text())
curve = ZeroCurve(**curve_info["svensson"])
assumptions = load_assumptions()
finma = data.finma_shocks()
key = data.key_figures()
TIER1 = key["tier1_parent"]["value"]
SIZES = dict(parallel_bp=finma["parallel_bp"], short_bp=finma["short_bp"], long_bp=finma["long_bp"])
SHOCKS = shocks.all_shocks(**SIZES)
published = data.bcv_published_irrbb1()
pub = published[published.date == "2025-12-31"].set_index("scenario")


def run(positions, shock_set=SHOCKS, the_curve=curve):
    eve = {name: delta_eve(positions, the_curve, f)[0] for name, f in shock_set.items()}
    nii = {name: delta_nii(positions, f)[0] for name, f in shock_set.items()}
    return eve, nii


# ---------------------------------------------------------------- 1. base model
positions = build_positions(assumptions)
eve, nii = run(positions)
base_value = sum(economic_value(positions, curve).values())
worst, ratio = outlier_ratio(eve, TIER1)
_, by_position_up = delta_eve(positions, curve, SHOCKS["parallel_up"])
_, nii_by_position_up = delta_nii(positions, SHOCKS["parallel_up"])
_, nii_by_position_down = delta_nii(positions, SHOCKS["parallel_down"])

table = pd.DataFrame({
    "model_delta_eve": pd.Series(eve), "published_delta_eve": pub["delta_eve_chf_m"],
    "model_delta_nii": pd.Series(nii), "published_delta_nii": pub["delta_nii_chf_m"],
}).loc[shocks.SCENARIOS]
table["eve_gap"] = table.model_delta_eve - table.published_delta_eve
table["model_eve_pct_tier1"] = table.model_delta_eve / TIER1 * 100
table["published_eve_pct_tier1"] = table.published_delta_eve / TIER1 * 100
table.round(1).to_csv(RESULTS / "model_vs_published.csv")
print(table.round(0).to_string())
print(f"EVE of the banking book (model): {base_value:,.0f}; worst {worst}: {ratio:.1%} of Tier 1")

# ΔNII without the zero floor on deposit rates (negative rates passed to clients)
no_floor = build_positions(assumptions, deposit_floor=None)
nii_no_floor = {s: delta_nii(no_floor, SHOCKS[s])[0] for s in ("parallel_up", "parallel_down")}

# ---------------------------------------------------------------- 2. explaining the gaps
def calibrated_coupons_to_curve():
    """Cash flows without commercial margin: every coupon = risk-free zero rate at its ladder length."""
    a = json.loads(json.dumps(assumptions))
    for p in build_positions(assumptions):
        if p.kind == "ladder":
            a["coupons"][p.name] = float(curve.zero(p.max_y))
    for item in a["client_rates"]:
        a["client_rates"][item] = float(curve.zero(1 / 365)) if item == "mortgages_variable" else 0.0
    return a


variants = {}
variants["Base model"] = eve
one_month = [replace(p, volatile_tenor=1 / 12) if p.kind == "nmd" else p for p in positions]
variants["NMD volatile part at 1 month (BCV rule)"] = run(one_month)[0]
variants["No commercial margin in cash flows"] = run(build_positions(calibrated_coupons_to_curve()))[0]
spread_curve = curve.shifted(20)
variants["Curve +20 bp (swap-spread proxy)"] = run(positions, the_curve=spread_curve)[0]

# NMD stable shares that best reproduce the six published ΔEVE (max tenors kept at 5 and 10 y)
grid = np.round(np.arange(0.0, 1.0001, 0.02), 2)
best = None
for s_sight in grid:
    for s_sav in grid:
        trial = build_positions(assumptions, stable_shares={"sight_deposits": s_sight, "savings_deposits": s_sav})
        trial_eve = run(trial)[0]
        rmse = np.sqrt(np.mean([(trial_eve[s] - pub.loc[s, "delta_eve_chf_m"]) ** 2 for s in shocks.SCENARIOS]))
        if best is None or rmse < best[0]:
            best = (rmse, s_sight, s_sav, trial_eve)
rmse_fit, s_sight, s_sav, eve_fit = best
variants[f"NMD stable shares fitted ({s_sight:.0%} sight, {s_sav:.0%} savings)"] = eve_fit
variant_table = pd.DataFrame(variants).T[shocks.SCENARIOS]
variant_table.loc["Published (BCV)"] = pub.loc[shocks.SCENARIOS, "delta_eve_chf_m"]
variant_table["RMSE vs published"] = [
    np.sqrt(np.mean([(row[s] - pub.loc[s, "delta_eve_chf_m"]) ** 2 for s in shocks.SCENARIOS]))
    for _, row in variant_table.iterrows()]
variant_table.round(0).to_csv(RESULTS / "gap_variants.csv")
print(variant_table.round(0).to_string())
base_rmse = float(variant_table.loc["Base model", "RMSE vs published"])

# ---------------------------------------------------------------- 3. outlier test, history and 2028 draft
history = published.copy()
history["ratio_pct"] = -published.delta_eve_chf_m / published.tier1_chf_m * 100
worst_by_date = history.groupby("date").ratio_pct.max()
draft = shocks.all_shocks(parallel_bp=finma["draft_2028_parallel_bp"], short_bp=finma["draft_2028_short_bp"],
                          long_bp=finma["long_bp"])
eve_draft = {name: delta_eve(positions, curve, draft[name])[0] for name in shocks.SCENARIOS}
worst_draft, ratio_draft = outlier_ratio(eve_draft, TIER1)
published_ratio = -pub.loc["parallel_up", "delta_eve_chf_m"] / TIER1
scaled_published_draft = published_ratio * finma["draft_2028_parallel_bp"] / finma["parallel_bp"]

# ---------------------------------------------------------------- 4. consistency checks
bs = key["parent_balance_sheet"]
irr = data.bcv_positions().set_index("item").volume_chf_m
checks = pd.DataFrame({
    "IRRBBA1": [irr[["mortgages_money_market", "mortgages_fixed", "mortgages_variable"]].sum(),
                -irr[["customer_term_deposits", "sight_deposits", "savings_deposits"]].sum(),
                -irr["pfandbrief_loans"], irr["financial_investments"], irr["other_claims"]],
    "Annual report": [bs["mortgage_loans"], bs["customer_deposits"], bs["pfandbrief_loans"],
                      bs["financial_investments"], bs["liquidity"]],
}, index=["Mortgages (fixed + money-market + variable)", "Customer deposits (term + sight + savings)",
          "Pfandbrief loans", "Financial investments", "Other claims vs liquidity"])
checks["gap %"] = (checks["IRRBBA1"] / checks["Annual report"] - 1) * 100
checks.round(1).to_csv(RESULTS / "balance_sheet_check.csv")
print(checks.round(1).to_string())
income = sum(p.volume * p.coupon for p in positions if p.volume > 0)
expense = sum(p.volume * p.coupon for p in positions if p.volume < 0)
income_reported = key["parent_income_2025"]

# ---------------------------------------------------------------- save
results = {
    "tier1_parent": TIER1,
    "eve_banking_book_model": base_value,
    "delta_eve_model": eve, "delta_nii_model": nii,
    "delta_nii_no_deposit_floor": nii_no_floor,
    "delta_eve_published": pub["delta_eve_chf_m"].to_dict(),
    "delta_nii_published": pub["delta_nii_chf_m"].dropna().to_dict(),
    "worst_scenario_model": worst, "worst_ratio_model": ratio,
    "worst_ratio_published": published_ratio,
    "parallel_up_gap_pct": (eve["parallel_up"] / pub.loc["parallel_up", "delta_eve_chf_m"] - 1) * 100,
    "parallel_down_gap_pct": (eve["parallel_down"] / pub.loc["parallel_down", "delta_eve_chf_m"] - 1) * 100,
    "rmse_six_scenarios_base": base_rmse,
    "fitted_nmd_stable_shares": {"sight": s_sight, "savings": s_sav, "rmse": rmse_fit,
                                 "implied_avg_repricing_sight_y": s_sight * 5 / 2,
                                 "implied_avg_repricing_savings_y": s_sav * 10 / 2},
    "outlier_history_pct": worst_by_date.to_dict(),
    "draft_2028": {"delta_eve": eve_draft, "worst": worst_draft, "ratio": ratio_draft,
                   "published_scaled_ratio": scaled_published_draft},
    "delta_eve_by_position_parallel_up": by_position_up,
    "delta_nii_by_position_parallel_up": nii_by_position_up,
    "delta_nii_by_position_parallel_down": nii_by_position_down,
    "modelled_annual_interest": {"income": income, "expense": expense, "net": income + expense,
                                 "reported_2025": income_reported},
}
(RESULTS / "irrbb_results.json").write_text(json.dumps(results, indent=2, default=float))
print(f"Draft 2028 shocks: worst {worst_draft} {ratio_draft:.1%} of Tier 1 "
      f"(published parallel-up scaled: {scaled_published_draft:.1%})")
print(f"Modelled run-rate interest: income {income:.0f}, expense {expense:.0f}, net {income + expense:.0f} "
      f"(reported 2025 gross interest result {income_reported['gross_interest_income']})")
print(f"Fitted NMD shares: sight {s_sight:.2f}, savings {s_sav:.2f}, RMSE {rmse_fit:.0f} vs base {base_rmse:.0f}")

# ---------------------------------------------------------------- figures
labels = [shocks.LABELS[s] for s in shocks.SCENARIOS]

# Fig 3: repricing gap by FINMA bucket
edges = [b[2] for b in shocks.BUCKETS]
names = [b[0] for b in shocks.BUCKETS]
assets_b, liabs_b = np.zeros(len(names)), np.zeros(len(names))
for p in positions:
    times, notionals = repricing_profile(p)
    idx = np.searchsorted(edges, times, side="left")
    idx = np.minimum(idx, len(names) - 1)
    for i, n in zip(idx, notionals):
        (assets_b if n > 0 else liabs_b)[i] += n
fig, ax = plt.subplots(figsize=(11, 4))
x = np.arange(len(names))
ax.bar(x, assets_b / 1e3, color=COLORS[0], label="Assets (incl. swap receive legs)")
ax.bar(x, liabs_b / 1e3, color=COLORS[1], label="Liabilities (incl. swap pay legs, NMD replication)")
ax.plot(x, (assets_b + liabs_b) / 1e3, "ko-", ms=4, lw=1.2, label="Net gap")
ax.set_xticks(x, names, rotation=45)
ax.set(ylabel="CHF bn", title="Repricing gap of BCV's banking book by FINMA time bucket (model, 31 Dec 2025)")
ax.axhline(0, color="grey", lw=0.6)
ax.legend(fontsize=8)
save(fig, "fig3_repricing_gap")

# Fig 4: model vs published
fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [3, 1.3]})
ax = axes[0]
x = np.arange(len(labels))
ax.bar(x - 0.2, table.model_delta_eve, 0.4, color=COLORS[0], label="This model")
ax.bar(x + 0.2, table.published_delta_eve, 0.4, color=COLORS[4], label="BCV published (IRRBB1)")
for xi, (m, p_) in enumerate(zip(table.model_delta_eve, table.published_delta_eve)):
    ax.text(xi - 0.2, m + (12 if m > 0 else -30), f"{m:.0f}", ha="center", fontsize=7)
    ax.text(xi + 0.2, p_ + (12 if p_ > 0 else -30), f"{p_:.0f}", ha="center", fontsize=7)
ax.set_xticks(x, labels, rotation=20)
ax.axhline(0, color="grey", lw=0.6)
ax.set(ylabel="CHF m", title="ΔEVE by scenario, 31 Dec 2025")
ax.legend(fontsize=8)
ax = axes[1]
nl = ["Parallel up", "Parallel down"]
mv = [nii["parallel_up"], nii["parallel_down"]]
pv = [pub.loc["parallel_up", "delta_nii_chf_m"], pub.loc["parallel_down", "delta_nii_chf_m"]]
nf = [nii_no_floor["parallel_up"], nii_no_floor["parallel_down"]]
x = np.arange(2)
ax.bar(x - 0.27, mv, 0.27, color=COLORS[0], label="Model (deposit rates ≥ 0)")
ax.bar(x, nf, 0.27, color=COLORS[5], label="Model (no floor)")
ax.bar(x + 0.27, pv, 0.27, color=COLORS[4], label="BCV published")
ax.set_xticks(x, nl)
ax.axhline(0, color="grey", lw=0.6)
ax.set(ylabel="CHF m", title="ΔNII, 12 months")
ax.legend(fontsize=7)
save(fig, "fig4_model_vs_published")

# Fig 5: ΔEVE by position, parallel up
order = sorted(by_position_up, key=lambda k: by_position_up[k])
vals = [by_position_up[k] for k in order if abs(by_position_up[k]) > 1]
keys = [k for k in order if abs(by_position_up[k]) > 1]
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.barh(keys, vals, color=[COLORS[1] if v < 0 else COLORS[2] for v in vals])
ax.axvline(0, color="grey", lw=0.6)
ax.set(xlabel="CHF m", title=f"Who drives ΔEVE in the parallel-up shock (total {eve['parallel_up']:.0f})")
save(fig, "fig5_eve_by_position")

# Fig 6: outlier test
fig, ax = plt.subplots(figsize=(9, 4))
dates = list(worst_by_date.index)
vals = list(worst_by_date.values)
xs = np.arange(len(dates) + 2)
ax.bar(xs[:len(dates)], vals, color=COLORS[4], label="BCV published (worst scenario)")
ax.bar(xs[len(dates)], ratio * 100, color=COLORS[0], label="This model, 31.12.2025")
ax.bar(xs[len(dates) + 1], ratio_draft * 100, color=COLORS[3], label="Model with 2028 draft shocks")
ax.axhline(15, color=COLORS[1], ls="--", lw=1.5, label="FINMA outlier threshold 15%")
for xi, v in zip(xs, vals + [ratio * 100, ratio_draft * 100]):
    ax.text(xi, v + 0.3, f"{v:.1f}%", ha="center", fontsize=8)
ax.set_xticks(xs, dates + ["model\n31.12.2025", "model\ndraft 2028"])
ax.set(ylabel="−ΔEVE / Tier 1 (%)", ylim=(0, 18), title="Supervisory outlier test (ΔEVE / Tier 1)")
ax.legend(fontsize=8, loc="lower right")
save(fig, "fig6_outlier_test")
