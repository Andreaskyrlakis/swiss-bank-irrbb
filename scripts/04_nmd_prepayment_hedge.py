"""Step 4: sensitivity to the deposit model, mortgage prepayment option, swap hedging.
-> results/sensitivity_results.json, figures 7-9
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data, shocks  # noqa: E402
from src.bcv import build_positions, load_assumptions  # noqa: E402
from src.curve import ZeroCurve  # noqa: E402
from src.model import delta_eve, delta_nii, economic_value, outlier_ratio, payer_swap  # noqa: E402
from src.plotting import COLORS, save, style  # noqa: E402

RESULTS = ROOT / "results"
style()
curve = ZeroCurve(**json.loads((RESULTS / "curve.json").read_text())["svensson"])
assumptions = load_assumptions()
finma = data.finma_shocks()
TIER1 = data.key_figures()["tier1_parent"]["value"]
SHOCKS = shocks.all_shocks(parallel_bp=finma["parallel_bp"], short_bp=finma["short_bp"], long_bp=finma["long_bp"])
DRAFT = shocks.all_shocks(parallel_bp=finma["draft_2028_parallel_bp"], short_bp=finma["draft_2028_short_bp"],
                          long_bp=finma["long_bp"])
UP, DOWN = SHOCKS["parallel_up"], SHOCKS["parallel_down"]
NMD = ("sight_deposits", "savings_deposits")
out = {}

# ---------------------------------------------------------------- 1. NMD model sensitivity
shares = np.linspace(0.0, 1.0, 11)
tenors = np.array([2.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0])
heat = np.zeros((len(tenors), len(shares)))
for i, tenor in enumerate(tenors):
    for j, share in enumerate(shares):
        positions = build_positions(assumptions, stable_shares={k: share for k in NMD},
                                    max_tenors={k: tenor for k in NMD})
        heat[i, j] = -delta_eve(positions, curve, UP)[0] / TIER1 * 100
out["nmd_grid"] = {"stable_shares": shares.tolist(), "max_tenors": tenors.tolist(),
                   "parallel_up_loss_pct_tier1": heat.tolist()}
# a one-at-a-time view around the base model
base_positions = build_positions(assumptions)
base_up = delta_eve(base_positions, curve, UP)[0]
oat = {}
for item in NMD:
    base_share = next(p.stable_share for p in base_positions if p.name == item)
    for factor in (0.5, 1.5):
        positions = build_positions(assumptions, stable_shares={item: min(1.0, base_share * factor)})
        oat[f"{item} stable share x{factor}"] = delta_eve(positions, curve, UP)[0]
out["nmd_one_at_a_time_parallel_up"] = oat
out["nmd_base_parallel_up"] = base_up

betas = np.linspace(0, 1, 11)
nii_up, nii_down, nii_down_nofloor = [], [], []
for b in betas:
    positions = build_positions(assumptions, betas={"savings_deposits": b, "sight_deposits": b})
    nii_up.append(delta_nii(positions, UP)[0])
    nii_down.append(delta_nii(positions, DOWN)[0])
    positions = build_positions(assumptions, betas={"savings_deposits": b, "sight_deposits": b}, deposit_floor=None)
    nii_down_nofloor.append(delta_nii(positions, DOWN)[0])
out["nii_vs_deposit_beta"] = {"betas": betas.tolist(), "parallel_up": nii_up, "parallel_down": nii_down,
                              "parallel_down_no_floor": nii_down_nofloor}
print("NMD: base parallel-up dEVE", round(base_up), {k: round(v) for k, v in oat.items()})
print("dNII up at beta 0 / 0.35 / 1:", [round(nii_up[i]) for i in (0, 3, 10)])

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
im = ax.imshow(heat, origin="lower", aspect="auto", cmap="RdYlGn_r",
               extent=[-0.05, 1.05, -0.5, len(tenors) - 0.5])
ax.contour(shares, np.arange(len(tenors)), heat, levels=[15], colors="black", linewidths=1.5)
ax.set_yticks(np.arange(len(tenors)), [f"{t:g}" for t in tenors])
ax.set(xlabel="Stable share of sight and savings deposits", ylabel="Longest replication tenor (years)",
       title="Parallel-up loss, % of Tier 1 (black: 15% line)")
fig.colorbar(im, ax=ax)
ax = axes[1]
ax.plot(betas, nii_up, "o-", color=COLORS[0], ms=4, label="Parallel up")
ax.plot(betas, nii_down, "o-", color=COLORS[1], ms=4, label="Parallel down, client rates ≥ 0")
ax.plot(betas, nii_down_nofloor, "o--", color=COLORS[3], ms=4, label="Parallel down, no floor")
ax.axvline(assumptions["pass_through"]["savings_deposits"]["beta"], color="grey", ls=":", lw=1)
ax.axhline(0, color="grey", lw=0.6)
ax.set(xlabel="Deposit pass-through beta (sight and savings)", ylabel="ΔNII, 12 months (CHF m)",
       title="ΔNII depends on how fast deposit rates follow")
ax.legend(fontsize=8)
save(fig, "fig7_nmd_sensitivity")

# ---------------------------------------------------------------- 2. prepayment option on fixed mortgages
BASE_CPR = 0.02          # illustrative base prepayment rate (no public BCV figure)
incentives = [0.0, 0.05, 0.10, 0.20]
shifts_bp = np.arange(-300, 301, 25)
mortgage_value = {}
for incentive in incentives:
    pos = [p for p in build_positions(assumptions, mortgage_cpr=BASE_CPR, mortgage_incentive=incentive)
           if p.name == "mortgages_fixed"]
    values = []
    for s in shifts_bp:
        shift = (lambda t, s=s: s / 1e4 + 0 * np.asarray(t, dtype=float))
        values.append(economic_value(pos, curve, shift)["mortgages_fixed"])
    mortgage_value[incentive] = np.array(values)
prepay = {}
for incentive in incentives:
    positions = build_positions(assumptions, mortgage_cpr=BASE_CPR, mortgage_incentive=incentive)
    prepay[incentive] = {s: delta_eve(positions, curve, SHOCKS[s])[0] for s in ("parallel_up", "parallel_down", "short_down")}
no_prepay = {s: delta_eve(base_positions, curve, SHOCKS[s])[0] for s in ("parallel_up", "parallel_down", "short_down")}
out["prepayment"] = {"base_cpr": BASE_CPR, "incentives_cpr_per_100bp": incentives,
                     "delta_eve": {str(k): v for k, v in prepay.items()}, "no_prepayment": no_prepay,
                     "mortgage_value_vs_shift": {"shifts_bp": shifts_bp.tolist(),
                                                 **{str(k): v.tolist() for k, v in mortgage_value.items()}}}
print("Prepayment dEVE parallel down:", {k: round(v["parallel_down"]) for k, v in prepay.items()},
      "no prepayment:", round(no_prepay["parallel_down"]))

# value of the mortgage book with no prepayment = the penalty regime (prepayment paid at market value)
penalty_pos = [p for p in base_positions if p.name == "mortgages_fixed"]
penalty_value = np.array([economic_value(penalty_pos, curve, (lambda t, s=s: s / 1e4 + 0 * np.asarray(t, dtype=float)))
                          ["mortgages_fixed"] for s in shifts_bp])
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
v0 = penalty_value[shifts_bp == 0][0]
ax.plot(shifts_bp, (penalty_value - v0) / 1e3, color="black", lw=2, label="Penalty (BCV): no option value")
for i, incentive in enumerate(incentives):
    v_inc = mortgage_value[incentive]
    ax.plot(shifts_bp, (v_inc - v_inc[shifts_bp == 0][0]) / 1e3, color=COLORS[i], lw=1.5,
            label=f"No penalty, CPR {BASE_CPR:.0%}" + ("" if incentive == 0 else f" +{incentive:.0%} per 100 bp fall"))
ax.axhline(0, color="grey", lw=0.6)
ax.axvline(0, color="grey", lw=0.6)
ax.set(xlabel="Parallel shift (bp)", ylabel="Change in value (CHF bn)",
       title="Fixed-rate mortgages (CHF 34.2 bn): prepayment option")
ax.legend(fontsize=7)
ax = axes[1]
bars = [no_prepay["parallel_down"]] + [prepay[k]["parallel_down"] for k in incentives]
names = ["Penalty (BCV)", f"CPR {BASE_CPR:.0%} flat"] + [f"+{k:.0%}/100bp" for k in incentives[1:]]
x = np.arange(len(bars))
ax.bar(x, bars, color=[COLORS[0]] + [COLORS[2]] * len(incentives))
for xi, v in zip(x, bars):
    ax.text(xi, v + (15 if v > 0 else -60), f"{v:.0f}", ha="center", fontsize=8)
ax.axhline(0, color="grey", lw=0.6)
ax.set_xticks(x, names)
ax.set(ylabel="ΔEVE parallel down (CHF m)", title="Without a penalty, falling rates could cost BCV money")
save(fig, "fig8_prepayment")

# ---------------------------------------------------------------- 3. hedging with a payer swap
eve_now = {s: delta_eve(base_positions, curve, f)[0] for s, f in SHOCKS.items()}
no_swaps = build_positions(assumptions, include_swaps=False)
eve_noswap = {s: delta_eve(no_swaps, curve, f)[0] for s, f in SHOCKS.items()}
TENOR, TARGET = 5, 0.10
rate = float(curve.par_swap_rate(TENOR))
unit = payer_swap(1000.0, TENOR, rate)          # CHF 1 bn notional
unit_eve = {s: delta_eve(unit, curve, f)[0] for s, f in SHOCKS.items()}
notional = (-TARGET * TIER1 - eve_now["parallel_up"]) / unit_eve["parallel_up"] * 1000.0
hedge = payer_swap(notional, TENOR, rate)
hedged = base_positions + hedge
eve_hedged = {s: delta_eve(hedged, curve, f)[0] for s, f in SHOCKS.items()}
nii_now = {s: delta_nii(base_positions, SHOCKS[s])[0] for s in ("parallel_up", "parallel_down")}
nii_hedged = {s: delta_nii(hedged, SHOCKS[s])[0] for s in ("parallel_up", "parallel_down")}
draft_hedged = {s: delta_eve(hedged, curve, f)[0] for s, f in DRAFT.items()}
carry = notional * (float(curve.zero(1 / 365)) - rate)
out["hedge"] = {
    "swap_tenor_y": TENOR, "swap_rate": rate, "target_ratio": TARGET, "notional_chf_m": notional,
    "delta_eve_per_bn": unit_eve, "delta_eve_before": eve_now, "delta_eve_after": eve_hedged,
    "delta_eve_without_existing_swaps": eve_noswap,
    "worst_before": outlier_ratio(eve_now, TIER1), "worst_after": outlier_ratio(eve_hedged, TIER1),
    "worst_without_swaps": outlier_ratio(eve_noswap, TIER1),
    "worst_after_draft_2028": outlier_ratio(draft_hedged, TIER1),
    "delta_nii_before": nii_now, "delta_nii_after": nii_hedged, "annual_carry_at_current_rates": carry,
}
print(f"Without existing swaps: worst {outlier_ratio(eve_noswap, TIER1)}")
print(f"Hedge: pay fixed {rate:.3%} for {TENOR}y on CHF {notional:,.0f} m -> worst {outlier_ratio(eve_hedged, TIER1)}; "
      f"dNII up {nii_now['parallel_up']:.0f}->{nii_hedged['parallel_up']:.0f}, down {nii_now['parallel_down']:.0f}->"
      f"{nii_hedged['parallel_down']:.0f}; carry {carry:.1f} m/yr")

fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(shocks.SCENARIOS))
for k, (label, values, color) in enumerate([
        ("Without BCV's existing swaps", eve_noswap, COLORS[4]),
        ("Model as at 31.12.2025", eve_now, COLORS[0]),
        (f"+ CHF {notional / 1e3:.1f} bn {TENOR}y payer swap", eve_hedged, COLORS[2])]):
    vals = [values[s] / TIER1 * 100 for s in shocks.SCENARIOS]
    ax.bar(x + (k - 1) * 0.27, vals, 0.27, color=color, label=label)
ax.axhline(-15, color=COLORS[1], ls="--", lw=1.2, label="−15% outlier threshold")
ax.axhline(0, color="grey", lw=0.6)
ax.set_xticks(x, [shocks.LABELS[s] for s in shocks.SCENARIOS])
ax.set(ylabel="ΔEVE / Tier 1 (%)", title="Swaps reduce the exposure to rising rates")
ax.legend(fontsize=8)
save(fig, "fig9_hedge")

(RESULTS / "sensitivity_results.json").write_text(json.dumps(out, indent=2, default=float))
