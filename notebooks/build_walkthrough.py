"""Builds notebooks/walkthrough.ipynb (short cells that tell the story of the project).

Run:  python notebooks/build_walkthrough.py
then: jupyter nbconvert --to notebook --execute --inplace notebooks/walkthrough.ipynb
"""

from pathlib import Path

import nbformat as nbf

cells = []
md = lambda text: cells.append(nbf.v4.new_markdown_cell(text))  # noqa: E731
code = lambda text: cells.append(nbf.v4.new_code_cell(text))  # noqa: E731

md("""# Interest rate risk in the banking book of a Swiss cantonal bank (BCV)

A short walkthrough. The full analysis is in `scripts/`, the report in `report/`.
Every number comes from public documents saved in `data/` (BCV Pillar 3 and annual report 2025,
FINMA Circular 2019/2, SNB data portal).""")

code("""import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import pandas as pd
from src import data, shocks
from src.bcv import build_positions
from src.curve import ZeroCurve
from src.model import delta_eve, delta_nii""")

md("""## 1. What BCV publishes
The Pillar 3 report gives, for the parent bank at 31 Dec 2025, the change in economic value of
equity (ΔEVE) and in net interest income (ΔNII) under the six FINMA shocks (table IRRBB1).""")

code("""published = data.bcv_published_irrbb1()
latest = published[published.date == "2025-12-31"].set_index("scenario")
tier1 = data.key_figures()["tier1_parent"]["value"]
latest[["delta_eve_chf_m", "delta_nii_chf_m"]]""")

md("""## 2. The balance sheet behind it
Table IRRBBA1 gives the volume and the average repricing date of each block of the banking book.""")

code("""positions_table = data.bcv_positions()
positions_table[["item", "volume_chf_m", "avg_repricing_y", "max_repricing_y"]]""")

md("""## 3. The CHF curve and the six shocks
Svensson curve fitted to 22 Confederation bonds and SARON on 30 Dec 2025 (`scripts/02_curve_and_inputs.py`).
FINMA sizes for CHF: parallel 150 bp, short 150 bp, long 100 bp, no floor.""")

code("""curve_info = json.loads((ROOT / "results" / "curve.json").read_text())
curve = ZeroCurve(**curve_info["svensson"])
print(f"fit error {curve_info['rmse_bp']:.1f} bp; 5y zero rate {curve.zero(5):.2%}, 10y {curve.zero(10):.2%}")
pd.DataFrame(shocks.bucket_table(), index=[b[0] for b in shocks.BUCKETS]).T""")

md("""## 4. Deposits: how much do client rates follow SARON?
Pass-through betas estimated on SNB client rates, June 2022 - December 2025.""")

code("""assumptions = json.loads((ROOT / "results" / "assumptions.json").read_text())
pd.DataFrame(assumptions["pass_through"]).T[["beta", "r2", "months"]]""")

md("""## 5. ΔEVE and ΔNII of the model vs BCV""")

code("""positions = build_positions(assumptions)
rows = {}
for name, shock in shocks.all_shocks().items():
    rows[name] = {"model ΔEVE": delta_eve(positions, curve, shock)[0],
                  "BCV ΔEVE": latest.loc[name, "delta_eve_chf_m"],
                  "model ΔNII": delta_nii(positions, shock)[0],
                  "BCV ΔNII": latest.loc[name, "delta_nii_chf_m"]}
comparison = pd.DataFrame(rows).T.round(0)
comparison""")

md("""## 6. Outlier test: worst ΔEVE / Tier 1 against 15%""")

code("""worst = comparison["model ΔEVE"].idxmin()
print(f"model:     {-comparison.loc[worst, 'model ΔEVE'] / tier1:.1%} ({worst})")
print(f"published: {-latest.delta_eve_chf_m.min() / tier1:.1%}")""")

md("""## 7. Sensitivity, prepayment and hedge
See `scripts/04_nmd_prepayment_hedge.py`. Key numbers:""")

code("""sens = json.loads((ROOT / "results" / "sensitivity_results.json").read_text())
hedge = sens["hedge"]
print("ΔEVE parallel up if the deposit stable share is halved / x1.5:")
for k, v in sens["nmd_one_at_a_time_parallel_up"].items():
    print(f"  {k:40s} {v:7.0f}")
print(f"worst ratio without BCV's swaps: {hedge['worst_without_swaps'][1]:.1%}")
print(f"extra 5y payer swap to reach 10%: CHF {hedge['notional_chf_m']:,.0f} m at {hedge['swap_rate']:.2%}")""")

notebook = nbf.v4.new_notebook()
notebook["cells"] = cells
notebook["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
target = Path(__file__).with_name("walkthrough.ipynb")
nbf.write(notebook, target)
print(f"wrote {target}")
