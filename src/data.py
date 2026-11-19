"""Loading the saved public data (SNB data portal CSVs and the transcribed BCV tables)."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def read_snb_cube(cube_id: str) -> pd.DataFrame:
    """Read an SNB cube saved by scripts/01_download_data.py (long format, ';' separated).

    The first three lines are metadata (cube id, publishing date, blank line).
    Returns columns Date, the dimension columns, Value (float, NaN when missing).
    """
    frame = pd.read_csv(RAW / "snb" / f"{cube_id}.csv", sep=";", skiprows=3)
    frame["Value"] = pd.to_numeric(frame["Value"], errors="coerce")
    return frame


def snb_series(cube_id: str, item: str, column: str = "D0", reference: str | None = None) -> pd.Series:
    """One series of an SNB cube, indexed by date, in percent."""
    frame = read_snb_cube(cube_id)
    if reference is not None:
        frame = frame[frame["D0"] == reference]
    frame = frame[frame[column] == item]
    series = frame.set_index("Date")["Value"].dropna()
    series.index = pd.to_datetime(series.index)
    return series.sort_index()


def confederation_bond_yields(date: str) -> pd.DataFrame:
    """Yield to maturity (%) and residual maturity (years) of every Confederation bond on a date
    (SNB cube rendoeid)."""
    frame = read_snb_cube("rendoeid")
    frame = frame[frame["Date"] == date]
    table = frame.pivot(index="d0", columns="d1", values="Value").dropna()
    table = table.rename(columns={"R1": "ytm_pct", "LZ": "maturity_y"})
    return table.sort_values("maturity_y")


def saron_fixing(date: str) -> float:
    """SARON fixing at the close of a trading day, in percent (SNB cube snbgwdzid)."""
    series = snb_series("snbgwdzid", "SARON")
    return float(series.loc[date])


def bcv_positions() -> pd.DataFrame:
    """BCV parent-company repricing table IRRBBA1 at 31.12.2025 (Pillar 3, Table 26, p.53-54)."""
    return pd.read_csv(PROCESSED / "bcv_irrbba1_2025-12-31.csv")


def bcv_published_irrbb1() -> pd.DataFrame:
    """Published Delta EVE / Delta NII (IRRBB1) at four reporting dates."""
    return pd.read_csv(PROCESSED / "bcv_irrbb1_published.csv")


def key_figures() -> dict:
    return json.loads((PROCESSED / "bcv_key_figures.json").read_text(encoding="utf-8"))


def finma_shocks() -> dict:
    return json.loads((PROCESSED / "finma_shocks.json").read_text(encoding="utf-8"))
