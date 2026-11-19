"""The six standardised interest-rate shocks of Basel (SRP31.90-93) / FINMA Circ. 2019/2, Annex 2.

For CHF (FINMA, in force at 31.12.2025): parallel 150 bp, short 150 bp, long 100 bp, x = 4 years,
no post-shock floor. Every function returns the shock in decimals as a function of maturity t.
"""

import numpy as np

SCENARIOS = ["parallel_up", "parallel_down", "steepener", "flattener", "short_up", "short_down"]

LABELS = {
    "parallel_up": "Parallel up",
    "parallel_down": "Parallel down",
    "steepener": "Steepener",
    "flattener": "Flattener",
    "short_up": "Short up",
    "short_down": "Short down",
}

# FINMA time buckets (lower bound, upper bound, midpoint in years), Annex 2
BUCKETS = [
    ("O/N", 0.0, 1 / 365, 0.0028), ("1M", 1 / 365, 1 / 12, 0.0417), ("3M", 1 / 12, 0.25, 0.1667),
    ("6M", 0.25, 0.5, 0.375), ("9M", 0.5, 0.75, 0.625), ("1Y", 0.75, 1.0, 0.875),
    ("1.5Y", 1.0, 1.5, 1.25), ("2Y", 1.5, 2.0, 1.75), ("3Y", 2.0, 3.0, 2.5), ("4Y", 3.0, 4.0, 3.5),
    ("5Y", 4.0, 5.0, 4.5), ("6Y", 5.0, 6.0, 5.5), ("7Y", 6.0, 7.0, 6.5), ("8Y", 7.0, 8.0, 7.5),
    ("9Y", 8.0, 9.0, 8.5), ("10Y", 9.0, 10.0, 9.5), ("15Y", 10.0, 15.0, 12.5),
    ("20Y", 15.0, 20.0, 17.5), (">20Y", 20.0, np.inf, 25.0),
]


def shock_function(scenario: str, parallel_bp=150.0, short_bp=150.0, long_bp=100.0, x=4.0):
    """Return t -> shock (decimal) for one of the six scenarios."""
    parallel, short, long_ = parallel_bp / 1e4, short_bp / 1e4, long_bp / 1e4

    def short_shape(t):
        return short * np.exp(-np.asarray(t, dtype=float) / x)

    def long_shape(t):
        return long_ * (1 - np.exp(-np.asarray(t, dtype=float) / x))

    functions = {
        "parallel_up": lambda t: parallel + 0 * np.asarray(t, dtype=float),
        "parallel_down": lambda t: -parallel + 0 * np.asarray(t, dtype=float),
        "short_up": short_shape,
        "short_down": lambda t: -short_shape(t),
        "steepener": lambda t: -0.65 * short_shape(t) + 0.9 * long_shape(t),
        "flattener": lambda t: 0.8 * short_shape(t) - 0.6 * long_shape(t),
    }
    if scenario not in functions:
        raise ValueError(f"unknown scenario {scenario!r}")
    return functions[scenario]


def all_shocks(**sizes):
    return {name: shock_function(name, **sizes) for name in SCENARIOS}


def bucket_table(**sizes):
    """Shocks (in bp, rounded) at the FINMA bucket midpoints, as printed in Annex 2."""
    mids = np.array([b[3] for b in BUCKETS])
    return {name: np.round(f(mids) * 1e4).astype(int) for name, f in all_shocks(**sizes).items()}
