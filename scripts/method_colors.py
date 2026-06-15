"""Shared colorblind-friendly palette for method comparison figures.

Paul Tol bright scheme — distinct hues, at most one green (#228833 for adaptive).
"""

from __future__ import annotations

# style_key -> hex color
METHOD_COLORS: dict[str, str] = {
    "galore2": "#4477AA",       # blue
    "baseline": "#4477AA",
    "lotus": "#EE6677",           # red/coral
    "baseline2": "#EE6677",
    "adaptive_stochastic": "#228833",  # green (only green in palette)
    "stochastic": "#CCBB44",      # yellow
    "stochastic_old": "#AA3377",  # purple
    "stoch_lotus": "#E69F00",     # orange
    "stoch_galore2": "#66CCEE",   # cyan
    "fisher": "#CC3311",          # vermillion
    "adammini": "#332288",        # indigo (dense baseline)
    "adam8bit": "#BBBBBB",        # gray
}

# Paul Tol bright — for spectrum / fisher boxplot groups (cycled)
GROUP_PALETTE: list[str] = [
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#AA3377",
    "#66CCEE",
    "#E69F00",
    "#CC3311",
]

LINEWIDTH: dict[str, float] = {
    "baseline": 2.4,
    "baseline2": 2.4,
    "galore2": 2.4,
    "lotus": 2.4,
    "adaptive_stochastic": 2.6,
    "stochastic": 2.6,
    "stochastic_old": 2.2,
    "stoch_lotus": 2.4,
    "stoch_galore2": 2.4,
    "fisher": 2.2,
    "adammini": 2.0,
    "adam8bit": 2.0,
}


def style_for(key: str) -> dict[str, float | str]:
    """Return matplotlib plot kwargs: color and linewidth."""
    return {
        "color": METHOD_COLORS[key],
        "lw": LINEWIDTH.get(key, 2.4),
    }
