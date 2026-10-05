"""Release surprises compared with the latest earlier estimate for that same month."""

import numpy as np
import pandas as pd

from .extension_features import join_releases


def innovation_events(events):
    events = events.copy()
    events["available_at"] = pd.to_datetime(events.available_at, utc=True)
    result = []
    for series, group in events.groupby("series"):
        known = {}
        for available, release in group.sort_values("available_at").groupby("available_at"):
            # Compute against the previous release, before updating any value from this release.
            for row in release.itertuples():
                if row.observation_period in known:
                    previous, previous_available = known[row.observation_period]
                    assert previous_available < available
                    if row.role in [
                        "last_month_estimate",
                        "current_month_forecast",
                        "three_month_forecast",
                    ]:
                        result.append(
                            {
                                "series": series,
                                "role": row.role + "_revision",
                                "value": row.value - previous,
                                "available_at": available.isoformat(),
                                "compared_available_at": previous_available.isoformat(),
                                "observation_period": row.observation_period,
                            }
                        )
            for row in release.itertuples():
                known[row.observation_period] = (row.value, available)
    return pd.DataFrame(result)


def add_innovations(x, events):
    revisions = innovation_events(events)
    values, lineage = join_releases(revisions, x.index, "innovation")
    # Only recently released revisions have immediate-event weight; older values decay.
    for c in list(values):
        if not c.endswith("_age_days"):
            values[c] = values[c] * np.exp(-values[c + "_age_days"] / 10)
    keep = [c for c in values if not c.endswith("_age_days")]
    features = x.join(values[keep])
    controls = [
        c
        for c in x
        if c.startswith("USO_")
        or c
        in [
            "curve_slope_10y_2y",
            "curve_slope_10y_3m",
            "extra_^VIX_level",
            "extra_^OVX_level",
            "UUP_return_21",
        ]
    ]
    return features, controls + keep, lineage, revisions
