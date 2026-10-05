import numpy as np
import pandas as pd
from oil_research.time_series import (
    BASE,
    SEQUENCE,
    lag_events,
    features_from_saved,
    recency_weights,
    predict,
)


def event_fixture(n=30):
    obs = pd.date_range("2013-01-04", periods=n, freq="7D")
    d = pd.DataFrame(
        {"observation_date": obs, "available_at": (obs + pd.Timedelta(days=6)).tz_localize("UTC")}
    )
    for c in BASE + SEQUENCE[3:]:
        d[c] = np.arange(n, dtype=float)
    return d


def test_weekly_lags_ignore_future_and_reject_missing_weeks():
    d = event_fixture()
    a = lag_events(d)
    changed = d.copy()
    changed.loc[20:, SEQUENCE] = 999
    pd.testing.assert_frame_equal(a.iloc[:20], lag_events(changed).iloc[:20])
    assert a.loc[20, "innovation_crude_change_lag8"] == 12
    missing = lag_events(d.drop(index=15))
    assert pd.isna(missing.loc[15, "innovation_crude_change_lag1"])
    assert pd.isna(missing.loc[20, "innovation_crude_change_lag8"])


def test_daily_copies_are_not_weekly_lags_and_nulls_not_backfilled():
    ix = pd.bdate_range("2014-01-06", periods=20)
    release = pd.DatetimeIndex(
        [pd.Timestamp("2014-01-06", tz="UTC")] * 10 + [pd.Timestamp("2014-01-20", tz="UTC")] * 10
    )
    line = pd.DataFrame(
        {
            "cutoff": ix.tz_localize("UTC") + pd.Timedelta(hours=22),
            "available_at": release,
            "observation_date": release.tz_localize(None) - pd.Timedelta(days=4),
        },
        index=ix,
    )
    f = pd.DataFrame({c: np.arange(20, dtype=float) for c in BASE + SEQUENCE[3:]}, index=ix)
    f.loc[ix[0], BASE[0]] = np.nan
    _, _, events = features_from_saved(f, line)
    assert len(events) == 2 and pd.isna(events.loc[0, BASE[0]])
    assert events.loc[1, BASE[0]] == 10
    assert events.loc[0, "available_at"] == line.iloc[0].cutoff
    assert pd.isna(events.loc[1, "innovation_crude_change_lag1"])


def test_recency_half_life_and_normalization():
    ix = pd.DatetimeIndex(["2020-01-01", "2022-01-01"])
    w = recency_weights(ix, "2023-01-01")
    assert abs(w.mean() - 1) < 1e-12
    assert abs(w[1] / w[0] - 2) < 0.002


def test_quarterly_prediction_cannot_use_unfinished_outcomes():
    ix = pd.bdate_range("2013-01-01", "2015-03-31")
    rng = np.random.default_rng(49)
    x = pd.DataFrame(rng.normal(size=(len(ix), 3)), index=ix)
    m = pd.DataFrame(
        {
            "target_5": rng.normal(0, 0.02, len(ix)),
            "annual_vol": 0.3,
            "cash_yield_annual": 0.02,
            "cash_return": 0.00001,
            "label_end": ix + pd.offsets.BDay(2),
            "label_end_5": ix + pd.offsets.BDay(6),
        },
        index=ix,
    )
    for kind in ["ridge", "adaptive", "boosting"]:
        p, a = predict(x, m, 5, kind)
        changed = m.copy()
        changed.loc[changed.label_end_5 >= pd.Timestamp("2015-01-01"), "target_5"] = 100
        q, _ = predict(x, changed, 5, kind)
        pd.testing.assert_series_equal(p, q)
        assert (pd.to_datetime(a.last_training_outcome) < pd.to_datetime(a.test_start)).all()
