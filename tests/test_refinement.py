import numpy as np
import pandas as pd
from oil_research.refinement import inventory_innovations, completed_error_rms, sized


def test_inventory_forecasts_ignore_current_and_future_observations():
    rng = np.random.default_rng(4)
    dates = pd.date_range("2011-01-07", periods=160, freq="7D")
    d = pd.DataFrame(
        {
            "observation_date": dates,
            "available_at": (dates + pd.Timedelta(days=6)).tz_localize("UTC"),
        }
    )
    for c in [
        "crude_change",
        "gasoline_change",
        "distillate_change",
        "imports",
        "exports",
        "refinery_inputs",
    ]:
        d[c] = rng.normal(size=len(d))
    _, a = inventory_innovations(d)
    changed = d.copy()
    changed.loc[150:, "crude_change"] = 1000
    _, b = inventory_innovations(changed)
    aa = a[a.series == "crude_change"].reset_index(drop=True)
    bb = b[b.series == "crude_change"].reset_index(drop=True)
    cutoff = str(dates[150].date())
    np.testing.assert_allclose(
        aa.loc[aa.observation_date <= cutoff, "prediction"],
        bb.loc[bb.observation_date <= cutoff, "prediction"],
    )
    assert (
        pd.to_datetime(a.last_training_release, utc=True) < pd.to_datetime(a.available_at, utc=True)
    ).all()


def test_uncertainty_waits_for_label_completion():
    i = pd.bdate_range("2020-01-01", periods=360)
    p = pd.Series(np.linspace(-0.3, 0.4, len(i)), index=i)
    m = pd.DataFrame(
        {
            "target_21": 0.01,
            "cash_yield_annual": 0.02,
            "annual_vol": 0.3,
            "label_end_21": i + pd.offsets.BDay(22),
        },
        index=i,
    )
    a = completed_error_rms(p, m)
    m.loc[i[200] :, "target_21"] = 5
    b = completed_error_rms(p, m)
    pd.testing.assert_series_equal(a.iloc[:222], b.iloc[:222])
    assert a.iloc[:147].isna().all()


def test_uncertainty_risk_rule_never_increases_exposure():
    i = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(2)
    p = pd.Series(rng.normal(0, 0.4, 300), index=i)
    m = pd.DataFrame({"target_return": rng.normal(0, 0.02, 300)}, index=i)
    u = pd.Series(0.8, index=i)
    baseline = sized(p, m, u, "existing")
    reduced = sized(p, m, u, "conservative")
    assert (reduced.abs() <= baseline.abs() + 1e-12).all()
