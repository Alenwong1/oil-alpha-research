import numpy as np
import pandas as pd
import pytest
from oil_research.scarcity import states, parse_products, completed_rms


def test_release_product_parser_checks_dates_and_units():
    raw = b'STUB_1,01/03/25,x,x\nProducts Supplied ,(31) Finished Motor Gasoline,"8,481",0\nProducts Supplied ,(33) Distillate Fuel Oil,"3,178",0\n'
    assert parse_products(raw, "2025-01-03") == {
        "gasoline_supplied": 8481.0,
        "distillate_supplied": 3178.0,
    }
    with pytest.raises(ValueError):
        parse_products(raw, "2025-01-10")


def test_scarcity_states_are_causal_and_gaps_disable_coverage():
    rng = np.random.default_rng(12)
    dates = pd.date_range("2010-01-01", periods=220, freq="7D")
    d = pd.DataFrame(
        {
            "observation_date": dates,
            "available_at": (dates + pd.Timedelta(days=6)).tz_localize("UTC"),
        }
    )
    for c in [
        "crude_stocks",
        "gasoline_stocks",
        "distillate_stocks",
        "gasoline_supplied",
        "distillate_supplied",
    ]:
        d[c] = 100 + rng.normal(0, 10, len(d))
    a = states(d)
    b = d.copy()
    b.loc[190:, ["crude_stocks", "gasoline_supplied"]] = 10000
    pd.testing.assert_frame_equal(a.iloc[:190], states(b).iloc[:190])
    assert a.loc[190, "scarcity_crude"] == states(b).loc[190, "scarcity_crude"]
    g = states(d.drop(index=180))
    assert pd.isna(g.loc[180, "scarcity_crude"])
    assert g.loc[180:182, "demand_gasoline"].isna().all()


@pytest.mark.parametrize("h", [5, 21])
def test_horizon_error_rms_waits_for_completion(h):
    i = pd.bdate_range("2020-01-01", periods=360)
    p = pd.Series(np.linspace(-0.3, 0.4, len(i)), index=i)
    m = pd.DataFrame(
        {
            f"target_{h}": 0.01,
            "cash_yield_annual": 0.02,
            "annual_vol": 0.3,
            f"label_end_{h}": i + pd.offsets.BDay(h + 1),
        },
        index=i,
    )
    a = completed_rms(p, m, h)
    m.loc[i[200] :, f"target_{h}"] = 5
    b = completed_rms(p, m, h)
    pd.testing.assert_series_equal(a.iloc[: 200 + h + 1], b.iloc[: 200 + h + 1])
