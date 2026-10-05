import numpy as np
import pandas as pd
import pytest
from oil_research.weekly_sources import parse_eia, excluded_cftc
from oil_research.weekly_features import seasonal_features, asof_features, position_features
from oil_research.weekly_experiments import simulate_scheduled, gate_targets, risk_scaled
from oil_research.extension_leverage import simulate_signed


def test_archive_parser_reads_current_and_previous_vintage_columns():
    b = b'"STUB_1","1/3/25","12/27/24"\n"Commercial (Excluding SPR)","410","415"\n"Total Motor Gasoline","230","225"\n"Distillate Fuel Oil","120","119"\n'
    d = parse_eia(b, "2025-01-08", 1)
    assert d["crude_change"] == -5
    assert pd.Timestamp(d["available_at"]) == pd.Timestamp("2025-01-09T05:00:00Z")
    with pytest.raises(ValueError):
        parse_eia(b, "2025-01-02", 1)


def test_cftc_outages_are_quarantined():
    for d in ["2013-10-08", "2019-01-08", "2023-02-07", "2025-10-07"]:
        assert excluded_cftc(pd.Timestamp(d))
    assert excluded_cftc(pd.Timestamp("2024-02-06")) is None


def test_asof_joins_do_not_backfill_and_expire_stale_reports():
    e = pd.DataFrame(
        {
            "available_at": pd.to_datetime(["2024-01-12T05:00Z"], utc=True),
            "observation_date": pd.to_datetime(["2024-01-02"]),
            "value": [3.0],
        }
    )
    x, l = asof_features(e, pd.to_datetime(["2024-01-11", "2024-01-12", "2024-02-01"]))
    assert np.isnan(x.value.iloc[0])
    assert x.value.iloc[1] == 3
    assert np.isnan(x.value.iloc[2])


def test_seasonal_z_ignores_future_releases():
    dates = pd.date_range("2010-01-01", periods=200, freq="7D")
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
        "refinery_inputs",
        "utilization",
        "imports",
        "exports",
    ]:
        d[c] = np.sin(np.arange(200) / 8) + np.arange(200) * 0.01
    first = seasonal_features(d)
    changed = d.copy()
    changed.loc[160:, "crude_change"] = 100000
    second = seasonal_features(changed)
    pd.testing.assert_frame_equal(first.iloc[:160], second.iloc[:160])


def inputs():
    i = pd.bdate_range("2024-01-01", periods=3)
    return (
        pd.Series([0.1, -0.02, 0.03], i),
        pd.Series(0.0, i),
        pd.Series(1 / 365, i),
        pd.Series(0.5, i),
    )


def test_weekly_hold_preserves_shares_instead_of_daily_rebalance():
    r, c, d, w = inputs()
    mask = pd.Series([True, False, False], index=r.index)
    l = simulate_scheduled(r, c, d, w, mask, cost_bps=0)
    assert l.weight.iloc[1] == pytest.approx(0.5 * 1.1 / 1.05)
    assert l.turnover.iloc[1] == 0
    assert l.equity.iloc[-1] == pytest.approx(0.5 + 0.5 * 1.1 * 0.98 * 1.03)


def test_daily_schedule_matches_signed_accounting():
    r, c, d, w = inputs()
    w.iloc[1] = -1.0
    w.iloc[2] = 1.5
    a = simulate_scheduled(r, c, d, w, pd.Series(True, r.index))
    b = simulate_signed(r, c, d, w)
    np.testing.assert_allclose(a.net_return, b.net_return, atol=1e-12)


def test_cost_gate_requires_more_edge_for_shorts():
    i = pd.RangeIndex(2)
    w = pd.Series([1.0, -1.0], i)
    edge = pd.Series([0.002, -0.002], i)
    out = gate_targets(w, edge, pd.Series(0.04, i))
    assert out.iloc[0] == 1
    assert out.iloc[1] == 0


def test_risk_scaling_cannot_use_unfinished_returns():
    i = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(1)
    p = pd.DataFrame(
        rng.normal(0, 0.4, (300, 3)), index=i, columns=["trend", "inventory", "positioning"]
    )
    m = pd.DataFrame({"target_return": rng.normal(0, 0.02, 300), "annual_vol": 0.3}, index=i)
    a, _, _ = risk_scaled(p, m)
    m.loc[i[200] :, "target_return"] = 100
    b, _, _ = risk_scaled(p, m)
    pd.testing.assert_frame_equal(a.iloc[:202], b.iloc[:202])


def test_four_digit_archive_year_is_supported():
    b = b'"STUB_1","12/19/2025","12/12/2025"\n"Commercial (Excluding SPR)","410","415"\n"Total Motor Gasoline","230","225"\n"Distillate Fuel Oil","120","119"\n'
    assert parse_eia(b, "2025-12-29", 1)["observation_date"] == "2025-12-19"
