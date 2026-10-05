import numpy as np
import pandas as pd
import pytest

from oil_research.extension_sources import (
    validate_vintage_csv,
    parse_archive_headlines,
    eia_releases,
)
from oil_research.extension_features import join_releases, news_features, cutoffs
from oil_research.extension_experiments import simulate_cash, predict_candidate
from oil_research.backtest import sharpe
from oil_research.extension_innovations import innovation_events


def test_every_alfred_series_must_have_requested_vintage():
    bad = b"observation_date,DGS3MO_20240101,DGS10_20261005\n2023-12-29,5.4,4.0\n"
    with pytest.raises(ValueError, match="vintage mismatch"):
        validate_vintage_csv(bad, ["DGS3MO", "DGS10"], "2024-01-01")
    good = bad.replace(b"DGS10_20261005", b"DGS10_20240101")
    assert list(validate_vintage_csv(good, ["DGS3MO", "DGS10"], "2024-01-01")) == [
        "DGS3MO",
        "DGS10",
    ]


def test_revision_only_appears_after_its_release():
    dates = pd.bdate_range("2024-01-01", periods=5)
    events = pd.DataFrame(
        {
            "series": ["opec", "opec"],
            "role": ["estimate", "estimate"],
            "value": [30.0, 99.0],
            "available_at": ["2024-01-02T00:00:00Z", "2024-01-04T23:00:00Z"],
        }
    )
    values, lineage = join_releases(events, dates, "test")
    assert np.isnan(values.iloc[0].test_opec_estimate)
    assert values.loc["2024-01-04", "test_opec_estimate"] == 30
    assert values.loc["2024-01-05", "test_opec_estimate"] == 99
    changed = events.copy()
    changed.loc[1, "value"] = -1000
    other, _ = join_releases(changed, dates, "test")
    pd.testing.assert_frame_equal(values.loc[:"2024-01-04"], other.loc[:"2024-01-04"])


def test_stale_macro_is_missing_not_fresh():
    e = pd.DataFrame(
        {
            "series": ["opec"],
            "role": ["estimate"],
            "value": [30.0],
            "available_at": ["2020-01-01T00:00:00Z"],
        }
    )
    v, _ = join_releases(e, pd.DatetimeIndex(["2021-01-01"]), "x")
    assert v.x_opec_estimate.isna().all()


def test_corrected_eia_workbook_is_quarantined():
    source = """<table><tr><td>January 2024 1/9/2024</td><td><a href="archives/jan24_base.xlsx">Excel</a></td></tr>
    <tr><td>February 2024 2/6/2024 Notice for release</td><td><a href="archives/feb24_base.xlsx">Excel</a></td></tr></table>"""
    allowed, excluded = eia_releases(source, 2024, 2024)
    assert len(allowed) == len(excluded) == 1
    assert allowed[0]["release_date"] == "2024-01-09"


def test_headline_availability_is_capture_plus_delay():
    html = '<a href="https://oilprice.com/Latest-Energy-News/World-News/Oil-Rises.html">Oil prices rise on supply cuts</a>'
    captured = pd.Timestamp("2024-01-02T23:00:00Z")
    rows = parse_archive_headlines(html, captured)
    assert len(rows) == 1
    assert pd.Timestamp(rows[0]["available_at"]) == captured + pd.Timedelta(days=1)
    h = pd.DataFrame(rows)
    cov = pd.DataFrame({"available_at": [rows[0]["available_at"]]})
    x = news_features(h, cov, pd.bdate_range("2024-01-02", periods=4))
    assert (
        x.loc["2024-01-03", "news_captures_7"] == 0
    )  # 17 NY = 22 UTC, before capture availability
    assert np.isnan(x.loc["2024-01-03", "news_titles_per_capture_7"])
    assert x.loc["2024-01-04", "news_titles_per_capture_7"] == 1


def test_signal_clock_handles_daylight_saving():
    clock = cutoffs(pd.DatetimeIndex(["2024-01-05", "2024-07-05"]))
    assert clock[0].hour == 22 and clock[1].hour == 21


def test_cash_does_not_artificially_create_excess_alpha():
    dates = pd.bdate_range("2024-01-01", periods=5)
    asset = pd.Series(0.01, index=dates)
    cash = pd.Series(0.0002, index=dates)
    ledger = simulate_cash(asset, cash, pd.Series(0.0, index=dates), 5)
    np.testing.assert_allclose(ledger.net_return, cash, atol=1e-15)
    np.testing.assert_allclose(ledger.excess_return, 0, atol=1e-15)
    assert sharpe(ledger.excess_return) == 0


def test_long_horizon_training_outcomes_are_purged(monkeypatch):
    import oil_research.extension_experiments as experiments

    dates = pd.bdate_range("2010-01-01", periods=2100)
    x = pd.DataFrame({"a": np.arange(len(dates))}, index=dates)
    clock = pd.Series(dates, index=dates)
    meta = pd.DataFrame(
        {
            "target_21": 0.02,
            "cash_yield_annual": 0.02,
            "annual_vol": 0.2,
            "label_end": clock.shift(-2),
            "label_end_21": clock.shift(-22),
            "cash_return": 0.0001,
        },
        index=dates,
    )

    class Fake:
        def fit(self, x, y):
            return self

        def predict(self, x):
            return np.zeros(len(x))

    monkeypatch.setattr(experiments, "model", lambda *a: Fake())
    cfg = {
        "development_start": "2017-01-01",
        "selection_end": "2018-01-01",
        "minimum_train_rows": 900,
        "seed": 42,
    }
    _, audit = predict_candidate(
        x, meta, {"horizon": 21, "window": "expanding", "model": "ridge"}, cfg
    )
    assert (pd.to_datetime(audit.last_training_outcome) < pd.to_datetime(audit.test_start)).all()


def test_surprise_compares_same_observation_month_to_earlier_release():
    events = pd.DataFrame(
        [
            {
                "series": "opec",
                "role": "current_month_forecast",
                "observation_period": "2024-01",
                "value": 30.0,
                "available_at": "2024-01-10T00:00:00Z",
            },
            {
                "series": "opec",
                "role": "last_month_estimate",
                "observation_period": "2024-01",
                "value": 29.0,
                "available_at": "2024-02-10T00:00:00Z",
            },
            {
                "series": "opec",
                "role": "previous_month_estimate",
                "observation_period": "2024-01",
                "value": 40.0,
                "available_at": "2024-03-10T00:00:00Z",
            },
        ]
    )
    revisions = innovation_events(events)
    assert len(revisions) == 1
    assert revisions.iloc[0].value == -1
    assert revisions.iloc[0].available_at.startswith("2024-02-10")
