import numpy as np
import pandas as pd
import pytest

from oil_research.features import build_features, dataset, rsi
from oil_research.backtest import simulate, metrics, paired_bootstrap
from oil_research.data import validate_bars
from oil_research.models import predict_period


def synthetic_bars(n=1100):
    rng = np.random.default_rng(41)
    dates = pd.bdate_range("2019-01-01", periods=n)
    bars = {}
    for ticker in ["USO", "SPY"]:
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        o = c * np.exp(rng.normal(0, 0.003, n))
        bars[ticker] = pd.DataFrame(
            {
                "Open": o,
                "Close": c,
                "High": np.maximum(c, o) * 1.01,
                "Low": np.minimum(c, o) * 0.99,
                "Volume": rng.integers(1000, 10000, n),
            },
            index=dates,
        )
    return bars


def test_future_prices_cannot_change_past_features():
    original = synthetic_bars()
    perturbed = {t: b.copy() for t, b in original.items()}
    cutoff = original["USO"].index[750]
    for b in perturbed.values():
        b.loc[b.index > cutoff, ["Open", "High", "Low", "Close"]] *= 3.7
        b.loc[b.index > cutoff, "Volume"] *= 4
    pd.testing.assert_frame_equal(
        build_features(original).loc[:cutoff], build_features(perturbed).loc[:cutoff]
    )


def test_target_is_next_open_to_following_open():
    bars = synthetic_bars()
    x, meta = dataset(bars)
    d = x.index[20]
    i = bars["USO"].index.get_loc(d)
    assert meta.loc[d, "target_return"] == pytest.approx(
        bars["USO"].Open.iloc[i + 2] / bars["USO"].Open.iloc[i + 1] - 1
    )
    assert meta.loc[d, "execution_date"] == bars["USO"].index[i + 1]
    assert meta.loc[d, "label_end"] == bars["USO"].index[i + 2]


def test_buy_hold_matches_adjusted_price_ratio():
    dates = pd.bdate_range("2020-01-01", periods=3)
    prices = np.array([100, 110, 99, 108.9])
    r = pd.Series(prices[1:] / prices[:-1] - 1, index=dates)
    ledger = simulate(r, pd.Series(1.0, index=dates), 0)
    assert ledger.equity.iloc[-1] == pytest.approx(prices[-1] / prices[0])
    assert ledger.turnover.iloc[1] == pytest.approx(0)


def test_entry_and_exit_fees_hand_calculated():
    dates = pd.bdate_range("2020-01-01", periods=1)
    ledger = simulate(pd.Series([0.1], index=dates), pd.Series([1.0], index=dates), 100)
    assert ledger.equity.iloc[-1] == pytest.approx((1 / 1.01) * 1.1 * 0.99)


def test_turnover_uses_drifted_weight():
    dates = pd.bdate_range("2020-01-01", periods=2)
    ledger = simulate(
        pd.Series([0.1, 0.0], index=dates), pd.Series([0.5, 0.5], index=dates), 0, liquidate=False
    )
    assert ledger.turnover.iloc[1] == pytest.approx(0.55 / 1.05 - 0.5)


def test_cash_has_no_fees_and_costs_reduce_wealth():
    dates = pd.bdate_range("2020-01-01", periods=4)
    r = pd.Series([0.1, -0.1, 0.05, 0], index=dates)
    w = pd.Series([1.0, 0.0, 0.5, 0.0], index=dates)
    cash = simulate(r, w * 0, 20)
    assert cash.equity.eq(1).all()
    assert cash.turnover.eq(0).all()
    assert simulate(r, w, 20).equity.iloc[-1] < simulate(r, w, 0).equity.iloc[-1]


def test_drawdown_includes_starting_capital():
    dates = pd.bdate_range("2020-01-01", periods=2)
    ledger = simulate(pd.Series([-0.2, 0.05], index=dates), pd.Series(1.0, index=dates), 0)
    assert metrics(ledger)["max_drawdown"] == pytest.approx(-0.2)


def test_bad_data_fails_loudly():
    b = synthetic_bars()["USO"]
    bad = pd.concat([b.iloc[:2], b.iloc[:2]])
    with pytest.raises(ValueError):
        validate_bars(bad, "USO")
    b.iloc[0, b.columns.get_loc("High")] = 0
    with pytest.raises(ValueError):
        validate_bars(b, "USO")


def test_flat_prices_have_neutral_rsi():
    assert rsi(pd.Series(np.ones(30))).iloc[-1] == 50


def test_block_bootstrap_of_identical_returns_is_zero():
    r = np.random.default_rng(1).normal(0, 0.01, 100)
    result = paired_bootstrap(r, r, repetitions=30)
    assert result == {"difference": 0.0, "lower_95": 0.0, "upper_95": 0.0}


def test_holdout_never_trains_on_holdout_labels(monkeypatch):
    import oil_research.models as models

    x, meta = dataset(synthetic_bars())
    cutoff = x.index[-100]
    captured = []

    class Constant:
        def predict(self, features):
            return np.full(len(features), 0.001)

    def spy(features, labels, kind, seed):
        captured.append((features.index.max(), labels.label_end.max()))
        return Constant(), {}

    monkeypatch.setattr(models, "fit_selected", spy)
    cfg = {
        "holdout_start": str(cutoff.date()),
        "data_end_exclusive": "2030-01-01",
        "target": "USO",
        "seed": 42,
    }
    first, audit = predict_period(x, meta, cfg, "holdout")
    assert len(captured) == 4  # One fit per variant, no holdout refitting.
    assert all(signal < cutoff and label < cutoff for signal, label in captured)
    changed = meta.copy()
    changed.loc[cutoff:, "target_return"] = 999.0
    second, _ = predict_period(x, changed, cfg, "holdout")
    pd.testing.assert_frame_equal(first, second)


def test_alignment_and_leverage_are_rejected():
    idx = pd.bdate_range("2020-01-01", periods=2)
    with pytest.raises(ValueError):
        simulate(pd.Series([0.1, 0.1], index=idx), pd.Series([1.2, 1.0], index=idx), 5)
    with pytest.raises(ValueError):
        simulate(pd.Series([0.1, 0.1], index=idx), pd.Series([1.0, 1.0], index=idx[::-1]), 5)
