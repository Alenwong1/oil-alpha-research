import numpy as np
import pandas as pd
import pytest
from oil_research.extension_leverage import simulate_signed


def inputs(r, w, cash=0.0):
    i = pd.bdate_range("2024-01-01", periods=len(r))
    return (
        pd.Series(r, index=i),
        pd.Series(cash, index=i),
        pd.Series(1 / 365, index=i),
        pd.Series(w, index=i),
    )


def test_short_profit_and_borrow_fee():
    ledger = simulate_signed(*inputs([-0.1], [-1.0]), cost_bps=0, borrow_annual=0.0365)
    assert ledger.net_return.iloc[0] == pytest.approx(0.1 - 0.0001)


def test_levered_long_financing():
    ledger = simulate_signed(
        *inputs([0.1], [2.0], cash=0.0002), cost_bps=0, financing_spread=0.0365
    )
    assert ledger.net_return.iloc[0] == pytest.approx(0.2 - 0.0003)


def test_short_collateral_does_not_earn_extra_interest():
    ledger = simulate_signed(*inputs([0.0], [-2.0], cash=0.0002), cost_bps=0, borrow_annual=0)
    assert ledger.net_return.iloc[0] == pytest.approx(0.0002)


def test_signed_turnover_and_fees():
    ledger = simulate_signed(*inputs([0.0], [-1.0]), cost_bps=100, borrow_annual=0)
    assert ledger.equity.iloc[0] == pytest.approx((1 / 1.01) * 0.99)


def test_leverage_cap_enforced():
    with pytest.raises(ValueError):
        simulate_signed(*inputs([0.1], [2.1]))


def test_insolvency_is_not_dropped_or_hidden():
    ledger = simulate_signed(*inputs([0.6, 0.1], [-2.0, -2.0]), cost_bps=0, borrow_annual=0)
    assert ledger.insolvent.sum() == 1
    assert ledger.equity.iloc[-1] == 0
    assert ledger.weight.iloc[-1] == 0


def test_daily_margin_call_liquidates_and_halts():
    ledger = simulate_signed(*inputs([-0.4, 0.1], [2.0, 2.0]), cost_bps=0, financing_spread=0)
    assert ledger.margin_call.sum() == 1
    assert ledger.equity.iloc[0] == pytest.approx(0.2)
    assert ledger.weight.iloc[-1] == 0


def test_long_only_accounting_matches_existing_engine():
    from oil_research.extension_experiments import simulate_cash

    asset, cash, dt, w = inputs([0.02, -0.03, 0.01], [0.4, 1.0, 0.0], cash=0.0001)
    a = simulate_signed(asset, cash, dt, w, cost_bps=5)
    b = simulate_cash(asset, cash, w, cost_bps=5)
    np.testing.assert_allclose(a.net_return, b.net_return, rtol=1e-12, atol=1e-12)
