from risk.risk_manager import RiskManager, RiskState


def test_risk_manager_disable_after_losses():
    rm = RiskManager(0.01, 0.015, 0.03, 0.08, 3, 5, 0.0, 0.005)
    state = RiskState(equity=10_000, peak_equity=10_000, consecutive_losses=5)
    assert rm.can_trade(state) is False
