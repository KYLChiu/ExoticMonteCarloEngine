from ExoticEngine.Payoff import Options as O


def test_call_payoff():
    call = O.PayOffCall(strike=1.0)
    assert call.payoff(spot=1.0) == 0
    assert call.payoff(spot=0.5) == 0
    assert call.payoff(spot=1.5) == 0.5


def test_put_payoff():
    put = O.PayOffPut(strike=3.0)
    assert put.payoff(spot=1.0) == 2.0
    assert put.payoff(spot=0.5) == 2.5
    assert put.payoff(spot=5.5) == 0.0


def test_double_digital_payoff():
    double_digital = O.PayOffDoubleDigital(upper=30.0, lower=20.0)
    assert double_digital.payoff(spot=25) == 1
    assert double_digital.payoff(spot=0.5) == 0
    assert double_digital.payoff(spot=50) == 0.0


def test_put_option_payoff():
    put = O.PayOffPut(strike=3.0)
    put_option = O.VanillaOption(put, 5.0)
    assert put_option.get_payoff(spot=1.0) == 2.0
    assert put_option.get_payoff(spot=6.0) == 0.0


def test_option_expiry():
    call = O.PayOffCall(strike=3.0)
    expiry = 5.0
    call_option = O.VanillaOption(call, expiry)
    assert call_option.get_expiry() == expiry
