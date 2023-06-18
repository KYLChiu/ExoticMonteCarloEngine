from ExoticEngine.Payoff import Options as O

def test_call_payoff():
    Call = O.PayOffCall(strike=1.)
    assert Call.payoff(spot=1.) == 0
    assert Call.payoff(spot=0.5) == 0
    assert Call.payoff(spot=1.5) == 0.5

def test_put_payoff():
    Put = O.PayOffPut(strike=3.)
    assert Put.payoff(spot=1.) == 2.
    assert Put.payoff(spot=0.5) == 2.5
    assert Put.payoff(spot=5.5) == 0.

def test_double_digital_payoff():
    DD = O.PayOffDoubleDigital(upper=30., lower=20.)
    assert DD.payoff(spot=25) == 1
    assert DD.payoff(spot=0.5) == 0
    assert DD.payoff(spot=50) == 0.

def test_put_option_payoff():
    Put = O.PayOffPut(strike=3.)
    PutOption = O.VanillaOption(Put, 5.)
    assert PutOption.get_payoff(spot=1.) == 2.
    assert PutOption.get_payoff(spot=6.) == 0.

def test_option_expiry():
    Call = O.PayOffCall(strike=3.)
    expiry = 5.
    CallOption = O.VanillaOption(Call, expiry)
    assert CallOption.get_expiry() == expiry
