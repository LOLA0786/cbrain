from cbrain import evaluate_action

def test_high_risk():
    result = evaluate_action("transfer_funds")
    assert result.decision == "BLOCK"

def test_low_risk():
    result = evaluate_action("read_logs")
    assert result.decision == "ALLOW"
