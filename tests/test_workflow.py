from alpha.live.workflow import FAIL, IDLE, OK, WARN, Check, Stage, by_age, by_level, fmt_age


def test_age_and_level_rules():
    assert by_age(10, 15, 60) == OK and by_age(30, 15, 60) == WARN and by_age(90, 15, 60) == FAIL
    assert by_age(None, 15, 60) == FAIL                      # never seen = broken
    assert by_level(0.1, 0.15, 0.25) == OK and by_level(0.2, 0.15, 0.25) == WARN and by_level(0.3, 0.15, 0.25) == FAIL
    assert by_level(None, 1, 2) == IDLE
    assert fmt_age(None) == "never" and fmt_age(30) == "30s ago" and fmt_age(600) == "10m ago"


def test_stage_takes_worst_live_check_and_ignores_idle():
    s = Stage("x", "X", "")
    assert s.status == IDLE
    s.checks = [Check("a", IDLE, ""), Check("b", OK, "")]
    assert s.status == OK
    s.checks.append(Check("c", WARN, ""))
    assert s.status == WARN
    s.checks.append(Check("d", FAIL, ""))
    assert s.status == FAIL
    assert Stage("y", "Y", "", [Check("a", IDLE, "")]).status == IDLE


def test_finished_outage_is_amber_not_red():
    from alpha.live.workflow import disconnect_status
    assert disconnect_status(0, 0, True, None)[0] == OK
    assert disconnect_status(6, 30, False, 20)[0] == FAIL                 # still happening, no data
    st, text = disconnect_status(0, 36, True, 300)                        # tonight's case after recovery
    assert st == WARN and "recovered" in text
