"""Tests for Cohen's kappa, Gwet's AC1 and their bootstrap intervals."""

import pytest
from sklearn.metrics import cohen_kappa_score

from src.agreement import agreement_report, cohen_kappa, gwet_ac1

YN = ("yes", "no")


def _table(both_yes: int, yes_no: int, no_yes: int, both_no: int) -> tuple[list[str], list[str]]:
    """Expand a 2 by 2 agreement table into two label lists."""
    a = ["yes"] * (both_yes + yes_no) + ["no"] * (no_yes + both_no)
    b = ["yes"] * both_yes + ["no"] * yes_no + ["yes"] * no_yes + ["no"] * both_no
    return a, b


def test_kappa_matches_sklearn_on_three_classes() -> None:
    """Kappa agrees with scikit-learn's independent implementation."""
    cats = ("positive", "negative", "neither")
    a = ["positive"] * 20 + ["negative"] * 50 + ["neither"] * 30
    b = ["positive"] * 15 + ["neither"] * 5 + ["negative"] * 45 + ["neither"] * 5 + \
        ["neither"] * 20 + ["negative"] * 10
    assert cohen_kappa(a, b, cats) == pytest.approx(cohen_kappa_score(a, b))


def test_kappa_paradox_table_by_hand() -> None:
    """High raw agreement with one dominant class: kappa near 0, AC1 near 0.94.

    Table (rater A by rater B): yes/yes 118, yes/no 5, no/yes 2, no/no 0; n = 125.
    po = 118/125 = 0.944. Kappa: pe = 0.984 * 0.96 + 0.016 * 0.04 = 0.94528,
    kappa = (0.944 - 0.94528) / (1 - 0.94528) = -0.02339.
    AC1: pi_yes = (0.984 + 0.96) / 2 = 0.972, pe = 2 * 0.972 * 0.028 / (2 - 1) = 0.054432,
    AC1 = (0.944 - 0.054432) / (1 - 0.054432) = 0.94078.
    """
    a, b = _table(118, 5, 2, 0)
    assert cohen_kappa(a, b, YN) == pytest.approx(-0.02339, abs=1e-4)
    assert gwet_ac1(a, b, YN) == pytest.approx(0.94078, abs=1e-4)


def test_perfect_agreement_scores_one() -> None:
    """Identical labels with both classes present give 1 for both statistics."""
    a, b = _table(30, 0, 0, 10)
    assert cohen_kappa(a, b, YN) == pytest.approx(1.0)
    assert gwet_ac1(a, b, YN) == pytest.approx(1.0)


def test_bad_inputs_raise() -> None:
    """Different lengths and unknown labels are errors."""
    with pytest.raises(ValueError):
        cohen_kappa(["yes"], ["yes", "no"], YN)
    with pytest.raises(ValueError):
        gwet_ac1(["maybe"], ["yes"], YN)


def test_report_intervals_contain_point_and_are_deterministic() -> None:
    """The bootstrap intervals bracket the point estimates and repeat under the same seed."""
    a, b = _table(60, 8, 6, 26)
    r1 = agreement_report(a, b, YN, n_boot=500, seed=3)
    r2 = agreement_report(a, b, YN, n_boot=500, seed=3)
    assert r1 == r2
    assert r1["kappa_ci"][0] <= r1["kappa"] <= r1["kappa_ci"][1]
    assert r1["ac1_ci"][0] <= r1["ac1"] <= r1["ac1_ci"][1]
    assert r1["percent_agreement"] == pytest.approx(0.86)
    assert r1["confusion"]["yes"]["no"] == 8 and r1["confusion"]["no"]["yes"] == 6
