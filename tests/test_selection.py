"""The selection rule, including the tie-break that decides close runs.

The behaviour worth pinning is that the tie-break can only ever move the
choice towards a cheaper model, and only when cross-validation has failed
to separate the candidates. It reads nothing from the holdout, which is
what keeps it from being model selection on the test set.
"""

from __future__ import annotations

from olist_retention.models.train import SERVING_COST_ORDER, select_winner


def summary(mean: float, deviation: float) -> dict:
    return {
        "folds": 5,
        "roc_auc_mean": 0.6,
        "roc_auc_std": 0.01,
        "average_precision_mean": mean,
        "average_precision_std": deviation,
    }


def test_a_clear_leader_wins_without_a_tie_break():
    selection = select_winner(
        {
            "logistic_regression": summary(0.020, 0.002),
            "keras_mlp": summary(0.060, 0.002),
        }
    )
    assert selection.winner == "keras_mlp"
    assert selection.tie_break_applied is False
    assert selection.contenders == ["keras_mlp"]


def test_the_cheaper_model_wins_when_the_gap_is_inside_the_deviation():
    """The measured case: three candidates within one deviation."""
    selection = select_winner(
        {
            "logistic_regression": summary(0.0318, 0.0117),
            "gradient_boosting": summary(0.0278, 0.0121),
            "keras_mlp": summary(0.0364, 0.0091),
        }
    )
    assert selection.leader == "keras_mlp"
    assert selection.winner == "logistic_regression"
    assert selection.tie_break_applied is True
    assert set(selection.contenders) == {
        "logistic_regression",
        "gradient_boosting",
        "keras_mlp",
    }
    assert "does not separate" in selection.reason


def test_a_leader_that_is_already_cheapest_keeps_the_win():
    selection = select_winner(
        {
            "logistic_regression": summary(0.0364, 0.0091),
            "keras_mlp": summary(0.0330, 0.0100),
        }
    )
    assert selection.winner == "logistic_regression"
    assert selection.tie_break_applied is False


def test_the_tie_break_never_promotes_a_more_expensive_model():
    """Whatever the numbers, the winner cannot cost more than the leader."""
    cases = [
        {"logistic_regression": summary(0.030, 0.001), "keras_mlp": summary(0.031, 0.001)},
        {"logistic_regression": summary(0.030, 0.020), "keras_mlp": summary(0.031, 0.020)},
        {"gradient_boosting": summary(0.040, 0.010), "keras_mlp": summary(0.045, 0.010)},
    ]
    for cv_summary in cases:
        selection = select_winner(cv_summary)
        assert SERVING_COST_ORDER.index(selection.winner) <= SERVING_COST_ORDER.index(
            selection.leader
        )


def test_a_single_candidate_is_its_own_winner():
    selection = select_winner({"logistic_regression": summary(0.03, 0.01)})
    assert selection.winner == "logistic_regression"
    assert selection.tie_break_applied is False


def test_an_unknown_model_name_is_treated_as_the_most_expensive():
    """A new candidate has to be priced deliberately before it can win a tie."""
    selection = select_winner(
        {
            "mystery_model": summary(0.0364, 0.0091),
            "logistic_regression": summary(0.0330, 0.0091),
        }
    )
    assert selection.winner == "logistic_regression"
    assert selection.tie_break_applied is True


def test_the_reason_is_always_written_down():
    selection = select_winner(
        {"logistic_regression": summary(0.03, 0.01), "keras_mlp": summary(0.031, 0.01)}
    )
    assert selection.reason
    assert selection.winner in selection.reason
