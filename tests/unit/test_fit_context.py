import dataclasses

import numpy as np
import pytest

from hydrochoerus.model.fit_context import FitContext
from hydrochoerus.model.single_alpha_model import SingleAlpha


def test_should_hold_gap_and_observed(hand_grid):
    context = FitContext(gap=hand_grid, observed=np.array([0.4, 1.2]))
    expected = (hand_grid, [0.4, 1.2])
    assert (context.gap, context.observed.tolist()) == expected


def test_should_fix_nothing_by_default(hand_grid):
    expected = frozenset()
    assert FitContext(gap=hand_grid, observed=np.zeros(2)).fixed == expected


def test_should_not_cancel_by_default(hand_grid):
    assert FitContext(gap=hand_grid, observed=np.zeros(2)).should_cancel() is False


def test_should_ignore_steps_by_default(hand_grid):
    context = FitContext(gap=hand_grid, observed=np.zeros(2))
    assert context.on_step(SingleAlpha()) is None


def test_should_be_frozen(hand_grid):
    context = FitContext(gap=hand_grid, observed=np.zeros(2))
    with pytest.raises(dataclasses.FrozenInstanceError):
        context.fixed = frozenset({"alpha"})


def test_should_pass_steps_to_the_given_callback(mocker, hand_grid):
    on_step = mocker.Mock()
    model = SingleAlpha()
    FitContext(gap=hand_grid, observed=np.zeros(2), on_step=on_step).on_step(model)
    on_step.assert_called_once_with(model)
