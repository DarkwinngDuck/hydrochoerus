import numpy as np
import pytest
from pydantic import ValidationError

from hydrochoerus.consts import ALPHA_BCS
from hydrochoerus.core.specific_heat import ces_on_grid
from hydrochoerus.model.models import ModelType, ParameterType
from hydrochoerus.model.single_alpha_model import SingleAlpha


def test_should_have_single_alpha_type():
    expected = ModelType.SINGLE_ALPHA
    assert SingleAlpha().type is expected


def test_should_refuse_another_type():
    with pytest.raises(ValidationError, match="type"):
        SingleAlpha(type=ModelType.TWO_ALPHA)


def test_should_have_only_alpha_as_parameter():
    expected = ("alpha",)
    assert SingleAlpha.parameter_names() == expected


def test_should_default_alpha_to_bcs():
    expected = ALPHA_BCS
    assert SingleAlpha().alpha == expected


def test_should_bound_alpha():
    expected = ([0.01], [20.0])
    assert SingleAlpha.bounds(["alpha"]) == expected


def test_should_mark_alpha_as_gap():
    expected = ParameterType.GAP
    assert SingleAlpha.ptype("alpha") is expected


def test_should_return_what_the_injected_ces_returns(mocker, hand_grid):
    expected = np.array([0.3, 1.2])
    ces = mocker.Mock(return_value=expected)
    assert SingleAlpha(alpha=1.7).curve(hand_grid, ces=ces) is expected


def test_should_pass_grid_and_alpha_to_ces(mocker, hand_grid):
    ces = mocker.Mock(return_value=np.zeros(2))
    SingleAlpha(alpha=1.7).curve(hand_grid, ces=ces)
    ces.assert_called_once_with(hand_grid, 1.7)


def test_should_use_ces_on_grid_by_default(hand_grid):
    expected = ces_on_grid(hand_grid, 1.7).tolist()
    assert SingleAlpha(alpha=1.7).curve(hand_grid).tolist() == expected


def test_should_have_no_fit_method():
    assert not hasattr(SingleAlpha, "fit")
