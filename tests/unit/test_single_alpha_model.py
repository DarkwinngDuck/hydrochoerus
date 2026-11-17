import pytest
from pydantic import ValidationError

from hydrochoerus.consts import ALPHA_BCS
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
