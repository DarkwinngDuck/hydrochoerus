from typing import Literal

import numpy as np
import pytest
from annotated_types import Ge, Le
from pydantic import Field, ValidationError

from hydrochoerus.model.gap_grid import GapGrid
from hydrochoerus.model.models import Gap, Model, ModelType, ParameterType, Weight
from hydrochoerus.model.single_alpha_model import SingleAlpha
from hydrochoerus.model.two_alpha_model import TwoAlpha

ALL_MODELS = [SingleAlpha, TwoAlpha]


class PairModel(Model):
    alpha: Gap = Field(1.0, ge=0.01, le=20)
    x: Weight = Field(0.5, ge=0, le=1)


class TaggedModel(Model):
    type: Literal["tagged"] = "tagged"
    alpha: Gap = Field(1.0, ge=0.01, le=20)


def undeclared_fields(model_class: type[Model]) -> list[str]:
    def declared(metadata):
        return all(any(isinstance(m, kind) for m in metadata) for kind in (Ge, Le, ParameterType))

    return [
        name
        for name, f in model_class.model_fields.items()
        if name != "type" and not declared(f.metadata)
    ]


def test_should_hold_defaults():
    model = PairModel()
    expected = (1.0, 0.5)
    assert (model.alpha, model.x) == expected


def test_should_refuse_value_above_bound():
    with pytest.raises(ValidationError, match="alpha"):
        PairModel(alpha=50.0)


def test_should_refuse_value_below_bound():
    with pytest.raises(ValidationError, match="x"):
        PairModel(x=-0.1)


def test_should_accept_value_on_bound():
    expected = 20.0
    assert PairModel(alpha=20.0).alpha == expected


def test_should_refuse_unknown_keyword():
    with pytest.raises(ValidationError, match="alpah"):
        PairModel(alpah=2.0)


def test_should_be_frozen():
    model = PairModel()
    with pytest.raises(ValidationError):
        model.alpha = 2.0


def test_should_refuse_default_outside_bounds_when_built():
    class BadModel(Model):
        alpha: Gap = Field(30.0, ge=0.01, le=20)

    with pytest.raises(ValidationError, match="alpha"):
        BadModel()


def test_should_store_values_as_float():
    expected = float
    assert type(PairModel(alpha=2).alpha) is expected


def test_should_list_parameter_names_in_declaration_order():
    expected = ("alpha", "x")
    assert PairModel.parameter_names() == expected


def test_should_leave_type_out_of_parameter_names():
    expected = ("alpha",)
    assert TaggedModel.parameter_names() == expected


def test_should_return_bounds_in_free_order():
    expected = ([0.0, 0.01], [1.0, 20.0])
    assert PairModel.bounds(["x", "alpha"]) == expected


def test_should_raise_for_unknown_name_in_bounds():
    with pytest.raises(KeyError):
        PairModel.bounds(["beta"])


def test_should_return_parameter_type_of_field():
    expected = ParameterType.WEIGHT
    assert PairModel.ptype("x") is expected


def test_should_return_vector_in_free_order():
    expected = [0.3, 2.0]
    assert PairModel(alpha=2.0, x=0.3).vector(["x", "alpha"]) == expected


def test_should_replace_only_free_fields():
    expected = PairModel(alpha=2.0, x=0.7)
    assert PairModel(alpha=2.0, x=0.3).with_vector(["x"], [0.7]) == expected


def test_should_keep_type_through_with_vector():
    expected = TaggedModel(alpha=3.0)
    assert TaggedModel().with_vector(["alpha"], [3.0]) == expected


def test_should_accept_numpy_values_in_with_vector():
    expected = 2.5
    assert PairModel().with_vector(["alpha"], np.array([2.5])).alpha == expected


def test_should_validate_with_vector():
    with pytest.raises(ValidationError, match="x"):
        PairModel().with_vector(["x"], [1.5])


def test_should_refuse_wrong_value_count():
    with pytest.raises(ValueError, match="zip"):
        PairModel().with_vector(["alpha", "x"], [2.0])


def test_should_round_trip_free_fields():
    start = PairModel(alpha=3.0, x=0.5)
    target = PairModel(alpha=3.0, x=0.25)
    free = ["x"]
    assert start.with_vector(free, target.vector(free)) == target


def test_should_refuse_curve_on_a_model_without_one():
    grid = GapGrid(np.array([0.5]), np.array([0.9]), np.array([-1.0]))
    with pytest.raises(NotImplementedError, match="PairModel"):
        PairModel().curve(grid)


def test_should_return_itself_when_normalised():
    model = PairModel()
    assert model.normalised() is model


def test_should_find_field_without_declaration():
    class BadModel(Model):
        alpha: float = 1.0

    expected = ["alpha"]
    assert undeclared_fields(BadModel) == expected


@pytest.mark.parametrize("model_class", ALL_MODELS)
def test_should_declare_bounds_and_type_for_every_field(model_class):
    expected = []
    assert undeclared_fields(model_class) == expected


@pytest.mark.parametrize("model_class", ALL_MODELS)
def test_should_build_from_defaults(model_class):
    assert isinstance(model_class(), model_class)


def test_should_give_every_model_its_own_type():
    expected = set(ModelType)
    assert {m().type for m in ALL_MODELS} == expected
