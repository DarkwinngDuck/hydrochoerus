import dataclasses

import numpy as np
import pytest

from hydrochoerus.model.experimental_data import ExperimentalData
from hydrochoerus.utils.readonly_columns import ShapeError


def create_fixture():
    return ExperimentalData(
        "test.dat", "test_path/test.dat", np.array([0.5, 1.0]), np.array([2.0, 3.0])
    )


def test_t_cannot_be_reassigned():
    with pytest.raises(dataclasses.FrozenInstanceError):
        create_fixture().t = np.array([9.0])


def test_y_cannot_be_reassigned():
    with pytest.raises(dataclasses.FrozenInstanceError):
        create_fixture().y = np.array([9.0])


def test_values_inside_t_cannot_be_reassigned():
    with pytest.raises(ValueError, match="read-only"):
        create_fixture().t[0] = 9.0


def test_values_inside_y_cannot_be_reassigned():
    with pytest.raises(ValueError, match="read-only"):
        create_fixture().y[0] = 9.0


def test_experimental_t_is_immutable():
    first_t = 0.5
    latest_t = 4.0
    t = np.array([first_t])
    data = ExperimentalData("name", "path", t, np.array([0.0]))
    t[0] = latest_t

    assert data.t[0] == first_t


def test_experimental_y_is_immutable():
    first_y = 2.0
    latest_y = 5.0
    y = np.array([first_y])
    data = ExperimentalData("name", "path", np.array([0.0]), y)
    y[0] = latest_y

    assert data.y[0] == first_y


def test_t_and_y_must_have_the_same_shape():
    with pytest.raises(ShapeError):
        ExperimentalData("name", "path", np.array([0.5, 1.0]), np.array([2.0]))
