import numpy as np

from hydrochoerus.core.gap_cache import GapGridCache
from hydrochoerus.model.gap_grid import GapGrid


def fake_solve(reduced_temperatures: np.ndarray) -> GapGrid:
    zeros = np.zeros_like(reduced_temperatures)
    return GapGrid(reduced_temperatures, zeros, zeros)


def received(spy) -> list[list[float]]:
    return [call.args[0].tolist() for call in spy.call_args_list]


def test_should_return_expected_grid(mocker):
    reduced_temperatures = np.array([0.3, 0.8])
    mock_grid = fake_solve(reduced_temperatures)
    spy = mocker.Mock(return_value=mock_grid)
    assert GapGridCache(spy).get(reduced_temperatures) is mock_grid


def test_should_recalculate_grid():
    reduced_temperatures = np.array([0.3, 0.8])
    cache = GapGridCache(fake_solve)
    first = cache.get(reduced_temperatures)
    second = cache.get(reduced_temperatures, True)
    assert second is not first
