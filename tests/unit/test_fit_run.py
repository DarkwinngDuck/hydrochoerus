import pytest

from hydrochoerus.model.fit_run import (
    FitResult,
    FitRun,
    InvalidTransition,
    Metrics,
    RunStatus,
    SweepStep,
)


def mock_fit_result() -> FitResult:
    return FitResult({"alpha": 1.764}, Metrics(residual=0.0, r_squared=1.0, r_factor=0.0))


def mock_fit_run() -> FitRun:
    return FitRun(id=1, scan_name="test", model_name="test", inputs={"alpha": 1.7}, t_max=0.95)


def test_new_fit_is_running():
    assert mock_fit_run().status is RunStatus.RUNNING


def test_should_be_finished():
    run = mock_fit_run()
    run.finish(RunStatus.DONE, result=mock_fit_result())
    assert run.status is RunStatus.DONE


def test_should_not_change_after_finished():
    run = mock_fit_run()
    run.finish(RunStatus.DONE, result=mock_fit_result())
    with pytest.raises(InvalidTransition):
        run.finish(RunStatus.CANCELLED, result=mock_fit_result())


def test_should_not_finish_with_running_status():
    with pytest.raises(InvalidTransition):
        mock_fit_run().finish(RunStatus.RUNNING)


@pytest.mark.parametrize("status", [RunStatus.DONE, RunStatus.SIMULATED])
def test_should_have_result_if_done_or_simulated(status):
    run = mock_fit_run()
    with pytest.raises(ValueError, match="FitResult is required"):
        run.finish(status)
    assert run.status is RunStatus.RUNNING


def test_should_keep_error_if_failed():
    run = mock_fit_run()
    mock_error = "curve_fit did not converge"
    run.finish(RunStatus.FAILED, error=mock_error)
    assert run.error == mock_error


def test_should_not_have_result_if_failed():
    run = mock_fit_run()
    run.finish(RunStatus.FAILED, error="")
    assert run.result is None


def test_should_not_have_result_if_cancelled():
    run = mock_fit_run()
    run.finish(RunStatus.CANCELLED)
    assert run.status is RunStatus.CANCELLED
    assert run.result is None


def test_should_not_share_sweep_steps_between_rans():
    first, second = mock_fit_run(), mock_fit_run()
    first.sweep_steps.append(SweepStep(0.5, {}, Metrics(0, 1, 0)))
    assert second.sweep_steps == []
