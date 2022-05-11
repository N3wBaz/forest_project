from unittest import result
from click.testing import CliRunner
import pytest

from forest_ml.nested_cv import train_nested_cv


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_error_for_invalid_feature_select(runner: CliRunner) -> None:
    """It fails when feature select is not integer."""
    result = runner.invoke(
        train_nested_cv,
        [
            "--feature-select",
            "a",
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--feature-select'" in result.output


def test_error_for_invalid_random_state(runner: CliRunner) -> None:
    """It fails when random state is not integer."""
    result = runner.invoke(
        train_nested_cv,
        [
            "--random-state",
            "a",
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--random-state'" in result.output


def test_error_for_invalid_random_state1(runner: CliRunner) -> None:
    """It fails when random state is negative."""
    result = runner.invoke(
        train_nested_cv,
        [
            "--random-state",
            -1,
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--random-state'" in result.output
