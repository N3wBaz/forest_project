from unittest import result
from click.testing import CliRunner
import pytest

from forest_ml.train import train


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_error_for_invalid_feature_select(runner: CliRunner) -> None:
    """It fails when feature select is not integer."""
    result = runner.invoke(
        train,
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
        train,
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
        train,
        [
            "--random-state",
            -1,
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--random-state'" in result.output


def test_error_for_invalid_use_scaler(runner: CliRunner) -> None:
    """It fails when use scaler is not boolean."""
    result = runner.invoke(
        train,
        [
            "--use-scaler",
            "a",
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--use-scaler'" in result.output


def test_error_for_invalid_max_iter(runner: CliRunner) -> None:
    """It fails when use max iter is not integer."""
    result = runner.invoke(
        train,
        [
            "--max-iter",
            "a",
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--max-iter'" in result.output


def test_error_for_invalid_max_iter1(runner: CliRunner) -> None:
    """It fails when use max iter is zero or negative."""
    result = runner.invoke(
        train,
        [
            "--max-iter",
            -1,
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--max-iter'" in result.output


def test_error_for_invalid_logreg_c(runner: CliRunner) -> None:
    """It fails when use logreg_c is not integer."""
    result = runner.invoke(
        train,
        [
            "--logreg-c",
            "a",
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--logreg-c'" in result.output


def test_error_for_invalid_kf_part(runner: CliRunner) -> None:
    """It fails when use k-fold split is not integer."""
    result = runner.invoke(
        train,
        [
            "--kf-part",
            "a",
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--kf-part'" in result.output


def test_error_for_invalid_kf_part1(runner: CliRunner) -> None:
    """It fails when use k-fold split is zero or negative."""
    result = runner.invoke(
        train,
        [
            "--kf-part",
            -1,
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--kf-part'" in result.output


def test_error_for_invalid_other_model(runner: CliRunner) -> None:
    """It fails when use scaler is not boolean."""
    result = runner.invoke(
        train,
        [
            "--other-model",
            "a",
        ],
    )

    assert result.exit_code == 2
    assert "Invalid value for '--other-model'" in result.output


def test_error_for_invalid_max_depth(runner: CliRunner) -> None:
    """It fails when use scaler is not string."""
    result = runner.invoke(
        train,
        [
            "--max-depth",
            "a",
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--max-depth'" in result.output


def test_error_for_invalid_max_depth1(runner: CliRunner) -> None:
    """It fails when use scaler is not string."""
    result = runner.invoke(
        train,
        [
            "--max-depth",
            -1,
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--max-depth'" in result.output
