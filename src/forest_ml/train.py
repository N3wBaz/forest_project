from pathlib import Path

import click
import pandas as pd
import sys
import os
import warnings


from sklearn.metrics import accuracy_score
from .data import get_dataset
from .pipeline import create_pipeline

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True
)

@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True
)

@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True
)

@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)

@click.option(
    "--max-iter",
    default=300,
    type=int,
    show_default=True,
)

@click.option(
    "--logreg-c",
    default=1.0,
    type=float,
    show_default=True,
)


def train(
    dataset_path: Path,
    random_state: int,
    test_split_ratio: float,
    use_scaler: bool,
    max_iter: int,
    logreg_c: float,
) -> None:
    # data = pd.read_csv(dataset_path, 42, 0.5)
    features_train, features_val, target_train, target_val = get_dataset(
        dataset_path,
        random_state,
        test_split_ratio)
    # print(features_train.head(5))
    pipeline = create_pipeline(use_scaler, max_iter, logreg_c, random_state)
    pipeline.fit(features_train, target_train)
    accuracy = accuracy_score( target_val, pipeline.predict(features_val))

    if not sys.warnoptions:
                warnings.simplefilter("ignore")
                os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

    click.echo(f"Accuracy: {accuracy}.")
    # print(e.head(5))

