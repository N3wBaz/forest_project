from pathlib import Path

import click
import pandas as pd
from .data import get_dataset

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





def train(
    dataset_path: Path,
    random_state: int,
    test_split_ratio: float
) -> None:
    # data = pd.read_csv(dataset_path, 42, 0.5)
    features_train, features_val, target_train, target_val = get_dataset(
        dataset_path,
        random_state,
        test_split_ratio)

