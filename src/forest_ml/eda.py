import pandas_profiling


from pathlib import Path

import click
import sys
import os
import warnings


from .data import get_data


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
def eda(
    dataset_path: Path,
) -> None:

    data = get_data(dataset_path)

    data = get_data(dataset_path)
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses
        profile = data.profile_report(title="Pandas Profiling Report")

    profile.to_file("Forest_report.html")
