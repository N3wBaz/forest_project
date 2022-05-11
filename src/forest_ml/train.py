from pathlib import Path
from joblib import dump


import click

# import pandas as pd
import sys
import os
import warnings
from math import inf


import mlflow.sklearn
import mlflow


from sklearn.model_selection import cross_validate


from .data import get_dataset
from .pipeline import create_pipeline


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option("--feature-select", default=0, type=int, show_default=True)
@click.option(
    "--random-state", 
    default=42,
    type=click.IntRange(0, inf),
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
    default=100,
    type=click.IntRange(True, inf),
    show_default=True,
)
@click.option(
    "--logreg-c",
    default=1.0,
    type=float,
    show_default=True,
)
@click.option(
    "--kf-part",
    default=5,
    type=click.IntRange(True, inf),
    show_default=True,
)
@click.option(
    "--other-model",
    default=False,
    type=bool,
    show_default=True,
)
@click.option(
    "--criterion",
    default="gini",
    type=str,
    show_default=True,
)
@click.option(
    "--splitter",
    default="best",
    type=str,
    show_default=True,
)
@click.option(
    "--max-depth",
    default=5,
    type=click.IntRange(True, inf),
    show_default=True,
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    feature_select: int,
    random_state: int,
    use_scaler: bool,
    max_iter: int,
    logreg_c: float,
    kf_part: int,
    other_model: bool,
    criterion: str,
    splitter: str,
    max_depth: int,
) -> None:

    if not sys.warnoptions:
        warnings.simplefilter("ignore")

        # Also affect subprocesses
        os.environ["PYTHONWARNINGS"] = "ignore"

        features, target = get_dataset(dataset_path, feature_select)

    with mlflow.start_run():
        # mlflow server \
        #     --backend-store-uri sqlite:///mlflow.db \
        #     --default-artifact-root ./artifacts \
        #     --host 0.0.0.0
        # And set MLFLOW_TRACKING_URI environment variable to http://localhost:5000 or

        pipeline = create_pipeline(
            use_scaler,
            max_iter,
            logreg_c,
            random_state,
            other_model,
            criterion,
            splitter,
            max_depth,
        )
        if not sys.warnoptions:
            warnings.simplefilter("ignore")

            # Also affect subprocesses
            os.environ["PYTHONWARNINGS"] = "ignore"
            cv_results = cross_validate(
                pipeline,
                features,
                target,
                cv=kf_part,
                scoring=("accuracy", "f1_macro", "roc_auc_ovr"),
            )

        # Logging model parameters
        if other_model:
            mlflow.log_param("criterion", criterion)
            mlflow.log_param("splitter", splitter)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("model_type", "DecisionTree")

        else:
            mlflow.log_param("use_scaler", use_scaler)
            mlflow.log_param("max_iter", max_iter)
            mlflow.log_param("logreg_c", logreg_c)
            mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("feature_select", feature_select)
        mlflow.log_param("k_folds", kf_part)

        # Logging metrics
        mlflow.log_metric("accurasy", cv_results["test_accuracy"].mean())
        mlflow.log_metric("f1_score", cv_results["test_f1_macro"].mean())
        mlflow.log_metric("roc_auc ovr", cv_results["test_roc_auc_ovr"].mean())
        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")
        click.echo("Cross-validation scores for different methrics :")
        click.echo(f"accuracy : {cv_results['test_accuracy'].mean()}.")
        click.echo(f"f1_score : {cv_results['test_f1_macro'].mean()}.")
        click.echo(f"roc_auc : {cv_results['test_roc_auc_ovr'].mean()}.")
    mlflow.end_run()
