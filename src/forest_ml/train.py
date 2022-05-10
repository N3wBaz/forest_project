from pathlib import Path
from joblib import dump

import click
import pandas as pd
import sys
import os
import warnings
import pandas_profiling

import mlflow.sklearn
import mlflow


from sklearn.model_selection import cross_validate

# from sklearn.model_selection import StratifiedKFold
from .data import get_dataset
from .data import get_data
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
@click.option("--random-state", default=42, type=int, show_default=True)
@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True,
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
    type=int,
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
    type=int,
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
    type=int,
    show_default=True,
)
@click.option(
    "--nested-cv",
    defeault=False,
    type=bool,
    show_default=True,
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    feature_select: int,
    random_state: int,
    test_split_ratio: float,
    use_scaler: bool,
    max_iter: int,
    logreg_c: float,
    kf_part: int,
    other_model: bool,
    criterion: str,
    splitter: str,
    max_depth: int,
    nested_cv: bool,
) -> None:

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

        features, target = get_dataset(dataset_path, feature_select)

    with mlflow.start_run():

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
            os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses
            # cross_validate
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
        click.echo(f"Cross-validation scores for different methrics :")
        click.echo(f"accuracy : {cv_results['test_accuracy'].mean()}.")
        click.echo(f"f1_score : {cv_results['test_f1_macro'].mean()}.")
        click.echo(f"roc_auc : {cv_results['test_roc_auc_ovr'].mean()}.")
    # mlflow.end_run()

    # click.echo(f"Roc auc score {sum(roc_score)/len(roc_score)}.")
    # click.echo(f"Accuracy score {sum(acc_score)/len(acc_score)}.")
    # click.echo(f"F score {sum(f_score)/len(f_score)}.")

    # https://towardsdatascience.com/complete-guide-to-pythons-cross-validation-with-examples-a9676b5cac12
    # почитать про КФОЛД

    # Отлавливаем ConvergenceWarning
    # if not sys.warnoptions:
    #     warnings.simplefilter("ignore")
    #     os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
    #     # model = pipeline.fit(features, target)
    #     # scores = cross_validate(model, features, target, scoring=make_scorer(cus_roc_auc, greater_is_better=True), cv=3, n_jobs=-1)
    #     scores = cross_val_score(pipeline, features, target, scoring="accuracy", cv = 5)

    #     pipeline.fit(features, target)

    # Сохраняем модель
    # dump(pipeline, save_model_path)
    # click.echo(f"Model is saved to {save_model_path}.")
    # click.echo(f"Model is saved to {scores}.")

    # # Считаем метрики

    # target_pred = pipeline.predict(features_val)

    # roc_auc = roc_auc_score(target_val, pipeline.predict_proba(features_val), multi_class='ovr',)
    # click.echo(f"ROC-AUC score: {roc_auc}.")

    # f_measure = f1_score(target_val, target_pred,  average='macro')
    # click.echo(f"F-measure: {f_measure}.")

    # accuracy = accuracy_score(pipeline.predict(features), target)
    # click.echo(f"Accuracy: {accuracy}.")


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
    # profile.to_file(outputfile="data/profiling.html")
    # print(data)
    profile.to_file("Forest_report.html")

    # data.profile_report()
