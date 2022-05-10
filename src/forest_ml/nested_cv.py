from pathlib import Path

from joblib import dump

import click

import sys
import os
import warnings

import mlflow.sklearn
import mlflow
import numpy as np


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from .data import get_dataset

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler


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
    default="data/model_nested_cv.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option("--feature-select", default=0, type=int, show_default=True)
@click.option("--random-state", default=42, type=int, show_default=True)
def train_nested_cv(
    dataset_path: Path,
    save_model_path: Path,
    feature_select: int,
    random_state: int,
) -> None:
    print("Nested cross validation")
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

        features, target = get_dataset(dataset_path, feature_select)

    outer_loop = StratifiedKFold(n_splits=2, random_state=random_state, shuffle=True)

    clf1 = LogisticRegression(
        max_iter=100,
        tol=0.1,
        # scale = PowerTransformer()
        multi_class="multinomial",
        solver="newton-cg",
        random_state=1,
    )

    clf2 = DecisionTreeClassifier(
        criterion="gini", splitter="best", max_depth=5, random_state=random_state
    )

    clf3 = RandomForestClassifier(random_state=random_state)

    clf4 = KNeighborsClassifier()

    pipe1 = Pipeline([("scale", StandardScaler()), ("clf1", clf1)])

    pipe2 = Pipeline([("scale", StandardScaler()), ("clf2", clf2)])

    pipe3 = Pipeline([("scale", StandardScaler()), ("clf3", clf3)])

    pipe4 = Pipeline([("scale", StandardScaler()), ("clf4", clf4)])

    param_grid1 = [
        {"clf1__penalty": ["l2", "none"], "clf1__C": np.power(10.0, np.arange(-5, 5))}
    ]

    param_grid2 = [
        {
            "clf2__max_depth": [*range(1, 20)] + [None],
            "clf2__splitter": ["best", "random"],
            "clf2__criterion": ["gini", "entropy"],
        }
    ]

    param_grid3 = [
        {
            "clf3__max_depth": [*range(1, 26)] + [None],
            "clf3__n_estimators": [*range(5, 101, 10)],
        }
    ]

    param_grid4 = [
        {
            "clf4__n_neighbors": [*range(5, 106, 5)],
            "clf4__p": [1, 2],
        }
    ]

    grid_search = dict()
    inner_loop = StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state)

    with mlflow.start_run():
        # parameters_grid = (param_grid1, param_grid2, param_grid3, param_grid4)
        # pipe = (pipe1, pipe2, pipe3, pipe4)
        # names = ("LogReg", "DcsnTree", "RandForest", "K_neigoibors")

        parameters_grid = (param_grid1, param_grid2)
        pipe = (pipe1, pipe2)
        names = ("LogReg", "DcsnTree")

        for param_grid, estimator, name in zip(parameters_grid, pipe, names):
            search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                scoring="accuracy",
                n_jobs=-1,
                cv=inner_loop,
                verbose=0,
                refit=True,
            )
            grid_search[name] = search

        best_score = list()

        for name in grid_search:

            roc_score, acc_score, f_score = list(), list(), list()

            for train_index, test_index in outer_loop.split(features, target):
                X_train, X_test = (
                    features.iloc[train_index, :],
                    features.iloc[test_index, :],
                )
                y_train, y_test = target[train_index], target[test_index]
                grid_search[name].fit(X_train, y_train)
                best_model = grid_search[name].best_estimator_
                acc = accuracy_score(best_model.predict(X_test), y_test)
                roc_auc = roc_auc_score(
                    y_test,
                    best_model.predict_proba(X_test),
                    multi_class="ovr",
                    average="macro",
                )
                f_measure = f1_score(
                    y_test, best_model.predict(X_test), average="macro"
                )
                
                roc_score.append(roc_auc)
                acc_score.append(acc)
                f_score.append(f_measure)

            with mlflow.start_run(nested=True):
                mlflow.log_param("model_type", name)
                mlflow.log_metric("accurasy", np.array(acc).mean())
                mlflow.log_metric("roc_auc ovr", np.array(roc_score).mean())
                mlflow.log_metric("f1_score", np.array(f_measure).mean())

            print(f"{name}  :  roc_auc  : {np.array(roc_score).mean()}")
            print(f"           f1_score : {np.array(f_measure).mean()}")
            print(f"           accuracy : {np.array(acc).mean()}")
            best_score.append(np.array(acc).mean())
        idx = np.argmax(best_score)
        model_key = list(grid_search)
        print(
            f"best model : {model_key[idx]} {grid_search[model_key[idx]].best_params_}"
        )

        final = grid_search[model_key[idx]].best_estimator_

        final.fit(features, target)
        mlflow.sklearn.log_model(final, artifact_path="best-model")

        dump(final, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")

        for key in grid_search[model_key[idx]].best_params_:
            mlflow.log_param(key[6:], grid_search[model_key[idx]].best_params_[key])
        mlflow.log_param("feature_select", feature_select)
        mlflow.log_param("model_type", model_key[idx])
