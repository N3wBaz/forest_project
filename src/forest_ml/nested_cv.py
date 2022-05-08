from pathlib import Path
from random import shuffle
from joblib import dump

import click
import pandas as pd
import sys
import os
import warnings

import mlflow.sklearn
import mlflow
import numpy as np



from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold 
from sklearn.model_selection import GridSearchCV
from .data import get_dataset
from .pipeline import create_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True
)

@click.option(
    "-s",
    "--save-model-path",
    default="data/model_nested_cv.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True
)

@click.option(
    "--feature-select",
    default=0,
    type=int,
    show_default=True
)

@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True
)

def train_nested_cv(
    dataset_path: Path,
    save_model_path: Path,
    feature_select: int,
    random_state: int,

) -> None:
    print('Nested cross validation')
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

        features, target = get_dataset(dataset_path, feature_select)
    # pass

    # pipeline = create_pipeline(
    #     use_scaler=True, 
    #     max_iter=100, 
    #     logreg_c=1, 
    #     random_state=random_state, 
    #     other_model=False, 
    #     criterion='gini',
    #     splitter='best',
    #     max_depth='5'
    # )


    # logistic = LogisticRegression(max_iter=100, tol=0.1)
    # scaler = PowerTransformer()
    # pipe = Pipeline(steps=[("scaler", scaler), ("logistic", logistic)])

    # param_grid = {
    #     "logistic__C": np.logspace(-4, 4, 4),
    # }
    # search = GridSearchCV(pipe, param_grid, n_jobs=2)
    # search.fit(features, target)
    # print("Best parameter (CV score=%0.3f):" % search.best_score_)
    # print(search.best_params_)

    outer_loop = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True) 


    clf1 = LogisticRegression(
        max_iter=100, 
        tol=0.1,
    # scale = PowerTransformer()
        multi_class='multinomial',
        solver='newton-cg',
        random_state=1
    )


    clf2 = DecisionTreeClassifier(
        criterion='gini',
        splitter='best',
        max_depth=5,
        random_state=random_state
    )

    pipe1 = Pipeline([
        ('scale', PowerTransformer()),
        ('clf1', clf1)])

    pipe2 = Pipeline([('std', PowerTransformer()),
                    ('clf2', clf2)])

    param_grid1 = [{
        'clf1__penalty': ['l2', 'none'],
        'clf1__C': np.power(10., np.arange(-4, 4))
    }]

    param_grid2 = [{
        'clf2__max_depth': [*range(1, 5)] + [None],
        'clf2__splitter' : ['best', 'random'],
        'clf2__criterion': ['gini', 'entropy']
    }]



    grid_search = {}
    inner_loop = StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state)

    for param_grid, estimator, name in zip((param_grid1, param_grid2),
                                (pipe1, pipe2),
                                ('Logreg', 'DcsnTree')):
        search = GridSearchCV(estimator=estimator,
                        param_grid=param_grid,
                        scoring='accuracy',
                        n_jobs=-1,
                        cv=inner_loop,
                        verbose=0,
                        refit=True)
        grid_search[name] = search
    # print(gridcvs)
    # gridcvs['Logreg'].fit(features, target)
    # print(gridcvs['Logreg'].best_params_)
    best_score = []
    best_models = {}

    for name in grid_search:
        # print(name)
        # print(gridcvs[name])
        best_models[name] = []


        for train_index , test_index in outer_loop.split(features, target):
            X_train , X_test = features.iloc[train_index,:], features.iloc[test_index,:]
            y_train , y_test = target[train_index] , target[test_index]
            grid_search[name].fit(X_train, y_train)
            # print(gridcvs[name].best_params_)
            best_models[name].append(grid_search[name].best_params_)
            best_score.append((name, grid_search[name].best_estimator_.score(X_test, y_test)))



    print(best_models)
    for name in best_models:
        for i, v in enumerate(best_models[name]):
            print(f"{i + 1} модель      {v}   ")
    for i in best_score:
        print(f"{i[0]}            {i[1]}")














            # print(type(model))
    #     print(X_train.shape)
    #     if not sys.warnoptions:
    #         warnings.simplefilter("ignore")
    #         os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
    #         pipeline.fit(X_train,y_train)

    #     pred_values = pipeline.predict(X_test)
        
    #     acc = accuracy_score(pred_values , y_test)
    #     roc_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test), multi_class='ovr', average="macro")
    #     f_measure = f1_score(y_test, pred_values,  average='macro')

    #     roc_score.append(roc_auc)
    #     acc_score.append(acc)
    #     f_score.append(f_measure)

    # if not sys.warnoptions:
    #     warnings.simplefilter("ignore")
    #     os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
    #     # cross_validate
    #     cv_results = cross_validate(pipeline, features, target, cv=kf_part, scoring=('accuracy', 'f1_macro', 'roc_auc_ovr'),)
    