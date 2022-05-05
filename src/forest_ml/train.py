from pathlib import Path
from joblib import dump

import click
import pandas as pd
import sys
import os
import warnings
import pandas_profiling



from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
# from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
# from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold 
from .data import get_dataset
from .data import get_data
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
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
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


def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    test_split_ratio: float,
    use_scaler: bool,
    max_iter: int,
    logreg_c: float,
    kf_part: int,
) -> None:
    # features_train, features_val, target_train, target_val = get_dataset(
    #     dataset_path,
    #     random_state,
    #     test_split_ratio)
    
    features, target = get_dataset(dataset_path)

    pipeline = create_pipeline(use_scaler, max_iter, logreg_c, random_state)


    kf = StratifiedKFold(n_splits=kf_part, random_state=None)
    
    acc_score = []
    roc_score = []
    f_score = []
    
    for train_index , test_index in kf.split(features, target):
        X_train , X_test = features.iloc[train_index,:],features .iloc[test_index,:]
        y_train , y_test = target[train_index] , target[test_index]
        if not sys.warnoptions:
            warnings.simplefilter("ignore")
            os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
            pipeline.fit(X_train,y_train)

        pred_values = pipeline.predict(X_test)
        
        acc = accuracy_score(pred_values , y_test)
        roc_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test), multi_class='ovr', average="macro")
        f_measure = f1_score(y_test, pred_values,  average='macro')

        roc_score.append(roc_auc)
        acc_score.append(acc)
        f_score.append(f_measure)
    
    dump(pipeline, save_model_path)
    click.echo(f"Model is saved to {save_model_path}.")

    click.echo(f"Roc auc score {sum(roc_score)/len(roc_score)}.") 
    click.echo(f"Accuracy score {sum(acc_score)/len(acc_score)}.") 
    click.echo(f"F score {sum(f_score)/len(f_score)}.") 

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
    show_default=True
)

def eda(
    dataset_path: Path,
    ) -> None:

    data = get_data(dataset_path)

    data = get_data(dataset_path)
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
        profile = data.profile_report(title='Pandas Profiling Report')
    # profile.to_file(outputfile="data/profiling.html")
    # print(data)
    profile.to_file("Forest_report.html")

    # data.profile_report()