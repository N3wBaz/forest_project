import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
import click
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier


def get_dataset(
    csv_path: Path, 
    feature_select: int,
    # random_state: int, 
    # test_split_ratio: float
    # ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
    ) -> Tuple[pd.DataFrame, pd.Series]: 

    data = pd.read_csv(csv_path)

    features = data.drop(columns=["Cover_Type"])    
    target = data['Cover_Type']

    if feature_select == 1:
        clf = ExtraTreesClassifier(n_estimators=50)
        clf = clf.fit(features, target)
        model = SelectFromModel(clf, prefit=True)

        print("Feature selection 1")

    if feature_select == 2:

        model = BorutaPy(RandomForestClassifier(max_depth=10), 
                                n_estimators='auto', 
                                verbose=0, 
                                max_iter=100,
                                random_state=42)

        model.fit(np.asarray(features), np.asarray(target))
        print("Feature selection 2")
    
    if feature_select == 0:
        print("Feature selection doesn't active")
        return features, target

    return model.transform(np.asarray(features)), target


# for EDA
def get_data(
    csv_path: Path
    ) -> Tuple[pd.DataFrame]:

    data = pd.read_csv(csv_path)

    return data

