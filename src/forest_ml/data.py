import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
import click
from sklearn.model_selection import train_test_split

def get_dataset(
    csv_path: Path, 
    random_state: int, 
    test_split_ratio: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 

    data = pd.read_csv(csv_path)

    click.echo(f"Dataset shape: {data.shape}.")

    features = data.drop(columns=["Cover_Type"])    
    target = data['Cover_Type']

    features_train, features_val, target_train, target_val = train_test_split(
        features, target, test_size=test_split_ratio, random_state=random_state
    )

    return features_train, features_val, target_train, target_val


