from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import PowerTransformer




def create_pipeline(
    use_scaler: bool, max_iter: int, logreg_C: float, random_state: int, other_model: bool,
    criterion: str, splitter: str, max_depth: int
) -> Pipeline:
    pipeline_steps = []

    if use_scaler:

        pipeline_steps.append(("scaler", PowerTransformer()))

    if other_model:

        pipeline_steps.append(
            (
                "classifier",
                DecisionTreeClassifier(
                    splitter=splitter,
                    criterion=criterion,
                    max_depth=max_depth,
                    random_state=random_state, 
                ),
            )
        )
    else:
        pipeline_steps.append(
            (
                "classifier",
                LogisticRegression(
                    random_state=random_state, max_iter=max_iter, C=logreg_C, 
                ),
                # KNeighborsClassifier(n_neighbors=100, metric='hamming')
                # braycurtis
                # 0.9209816963265073.  canberra
                # 
                # braycurtis  0.9178923422292442.


            )
        )        

    return Pipeline(steps=pipeline_steps)
