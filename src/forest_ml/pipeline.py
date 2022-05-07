from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler




def create_pipeline(
    use_scaler: bool, max_iter: int, logreg_C: float, random_state: int, other_model: bool,
    criterion: str, splitter: str, max_depth: int
) -> Pipeline:
    pipeline_steps = []

    if use_scaler:
        pipeline_steps.append(("scaler", MinMaxScaler()))
    



    if other_model:
        # print(criterion)
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
            )
        )        

    return Pipeline(steps=pipeline_steps)
