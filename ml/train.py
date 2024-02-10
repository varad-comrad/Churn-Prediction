import sklearn
from sklearn.compose import ColumnTransformer
from typing import Self
import joblib
import pandas as pd
import numpy as np
import pathlib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


data = pd.read_csv(str((pathlib.Path(__file__).parent.parent / 'dataset' / 'dataset.csv').absolute()))

def split_dataset(data: pd.DataFrame, target_col: str, test_size=0.2, random_state=None) -> tuple:
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

X_train, X_test, y_train, y_test = split_dataset(
    data, 'Churn', test_size=0.3, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
categorical_cols = X_train.select_dtypes(include=['object']).columns

encoder = OneHotEncoder(sparse_output=False)
encoder.fit(X_train[categorical_cols])
X_train_cat = pd.DataFrame(encoder.transform(X_train[categorical_cols]))
X_test_cat = pd.DataFrame(encoder.transform(X_test[categorical_cols]))

X_train_cat.columns = encoder.get_feature_names_out(categorical_cols)
X_test_cat.columns = encoder.get_feature_names_out(categorical_cols)

X_train_preprocessed_cat = pd.concat([X_train.drop(
    categorical_cols, axis=1).reset_index(drop=True), X_train_cat], axis=1)
X_test_preprocessed_cat = pd.concat([X_test.drop(
    categorical_cols, axis=1).reset_index(drop=True), X_test_cat], axis=1)

numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
scaler.fit(X_train_preprocessed_cat[numeric_cols])
X_train_num = pd.DataFrame(scaler.transform(
    X_train_preprocessed_cat[numeric_cols]))
X_test_num = pd.DataFrame(scaler.transform(
    X_test_preprocessed_cat[numeric_cols]))

X_train_num.columns = numeric_cols
X_test_num.columns = numeric_cols

X_train_preprocessed_cat[numeric_cols] = X_train_num
X_test_preprocessed_cat[numeric_cols] = X_test_num


X_train_preprocessed = X_train_preprocessed_cat
X_test_preprocessed = X_test_preprocessed_cat


class Report:
    def __init__(self, model, X_test, y_test) -> None:
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def accuracy(self) -> float:
        return accuracy_score(self.y_test, self.model.predict(self.X_test))

    def confusion_matrix(self) -> np.ndarray:
        return confusion_matrix(self.y_test, self.model.predict(self.X_test))

    def report(self) -> str:
        return classification_report(self.y_test, self.model.predict(self.X_test))

    def cross_val_score(self, cv=5) -> list:
        return cross_val_score(self.model, self.X_test, self.y_test, cv=cv)


class SKLearnModelSelection:

    selectors = {
        'grid_search': sklearn.model_selection.GridSearchCV,
        'random_search': sklearn.model_selection.RandomizedSearchCV

    }

    def __init__(self, models: list, selector: str | sklearn.model_selection._search.BaseSearchCV = 'grid_search',
                 random_state: int | list[int | None] | None = None) -> None:
        self.model_classes = models
        if not isinstance(selector, (str, sklearn.model_selection._search.BaseSearchCV)):
            raise TypeError(
                'Selector must be a string or an instance of sklearn.model_selection.BaseSearchCV')

        if isinstance(selector, str):
            try:
                self.selector = self.selectors[selector]
            except KeyError:
                raise TypeError(
                    f'Desired selector is not within the available selectors. They are:\n{*self.selectors.keys(), }')
        else:
            self.selector = selector

        self.random_state = random_state
        self.best_model = None
        self.best_params = None

    def compile(self, params,
                cv: int | None = 5,
                n_jobs: int = -1) -> Self:
        self.params = params
        self.cv = cv
        self.n_jobs = n_jobs

        return self

    def __fit_once(self, model, param):
        selector = self.selector(
            model,
            param_grid=param,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs
        )
        selector.fit(self.X_train, self.y_train)
        if self.verbose:
            self.__verbose_msg(model, selector)

        if self.keep_all_models:
            self.all_models_.append(
                (type(model).__name__, selector.best_estimator_, selector.best_params_, selector.best_score_))
        if selector.best_score_ > self.best_score:
            self.best_random_state = model.random_state
            self.best_score = selector.best_score_
            self.best_model = selector.best_estimator_
            self.best_params = selector.best_params_

    def __fit_loop(self) -> Self:
        models = []
        if isinstance(self.random_state, list):
            for random_state in self.random_state:
                models.extend([model(random_state=random_state)
                              for model in self.model_classes])
            self.params *= len(self.random_state)
        else:
            models = [model(random_state=self.random_state)
                      for model in self.model_classes]
        for model, param in zip(models, self.params):
            self.__fit_once(model, param)
        if self.keep_all_models:
            self.results = pd.DataFrame(self.all_models_, columns=[
                                        'Model', 'Best Estimator', 'Best params', 'Best score'])
        return self

    def __verbose_msg(self, model, selector):
        print(pd.DataFrame({
            'Model': [type(model).__name__],
            'Best estimator': [selector.best_estimator_],
            'Best params': [selector.best_params_],
            'Best score': [selector.best_score_]
        }),)
        print('\n' + '=' * 150)

    def fit(self, X_train, y_train, scoring: str = 'accuracy', keep_all_models: bool = False, verbose=True) -> Self:
        self.X_train = X_train
        self.y_train = y_train

        self.scoring = scoring
        self.best_score = 0
        self.keep_all_models = keep_all_models
        self.verbose = verbose
        if keep_all_models:
            self.all_models_ = []
        return self.__fit_loop()

    def build_best_model(self):
        return type(self.best_model)(random_state=self.best_random_state, **self.best_params)


params = [
    {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2, 4]
    },
    {
        'loss': ['log_loss', 'exponential'],
        'learning_rate': [0.05, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'max_depth': [1, 3, 5, 10],
        'max_leaf_nodes': [None, 5, 10, 15],
    },
    {
        'estimator': [RandomForestClassifier(random_state=42), GradientBoostingClassifier(random_state=42)],
        'n_estimators': [25, 50, 100],
        'learning_rate': [0.5, 1, 2]
    }
]



model_selector = SKLearnModelSelection([RandomForestClassifier, GradientBoostingClassifier], random_state=42).compile(params, cv=5, n_jobs=-1).fit(X_train_preprocessed, y_train, scoring='accuracy')


model_selector.best_model, model_selector.best_params, model_selector.best_score
report = Report(model_selector.best_model, X_test_preprocessed, y_test)
report.accuracy()
report.confusion_matrix()
print(report.report())

model_v3 = model_selector.build_best_model()
column_transformer = ColumnTransformer([
    ('encoder', OneHotEncoder(sparse_output=False), categorical_cols),
    ('scaler', StandardScaler(), numeric_cols),
])

pipeline = Pipeline([
    ('preprocessing', column_transformer),
    ('model', model_v3)
])

pipeline.fit(X_train, y_train)


model_path = pathlib.Path(__file__).parent.parent / 'model' / 'model.pkl'

joblib.dump(pipeline, str(model_path.absolute()))
