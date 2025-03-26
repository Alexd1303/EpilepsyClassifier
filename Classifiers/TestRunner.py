from typing import Any

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def fit_classifier(classifier, x, y, param_grid, cv=5, scoring='accuracy') -> tuple[Any, Any]:
    grid_search = GridSearchCV(estimator=classifier,
                               param_grid=param_grid,
                               cv=cv,
                               scoring=scoring)

    grid_search.fit(x, y)
    return grid_search.best_estimator_, grid_search.best_params_
