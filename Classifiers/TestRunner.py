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

def fit_decision_tree(x, y, param_grid, cv=5, scoring='accuracy') -> DecisionTreeClassifier:
    grid_search_decision_tree = GridSearchCV(estimator=DecisionTreeClassifier(),
                                             param_grid=param_grid,
                                             cv=cv,
                                             scoring=scoring)

    grid_search_decision_tree.fit(x, y)
    return grid_search_decision_tree.best_estimator_, grid_search_decision_tree.best_params_

def fit_svm(x, y, param_grid, cv=5, scoring='accuracy') -> SVC:
    grid_search_svm = GridSearchCV(estimator=SVC(),
                                   param_grid=param_grid,
                                   cv=cv,
                                   scoring=scoring)
    grid_search_svm.fit(x, y)
    return grid_search_svm.best_estimator_, grid_search_svm.best_params_
