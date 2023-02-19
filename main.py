import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score
from joblib import dump
import matplotlib.pyplot as plt

# load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# define the model and parameter grid
model = DecisionTreeClassifier(random_state=42)
param_space = {
    "criterion": ["gini", "entropy"],
    "splitter": ["best", "random"],
    "max_depth": [None] + list(np.arange(1, 20)),
    "min_samples_split": list(np.arange(2, 21)),
    "min_samples_leaf": list(np.arange(1, 21)),
}

# define the nested cross-validation method
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# define the evaluation metric
scorer = make_scorer(accuracy_score)

# define the grid search object
grid_search = RandomizedSearchCV(
    model, param_space, n_iter=1000, scoring="accuracy", n_jobs=-1, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
)

# fit the grid search object using the nested cross-validation method
scores = []
for train_index, test_index in outer_cv.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # fit the grid search object
    grid_search.fit(X_train, y_train)
    
    # evaluate the model on the test set and store the results
    y_pred = grid_search.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    scores.append(score)

# summarize the performance of the model
print(f'Scores: {scores}')
print(f'Mean Accuracy: {sum(scores)/len(scores):.3f}')

# fit the model using the best hyperparameters and save the model
best_model = DecisionTreeClassifier(max_depth=grid_search.best_params_['max_depth'], random_state=42)
best_model.fit(X, y)

# visualize the decision tree
fig, ax = plt.subplots(figsize=(12, 8))
plot_tree(best_model, filled=True, feature_names=iris.feature_names, class_names=iris.target_names, ax=ax)
plt.show()


dump(best_model, 'best_model.joblib')
