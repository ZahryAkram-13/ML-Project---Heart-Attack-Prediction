from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans


LOGISTIC_REGRESSION = 'LOGISTIC_REGRESSION'
SUPPORT_VECTOR_MACHINES = 'SVC'
KNN = 'KNeighborsClassifier'
DECISION_TREE = 'DecisionTreeClassifier'
RANDOM_FOREST = 'RandomForestClassifier'
GRADIENT_BOOSTING_MACHINES = 'GradientBoostingClassifier'
XGBOOST = 'XGBClassifier'
LIGHTGBM = 'LGBMClassifier'
KMEANS = 'KMeans'
NAIVE_BAYES = 'GaussianNB'

import xgboost as xgb
import lightgbm as lgb



MODELS = {
    # LOGISTIC_REGRESSION : LogisticRegression(),
    # SUPPORT_VECTOR_MACHINES : SVC(kernel = 'linear', C = 0.1, gamma = 'scale'),
    KNN : KNeighborsClassifier(n_neighbors = 50),
    # DECISION_TREE : DecisionTreeClassifier(max_depth = 10, min_samples_leaf = 2 ),
    # RANDOM_FOREST : RandomForestClassifier(n_estimators = 100, max_depth = 10, bootstrap = True, class_weight='balanced'),
    # GRADIENT_BOOSTING_MACHINES : GradientBoostingClassifier(),
    # XGBOOST : xgb.XGBClassifier(),
    # LIGHTGBM : lgb.LGBMClassifier(),
    # KMEANS : KMeans(),
    # NAIVE_BAYES : GaussianNB()
}

def apply_model(**kwarg):
    from sklearn.metrics import classification_report
    from plotting import show_crossvalidation_box, show_confusion_matrix, show_metrics
 
    x_train_scaled, y_train = kwarg['x_train_scaled'], kwarg['y_train']
    x_test_scaled, y_test = kwarg['x_test_scaled'],kwarg['y_test']
    model_name = kwarg['model_name']
    model = MODELS[model_name]
    
    model.fit(x_train_scaled, y_train)
    y_prediction = model.predict(x_test_scaled)

    # accuracy_score(rf_train_predict, y_test), rf.score(x_test, y_test)
    scores = show_crossvalidation_box(model, x_train_scaled, y_train)
    show_confusion_matrix(y_test, y_prediction)
    metrics = show_metrics(y_test, y_prediction)
    repport = classification_report(y_test, y_prediction)
    print(repport)

    return metrics, scores


# Hyperparamètres à tester pour chaque modèle
PARAM_GRID = {
    'SVC': {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']
    },
    'KNN': {
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'DecisionTree': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'RandomForest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
}



