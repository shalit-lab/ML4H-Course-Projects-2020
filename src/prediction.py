from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

models = {'SVM': SVC(kernel="rbf"), 'RandomForest': RandomForestClassifier(max_depth=10),
          'DecisionTrees': DecisionTreeClassifier(), 'KNN': KNeighborsClassifier(n_neighbors=8, weights='distance')}


class Model:
    """
    Machine learning model for the different tasks
    """
    def __init__(self, model):
        if model not in models:
            raise(NotImplementedError(f"Support only the {list(models.keys())}"))
        self.model = models[model]

    def fit(self, train_x, train_y):
        self.model.fit(train_x, train_y)

    def predict(self, x):
        return self.model.predict(x)

    def score(self, test_x, test_y, options):
        for x in options:
            if x not in ('precision', 'recall', 'F1', 'AUC'):
                raise(NotImplementedError("Supports only precision, recall, F1-score and AUC"))
        scores = {}
        predictions = self.predict(test_x)
        if 'precision' in options:
            scores['precision'] = precision_score(predictions, test_y)
        if 'recall' in options:
            scores['recall'] = recall_score(predictions, test_y)
        if 'F1' in options:
            scores['F1-score'] = f1_score(predictions, test_y)
        if 'AUC' in options:
            scores['AUC'] = roc_auc_score(predictions, test_y)
        return scores
