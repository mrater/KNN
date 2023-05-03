import numpy as np
import pandas as pd
import reader

class KNeighborsClassifier:
    train_set_x : pd.DataFrame
    train_set_y : pd.DataFrame
    AVAILABLE_METRICS = ['manhattan', 'euclidean']

    n_neighbors : int
    metric : str

    def __init__(self, n_neighbors, metric) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric

    def fit(self, x_train, y_train):
        self.train_set_x = x_train
        self.train_set_y = y_train

    def predict(self, x_test : pd.DataFrame):
        y_test = np.zeros(len(x_test.index))
        for x in x_test:
            X_with_distance = self.train_set_x.copy()
            X_with_distance["_distance"] = [self.get_distance(i, x) for i in X_with_distance]
            X_with_distance.sort_values("_distance", inplace=True)
            X_with_distance = X_with_distance[:self.n_neighbors].value_counts(X_with_distance.iloc[:,-1])

    def calculate_acuracy(x_test, y_test):
        pass

    # return distance defined by euclidean metric (i.e. square difference)
    def euclidean_comparison(self, score_a, score_b):
        return abs(score_a**2 - score_b**2)
    
    # return distance defined by manhattan metric 
    def manhattan_comparison(self, score_a, score_b):
        return abs(score_a - score_b)
    
    #get distance from A to B defined by metric
    def get_distance(self, point_a, point_b, metric):
        if metric not in self.AVAILABLE_METRICS:
            raise Exception('Metric {} is unknown'.format(metric))

        result = 0.0
        for feature_a, feature_b in zip(point_a, point_b):
            if metric == 'euclidean':
                result += self.euclidean_comparison(feature_a, feature_b)
            elif metric == 'manhattan':
                result += self.manhattan_comparison(feature_a, feature_b)
            
        return result