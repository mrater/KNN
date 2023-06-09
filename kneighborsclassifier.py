import numpy as np
import pandas as pd
import reader
from collections import Counter


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

    def predict(self, x_test_item):
        nearest = self.get_nearest(x_test_item)
        c = Counter([self.train_set_y.iloc[i] for i in nearest])
        result = c.most_common()[0][0]
        return result 
        

    def get_nearest(self, x_test_item):
        distances = [self.get_distance(x_test_item, x_train_item, metric=self.metric) for _, x_train_item in self.train_set_x.iterrows()]
        indices = np.argpartition(distances, self.n_neighbors)[:self.n_neighbors]
        # print(distances)
        # print(indices)
        return indices
    
        
    def calculate_accuracy(self, x_test, y_test):
        result = 0
        for i in range(len(x_test)):
            if self.predict(x_test.iloc[i]) == y_test.iloc[i]:
                result += 1
        return result, len(x_test)

    # return distance defined by euclidean metric (i.e. square difference)
    def euclidean_comparison(self, score_a, score_b) -> float:
        return (score_a - score_b)**2
    
    # return distance defined by manhattan metric 
    def manhattan_comparison(self, score_a, score_b) -> float:
        return abs(score_a - score_b)
    
    #get distance from A to B defined by metric
    def get_distance(self, point_a, point_b, metric) -> float:
        if metric not in self.AVAILABLE_METRICS:
            raise Exception('Metric {} is unknown'.format(metric))

        result = 0.0

        for feature in point_a.index:
            if metric == 'euclidean':
                result += self.euclidean_comparison(point_a[feature], point_b[feature])
            elif metric == 'manhattan':
                result += self.manhattan_comparison(point_a[feature], point_b[feature])
            
        return result