import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn import cluster, mixture

class MultiSegmentRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_clusters=3, feature_set=[], cluster_features=['sold_price'], cluster_on_target=False):
        self.n_clusters = n_clusters
        self.feature_set = feature_set
        self.cluster_features = cluster_features
        self.models = [RandomForestRegressor(n_estimators=100, max_depth=14, n_jobs=-1, random_state=17) for i in range(n_clusters)]
        self.cluster_on_target = cluster_on_target

    def fit(self, X, y):
        # Check that X and y have correct shape
        self.X = X
        X_cluster = X[self.cluster_features]
        X_ = X[self.feature_set]
        X_, y_ = check_X_y(X_, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X_
        self.y_ = y_

        self.clustering = cluster.MiniBatchKMeans(n_clusters=self.n_clusters)
        #self.clustering =  mixture.BayesianGaussianMixture(n_components=self.n_clusters, covariance_type='full')
        self.clustering.fit(X_cluster)
        self.y_train_cluster = self.clustering.predict(X_cluster)

        results = []
        for i in range(self.n_clusters):
            X_subset = self.X_[self.y_train_cluster==i]
            y_subset = self.y_[self.y_train_cluster==i]
            self.models[i].fit(X_subset, y_subset)
        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X_ = X[self.feature_set]
        X_ = check_array(X_)

        self.y_predicts = []
        if self.cluster_on_target:
            for i in range(self.n_clusters):
                self.y_predicts.append(self.models[i].predict(X_))
            return self.vote_(self.y_predicts)
        else:
            self.y_cluster = self.clustering.predict(X[self.cluster_features])
            for i in range(self.n_clusters):
                self.y_predicts.append(self.models[i].predict(X_))
            return self.merge_(self.y_predicts, self.y_cluster)

    def merge_(self, y_predicts, y_cluster):
        result = []
        for i in range(len(y_cluster)):
            result.append(y_predicts[y_cluster[i]][i])
        return np.array(result) 

    def vote_(self, y_predicts):
        y_clusters = []
        for i in range(self.n_clusters):
            y_predict = y_predicts[i].reshape(-1, 1)
            y_clusters.append(self.clustering.predict(y_predict))

        columns = ['predict{}'.format(i) for i in range(self.n_clusters)] + ['cluster{}'.format(i) for i in range(self.n_clusters)]
        self.results_ = pd.DataFrame.from_items(zip(columns, y_predicts+y_clusters))

        # TODO: implement voting here
        return y_predicts[0]

    def inspect(self):
        return self.results_

