import logging,pprint
from enum import Enum
import pandas as pd
import numpy as np
from utils import *
from plot_utils import *

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
from math import sqrt

#from modeldb.sklearn_native.ModelDbSyncer import *
#from modeldb.sklearn_native import SyncableMetrics

logger = logging.getLogger('dp')

class ModelType(Enum):
    NORMAL = 1
    PIPELINE = 2
    GRID_SEARCH = 3
    BAYESIAN = 4

class ModelManager:
    def __init__(self, df, features, target, model, **kwargs):
        self.feature_set = features
        self.X = df[features]
        self.y = np.ravel(df[target])
        self.model = model
        self.modeldb = kwargs.get('modeldb', False)
        self.metrics = kwargs.get('metrics', lambda y_act,y_pred: sqrt(mean_squared_error(y_act, y_pred)))

        mname = type(model).__name__
        if mname == 'GridSearchCV':
            mname = type(model.estimator).__name__
            self.type = ModelType.GRID_SEARCH
            if mname == 'Pipeline':
                mname = '->'.join([type(s[1]).__name__ for s in model.estimator.steps])
        elif mname == 'Pipeline':
            mname = '->'.join([type(s[1]).__name__ for s in model.steps])
            self.type = ModelType.PIPELINE
        else:
            self.type = ModelType.NORMAL
        self.model_name = mname

        self._predicted = False
        if self.modeldb:
            self._init_modeldb(**self.modeldb)

    def _init_modeldb(self, **kwargs):
        name = kwargs.get('name', "house sold_price estimate")
        author = kwargs.get('author', 'wenyan')
        description = kwargs.get('description', 'modeling for sdra')
        self.modeldb_syncer = Syncer(
                NewOrExistingProject(name, author, description),
                DefaultExperiment(),
                NewExperimentRun("Abc"))

    def split(self, **kwargs):
        X_train, X_test, y_train, y_test = None, None, None, None
        if self.modeldb:
            X_train, X_test, y_train, y_test = train_test_split_sync(self.X, self.y, test_size=0.2, random_state=5)
        else:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=5)
        self.X_train_val = X_train
        self.y_train_val = y_train
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = X_test
        self.y_test = y_test

    def validate(self, **kwargs):
        if self.type == ModelType.GRID_SEARCH:
            return

    def train(self):
        if self.modeldb:
            self.model.fit_sync(self.X_train_val, self.y_train_val)
        else:
            self.model.fit(self.X_train_val, self.y_train_val)

    def predict(self):
        y = self.model.predict(self.X_train_val)
        self.train_error = self.measure_metrics(y, self.y_train_val)

        if self.modeldb:
            self.y_predict = self.model.predict_sync(self.X_test)
        else:
            self.y_predict = self.model.predict(self.X_test)
        self.residual = self.y_predict - self.y_test
        self.test_error = self.measure_metrics(self.y_predict, self.y_test)

        self.predicted = True
        return self.test_error

    def measure_metrics(self, y_true, y_pred):
        error = self.metrics(y_true, y_pred)
        return error

    def run(self, **kwargs):
        self.split(**kwargs.get('split', {}))
        self.validate(**kwargs.get('validate', {}))
        self.train()
        return self.predict()

    def check_predicted(self):
        if not self.predicted:
            logger.error("model not trained yet")
            return False
        return True

    def summary(self, **kwargs):
        if not self.check_predicted(): return

        if self.type == ModelType.GRID_SEARCH:
            logger.info('best params: {}'.format(self.model.best_params_))
            logger.info('best score: {}'.format(self.model.best_score_))
            logger.debug('cv results:\n{}'.format(pprint.pformat(self.model.cv_results_)))

        logger.info('training error: {}'.format(self.train_error))
        logger.info('testing error: {}'.format(self.test_error))

    def get_best_model(self):
        if self.type == ModelType.GRID_SEARCH:
            return self.model.best_estimator_
        else:
            return self.model

    def plot_feature_importance(self, **kwargs):
        if not self.check_predicted(): return
        features = kwargs.get('feature_set', self.feature_set)
        best_model = self.get_best_model()
        if not hasattr(best_model, 'feature_importances_'):
            logger.warn("{} has no feature_importances_".format(self.model_name))
            return
        importances = pd.DataFrame({'Feature':features, 'Importance':best_model.feature_importances_})
        importances = importances.sort_values('Importance',ascending=False).set_index('Feature')
        importances.iloc[::-1].plot(kind='barh',legend=False)
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance by ' + self.model_name)
        plt.show()

    def plot_residual(self, **kwargs):
        if not self.check_predicted(): return
        fig = plt.figure(figsize=(18, 9))
        ax1 = fig.add_subplot(121)
        ax1.scatter(self.y_test, self.y_predict, s=10, c='b', alpha=0.2, marker="s", label=self.model_name)
        plt.xlabel('Actual')
        plt.ylabel('Prediction')
        plt.legend(loc='upper left');
        plt.title('Predication vs Actual')

        ax2 = fig.add_subplot(122)
        ax2.scatter(self.y_test, self.residual, s=10, c='r', alpha=0.2, marker="s", label=self.model_name)
        plt.xlabel('Actual')
        plt.ylabel('Residual')
        plt.legend(loc='best');
        plt.title('Residual vs Actual')

        plt.show()

    def plot_learning_curve(self, **kwargs):
        results = self.model.cv_results_
        learning = {k:results[k] for k in ['mean_train_score','mean_test_score','rank_test_score']}
        df_learning = pd.DataFrame.from_dict(learning)
        df_learning_curve = df_learning.set_index('rank_test_score').sort_index(ascending=False)
        plot_trends(df_learning_curve, ['mean_train_score','mean_test_score'])

        return [results['params'][i] for i in np.argsort(results['rank_test_score'])[::-1]]
