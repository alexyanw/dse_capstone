import logging,pprint
from enum import Enum
import pandas as pd
import numpy as np
from utils import *

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error

#from modeldb.sklearn_native.ModelDbSyncer import *
#from modeldb.sklearn_native import SyncableMetrics

logger = logging.getLogger('dp')

class ModelType(Enum):
    NORMAL = 1
    GRID_SEARCH = 2
    BAYESIAN = 3

class ModelManager:
    def __init__(self, df, features, target, model, **kwargs):
        self.X = df[features]
        self.y = np.ravel(df[target])
        self.model = model
        self.modeldb = kwargs.get('modeldb', False)
        self.metrics = kwargs.get('metrics', mean_squared_error)

        mname = type(model).__name__
        if mname == 'GridSearchCV':
            mname = type(model.estimator).__name__
            self.type = ModelType.GRID_SEARCH
        else:
            self.type = ModelType.NORMAL

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
            y_predict = self.model.predict_sync(self.X_test)
        else:
            y_predict = self.model.predict(self.X_test)
        self.test_error = self.measure_metrics(y_predict, self.y_test)

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

    def summary(self, **kwargs):
        if not self.predicted:
            logger.error("model not trained yet")
            return

        if self.type == ModelType.GRID_SEARCH:
            logger.info('best params: {}'.format(self.model.best_params_))
            logger.info('best score: {}'.format(self.model.best_score_))
            logger.debug('cv results:\n{}'.format(pprint.pformat(self.model.cv_results_)))

        logger.info('training error: {}'.format(self.train_error))
        logger.info('testing error: {}'.format(self.test_error))

