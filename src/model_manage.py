import logging,pprint
from enum import Enum
import pandas as pd
import numpy as np
import itertools
from utils import *
from plot_utils import *

from sklearn.model_selection import train_test_split, TimeSeriesSplit
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

def remeasure(df_check, threshold=200000):
    removed = df_check[df_check['residual']>=threshold]
    print("removed records:", removed.shape[0])
    df_left = df_check[df_check['residual']<threshold]
    return sqrt(mean_squared_error(df_left['predict'], df_left['sold_price']))

def get_valid_columns(df):
    extra_columns = ['id', 'pin', 'date']
    return df[list(set(df.columns) - set(extra_columns))]


def walk_forward_split(n_folds, test_window, sliding):
    splits = []
    m = 0
    for i in range(n_folds):
        start = i*test_window
        train = np.array(range(start, start+sliding))
        test = np.array(range(start+sliding, start+sliding+test_window))
        splits.append([train, test])
    return splits

class ModelManager:
    def __init__(self, df, features, model, **kwargs):
        self.target = kwargs.get('target', 'sold_price')
        self.feature_set = features
        self.model = model
        self.modeldb = kwargs.get('modeldb', False)
        self.metrics = kwargs.get('metrics', lambda y_act,y_pred: sqrt(mean_squared_error(y_act, y_pred)))
        self.time_series = kwargs.get('time_series', False)

        mname = type(model).__name__
        if mname == 'Pipeline':
            mname = '->'.join([type(s[1]).__name__ for s in model.steps])
            self.type = ModelType.PIPELINE
        else:
            self.type = ModelType.NORMAL
        self.model_name = mname

        self._predicted = False
        if self.modeldb:
            self._init_modeldb(**self.modeldb)

        if df is not None: self.set_data(df)

    def set_data(self, df):
        self.df = df.copy(deep=True)
        self.X = self.df[self.feature_set+['id']]
        self.y = np.ravel(self.df[self.target])

    def _init_modeldb(self, **kwargs):
        name = kwargs.get('name', "house sold_price estimate")
        author = kwargs.get('author', 'wenyan')
        description = kwargs.get('description', 'modeling for sdra')
        self.modeldb_syncer = Syncer(
                NewOrExistingProject(name, author, description),
                DefaultExperiment(),
                NewExperimentRun("Abc"))

    def split(self, **kwargs):
        X_train_val, X_train, X_test, y_train_val, y_train, y_test = None, None, None, None, None, None
        test_size = kwargs.get('test_size', 0.2)
        if kwargs.get('size_in_month', False):
            test_start_date = self.X['date'].max() - pd.tseries.offsets.DateOffset(months=test_size)
            test_size_in_sample = self.X[self.X['date'] > test_start_date].shape[0]
            X_train_val, X_test = self.X[:-test_size_in_sample], self.X[-test_size_in_sample:]
            y_train_val, y_test = self.y[:-test_size_in_sample], self.y[-test_size_in_sample:]
        elif type(test_size) is float:
            split_index = int(self.X.shape[0] * (1-test_size))
            X_train_val, X_test = self.X[:split_index], self.X[split_index:]
            y_train_val, y_test = self.y[:split_index], self.y[split_index:]
        else:
            X_train_val, X_test = self.X[:-test_size], self.X[-test_size:]
            y_train_val, y_test = self.y[:-test_size], self.y[-test_size:]

        if self.time_series:
            self.test_window = X_test.shape[0]
            self.track_window = X_train_val.shape[0]

            sliding_window = kwargs.get('sliding_window', None)
            if sliding_window:
                if kwargs.get('size_in_month', False):
                    train_end_date = X_train_val['date'].max() - pd.tseries.offsets.DateOffset(months=test_size)
                    train_start_date = train_end_date - pd.tseries.offsets.DateOffset(months=sliding_window)
                    valid_size = X_train_val[(X_train_val['date'] > train_end_date)].shape[0]
                    self.sliding_window = X_train_val[(X_train_val['date'] > train_start_date) & (X_train_val['date'] <= train_end_date)].shape[0]
                else:
                    valid_size = test_size
                    self.sliding_window = sliding_window
                X_train, X_val = X_train_val[-valid_size-self.sliding_window:-valid_size], X_train_val[-valid_size:]
                y_train, y_val = y_train_val[-valid_size-self.sliding_window:-valid_size], y_train_val[-valid_size:]
            else:
                valid_size = kwargs.get('valid_size', 0.2)
                if type(valid_size) is float:
                    split_index = int(X_train_val.shape[0] * (1-valid_size))
                    X_train, X_val = X_train_val[:split_index], X_train_val[split_index:]
                    y_train, y_val = y_train_val[:split_index], X_train_val[split_index:]
                else:
                    X_train, X_val = X_train_val[:-valid_size], X_train_val[-valid_size:]
                    y_train, y_val = y_train_val[:-valid_size], y_train_val[-valid_size:]
                self.sliding_window = X_train.shape[0]

            logger.info("track/sliding/test window size: {}, {}, {}".format(self.track_window, self.sliding_window, self.test_window))
        else:
            valid_size = kwargs.get('valid_size', 0.2)
            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=valid_size, random_state=5)
            logger.debug("train/valid/test size: {}, {}, {}".format(X_train.shape[0], X_val.shape[0], X_test.shape[0]))

        self.X_train_val = X_train_val
        self.y_train_val = y_train_val
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

    def validate(self, **kwargs):
        param_grid = kwargs.get('param_grid', None)
        if param_grid == None:
            logger.debug("validation step is skipped")
            return

        if 'GridSearchCV' in kwargs:
            self.type = ModelType.GRID_SEARCH
            gs_args = kwargs.get('GridSearchCV')
            gs_args['scoring'] = 'neg_mean_squared_error'
            if self.time_series:
                n_splits = int(self.track_window / self.test_window + 0.5) - 1
                tscv = TimeSeriesSplit(max_train_size=self.sliding_window, n_splits=n_splits)
                gs_args['cv'] = tscv
                i = 0
                for train, test in tscv.split(self.X_train_val):
                    logger.debug("%s-fold walk-forward train:%s, test:%s, scope: %s ~ %s" % (i+1, train.shape, test.shape, train[0], test[-1]))
                    i += 1
                #n_splits = int((self.track_window - self.sliding_window) / self.test_window)
                #wfs = walk_forward_split(n_splits, self.test_window, self.sliding_window)
                #gs_args['cv'] = wfs
                #for train, test in wfs:
                #    logger.debug("walk-forward train:%s, test:%s, scope:%s ~ %s" % (train.shape, test.shape, train[0], test[-1]))
                #wfs = _split(self.track_window, self.test_window, self.sliding_window)
                #gs_args['cv'] = wfs
                #for train, test in wfs:
                #    logger.debug("walk-forward train:%s, test:%s, scope:%s ~ %s" % (train.shape, test.shape, train[0], test[-1]))
                self.model = GridSearchCV(self.model, param_grid, **gs_args)
            else:
                self.model = GridSearchCV(self.model, param_grid, **gs_args)
                fold = gs_args.get('cv', 3)
                logger.info("{}-fold cross validation: train/valid size {}".format(fold, int(self.X_train_val.shape[0]/fold)))

            X_train = get_valid_columns(self.X_train_val)
            y_train = self.y_train_val
            logger.debug('training size: {}'.format(X_train.shape[0]))
            self.model.fit(X_train, y_train)
            return self.model.best_score_
        elif 'BayesianOptimization' in kwargs:
            return
        else: # (not work) manual validation with train/valid dataset
            if self.time_series:
                params = param_grid.keys()
                scores = []
                for values in itertools.product(*param_grid.values()):
                    param_values = dict(zip(params, values))
                    estimator = self.model.set_params(**param_values)
                    estimator.fit(get_valid_columns(self.X_train), self.y_train)
                    y_pred = estimator.predict(get_valid_columns(self.X_val))
                    score = -self.measure_metrics(self.y_val, y_pred)
                    scores.append({'param': param_values, 'score': score})

                best_params = max(scores, key=lambda x: x['score'])
            else:
                return

            self.model = self.model.set_params(**best_params['param'])
            return best_params['score']


    def train(self, **kwargs):
        if self.time_series:
            X_train = get_valid_columns(self.X_train_val[-self.sliding_window:])
            y_train = self.y_train_val[-self.sliding_window:]
        else:
            X_train = get_valid_columns(self.X_train_val)
            y_train = self.y_train_val

        logger.info('training data size: {}'.format(X_train.shape))
        estimator = self.get_best_model()
        estimator.fit(X_train, y_train)
        y = estimator.predict(X_train)
        self.train_error = self.measure_metrics(y, y_train)

    def test(self):
        estimator = self.get_best_model()
        X_test = get_valid_columns(self.X_test)
        logger.info('test data size: {}'.format(X_test.shape))
        self.y_predict = estimator.predict(X_test)
        self.residual = self.y_predict - self.y_test
        self.test_error = self.measure_metrics(self.y_predict, self.y_test)

        self.predicted = True
        return self.test_error

    def predict(self, X_predict):
        X_test = get_valid_columns(X_predict[self.feature_set])
        logger.info('predict data size: {}'.format(X_test.shape))
        self.y_predict = self.model.predict(X_test)
        return self.train_error

    def measure_metrics(self, y_true, y_pred):
        error = self.metrics(y_true, y_pred)
        return error

    def run(self, **kwargs):
        if 'dataset' in kwargs: self.set_data(kwargs['dataset'])
        self.split(**kwargs)
        if kwargs.get('predict', None) is not None:
            self.train(**kwargs)
            return self.predict(kwargs['predict'])
        elif kwargs.get('param_grid', None):
            return self.validate(**kwargs)
        else:
            self.train(**kwargs)
            return self.test()

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
            return self.model.best_estimator_ if hasattr(self.model, 'best_estimator_') else self.model
        else:
            return self.model

    def get_result_df(self, pp=None):
        df_check = self.X_test.copy(deep=True)
        df_check[self.target] = self.y_test
        df_check['predict'] = self.y_predict
        df_check['residual'] = self.residual

        if pp != None:
            cols_to_use = list(pp.df_transaction.columns.difference(df_check.columns)) + ['id']
            return df_check.merge(pp.df_transaction[cols_to_use], on='id')
        return df_check

    def plot_feature_importance(self, **kwargs):
        if not self.check_predicted(): return
        #features = kwargs.get('feature_set', self.feature_set)
        features = list(get_valid_columns(self.X_test).columns)
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
        plt.title('Prediction vs Actual')

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
        df_learning_curve.index = df_learning_curve.index.map(str)
        plot_trends(df_learning_curve, ['mean_train_score','mean_test_score'])

        return [results['params'][i] for i in np.argsort(results['rank_test_score'])[::-1]]

