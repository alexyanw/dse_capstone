import logging,pprint
import pandas as pd
import numpy as np
import math
from utils import *
from plot_utils import *
from model_manage import *

logger = logging.getLogger('dp')

class ModelRegression:
    def __init__(self, features, model, **kwargs):
        self.features = features
        self.model = model
        self.test_window = 4
        self.sliding_windows = []
        self.track_window = kwargs.get('track_window', 24)
        self.validation_cycle = self.track_window / self.test_window

    def regress(self, df, **kwargs):
        kwargs['time_series'] = True
        self.start_date = df['date'].min()
        self.end_date = df['date'].max()
        months = (self.end_date.year - self.start_date.year) * 12 + (self.end_date.month - self.start_date.month)

        n = math.ceil((months - self.test_window) / self.test_window)
        results = []
        self.errors = []
        self.sliding_windows = []
        sliding_window = self.test_window
        mm = None
        for i in range(n):
            test_idx = (i+1) * self.test_window
            mm = ModelManager(None, self.features, self.model, **kwargs)
            if test_idx < self.track_window:
                sliding_window = min((i+1) * self.test_window, kwargs.get('sliding_window', 12))
            elif 'sliding_window' in kwargs:
                sliding_window = kwargs['sliding_window']
            elif test_idx % self.track_window == 0:
                mm, sliding_window = self.validate(df, test_idx, **kwargs)

            self.sliding_windows.append(sliding_window)
            end_month = (i+2) * self.test_window
            start_month = end_month - self.test_window - sliding_window
            df_track = self.get_window(df, start_month, end_month)
            logger.debug('prediction - data shape: {}, period: {} ~ {}, sliding(in month): {}, test: {}'.format(df_track.shape, df_track['date'].min().date(), df_track['date'].max().date(), sliding_window, self.test_window))
            error = mm.run(dataset=df_track, sliding_window=sliding_window, test_size=self.test_window, size_in_month=True)
            self.errors.append(error)
            results.append(mm.y_predict)
        self.results = np.concatenate(results)
        return self.errors

    def validate(self, df, test_idx, **kwargs):
        best_score = -math.inf
        mm,sliding_window = None, None
        for s in range(1, int(self.track_window/self.test_window)-2):   # make sure at least 3 folds
            mmc = ModelManager(None, self.features, self.model, **kwargs)
            start_month = test_idx - self.track_window
            end_month = start_month + self.track_window + self.test_window
            df_track = self.get_window(df, start_month, end_month)
            logger.info('validation - data shape: {}, sliding(in month): {}, period: {} ~ {}'.format(df_track.shape, s*self.test_window, df_track['date'].min().date(), df_track['date'].max().date()))
            #score = mmc.run(dataset=df_track, sliding_window=s*self.test_window, test_size=self.test_window, size_in_month=True, **kwargs)
            score = mmc.run(dataset=df_track, sliding_window=s*self.test_window, test_size=self.test_window, size_in_month=True, **kwargs)
            logger.debug('validation - sliding(in month):{}, score: {}'.format(s*self.test_window, score))
            if score > best_score:
                sliding_window = s*self.test_window
                mm = mmc
                best_score = score
        return mm, sliding_window


    def predict(self, df, df_property, **kwargs):
        return None

    def get_window(self, df, start_month, end_month):
        init_date = df['date'].min()
        start_date = init_date + pd.tseries.offsets.DateOffset(months=start_month)
        end_date = init_date + pd.tseries.offsets.DateOffset(months=end_month)
        return df[(df['date']>=start_date)&(df['date']<end_date)]

    def summary(self):
        logger.info('data range: {} - {}'.format(self.start_date.date(), self.end_date.date()))
        logger.info('sliding_windows: {}'.format(self.sliding_windows))
        logger.info('errors: {}'.format(self.errors))
        dates = [self.start_date + pd.tseries.offsets.DateOffset(months=4*i) for i in range(1, 1+len(self.errors))]
        return pd.DataFrame.from_dict({'date':dates, 'error': self.errors, 'sliding_windows': self.sliding_windows}).set_index('date')




