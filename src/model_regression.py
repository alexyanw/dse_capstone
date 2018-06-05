import logging,pprint
import pandas as pd
import numpy as np
import math
from utils import *
from plot_utils import *
from model_manage import *
from preprocessing import *

logger = logging.getLogger('dp')

class ModelRegression:
    def __init__(self, features, model, **kwargs):
        self.features = features
        self.model = model
        self.test_window = 4
        self.sliding_windows = []
        self.track_window = kwargs.get('track_window', 24)
        self.validation_cycle = self.track_window / self.test_window

    def regress(self, df_or_pp, **kwargs):
        predict = kwargs.get('predict', False)
        kwargs['time_series'] = True
        if 'begin' in kwargs:
            self.start_date = pd.to_datetime(kwargs.get('begin'))
            self.end_date = pd.to_datetime(kwargs.get('end'))
        if isinstance(df_or_pp, pd.core.frame.DataFrame):
            self.start_date = df_or_pp['date'].min()
            self.end_date = df_or_pp['date'].max()
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
                mm, sliding_window = self.validate(df_or_pp, test_idx, **kwargs)

            self.sliding_windows.append(sliding_window)
            end_month = (i+2) * self.test_window
            start_month = end_month - self.test_window - sliding_window
            df_track, df_prop_trans = self.get_window(df_or_pp, start_month, end_month, start_month1=start_month+sliding_window, predict=predict)
            logger.info('train and predict - period: {} ~ {}, sliding(in month): {}, test: {}, data shape: {}'.format(df_track['date'].min().date(), df_track['date'].max().date(), sliding_window, self.test_window, df_track.shape))
            error = mm.run(dataset=df_track, sliding_window=sliding_window, test_size=self.test_window, size_in_month=True, predict=df_prop_trans)
            self.errors.append(error)
            if predict:
                df_predict = df_prop_trans[['pin', 'date']].copy()
                df_predict['sold_price'] = mm.y_predict
                results.append(df_predict)
            else:
                results.append(mm.y_predict)
        if predict:
            self.results = pd.concat(results)
        else:
            self.results = np.concatenate(results)

        return self.errors

    def validate(self, df_or_pp, test_idx, **kwargs):
        best_score = -math.inf
        mm,sliding_window = None, None
        for s in range(1, int(self.track_window/self.test_window)-2):   # make sure at least 3 folds
            mmc = ModelManager(None, self.features, self.model, **kwargs)
            start_month = test_idx - self.track_window
            end_month = start_month + self.track_window + self.test_window
            df_track,dummy = self.get_window(df_or_pp, start_month, end_month)
            logger.info('validation - data shape: {}, sliding(in month): {}, period: {} ~ {}'.format(df_track.shape, s*self.test_window, df_track['date'].min().date(), df_track['date'].max().date()))
            #score = mmc.run(dataset=df_track, sliding_window=s*self.test_window, test_size=self.test_window, size_in_month=True, **kwargs)
            score = mmc.run(dataset=df_track, sliding_window=s*self.test_window, test_size=self.test_window, size_in_month=True, **kwargs)
            logger.debug('validation - sliding(in month):{}, score: {}'.format(s*self.test_window, score))
            if score > best_score:
                sliding_window = s*self.test_window
                mm = mmc
                best_score = score
        return mm, sliding_window


    def get_window(self, df_or_pp, start_month, end_month, **kwargs):
        init_date = self.start_date
        start_date = init_date + pd.tseries.offsets.DateOffset(months=start_month)
        end_date = init_date + pd.tseries.offsets.DateOffset(months=end_month)
        if isinstance(df_or_pp, pd.core.frame.DataFrame):
            return df_or_pp[(df_or_pp['date']>=start_date)&(df_or_pp['date']<end_date)], None
        else:
            if kwargs.get('predict', False):
                start_date1 = init_date + pd.tseries.offsets.DateOffset(months=kwargs.get('start_month1', start_month))
                #return df_or_pp.gen_dataset(feature='delivered', valid=True, clean=True, date=(start_date, end_date))
                return df_or_pp.gen_dataset(feature='delivered', valid=True, clean=True, date=(start_date, end_date), date1=(start_date1, end_date))
            else:
                return df_or_pp.dataset(feature='delivered', valid=True, clean=True, date=(start_date, end_date)), None

    def summary(self):
        #logger.info('data range: {} - {}'.format(self.start_date.date(), self.end_date.date()))
        #logger.info('sliding_windows: {}'.format(self.sliding_windows))
        #logger.info('errors: {}'.format(self.errors))
        dates = [self.start_date + pd.tseries.offsets.DateOffset(months=4*i) for i in range(1, 1+len(self.errors))]
        return pd.DataFrame.from_dict({'date':dates, 'error': self.errors, 'sliding_windows': self.sliding_windows}).set_index('date')

