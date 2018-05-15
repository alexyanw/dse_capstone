import logging,pprint
import pandas as pd
import numpy as np
import types
from functools import partial
from copy import copy
from utils import *

__all__ = ['Preprocess']

logger = logging.getLogger('dp')

def transform_YN(pp,df,col):
    return df[col].apply(lambda x: 1 if x=='Y' else 0)

def get_aggregation(pp, df_to_agg, df_to_join, col, **kwargs):
    if 'feature' not in kwargs or 'group' not in kwargs or 'aggfunc' not in kwargs:
        logger.error("Error: get_aggregation must specify feature,group,aggfunc in kwargs")
        exit(1)
    feature = kwargs['feature']
    group = kwargs['group']
    aggfunc = kwargs['aggfunc']
    suffix = '_' + aggfunc
    df_agg = df_to_agg.groupby(group)[feature].agg(aggfunc).reset_index()
    df_agg = df_agg.rename(index=str, columns={feature: col})
    df_joined = df_to_join.merge(df_agg, on=group, suffixes=('', suffix))
    return df_joined

class Preprocess:
    features_delivered = [
        'pin',
        'date',
        'sqft',
        'num_bed',
        'num_bath',
        'view',
        'pool',
        'sqft_zip_avg',
        'sqft_over_zip_avg',
        'sqft_price_zip_avg',
        'sold_price_zip_avg',
        'impr_over_land',
        'min_elem_distance',
        'min_middle_distance',
        'min_high_distance',
        'elem_rating',
        'middle_rating',
        'high_rating',
        'avg_elem_rating',
        'avg_middle_rating',
        'avg_high_rating',
        'lon',
        'lat',
        #'zip',
        'eval_zip_avg',
        'eval_over_zip_avg',
    ]
    features_underwork = [
        'zip',
        'usable_sqft',
        #'acre',
        'street',
        'year_built',
        'sold_price',
        'sqft_price',
        'sold_year',
        'sold_month',
        'sold_age',
        'sale_count_zip',
        #'prop_count_zip',
        'eval_land',
        'eval_imps',
        'eval',
        'eval_sqft_price',
    ]
    feature_transform = {
        #'year_built': int,
        'view': transform_YN,
        'pool': transform_YN,
        'date': lambda self,df,col: pd.to_datetime(df[col]),
        'sold_year': lambda self,df,col: pd.to_datetime(df['date']).dt.year,
        'sold_month': lambda self,df,col: pd.to_datetime(df['date']).dt.month,
        'sold_age': lambda self,df,col: pd.to_datetime(df['date']).dt.year - df['year_built'],
        'impr_over_land': lambda self,df,col: df['eval_imps'] / df['eval_land'],
        'eval': lambda self,df,col: df['eval_imps'] + df['eval_land'],
        'eval_sqft_price': lambda self,df,col: df['eval'] / df['sqft'],
        'min_elem_distance': lambda self,df,col: df[col].fillna(df[col].max()),
        'min_middle_distance': lambda self,df,col: df[col].fillna(df[col].max()),
        'min_high_distance': lambda self,df,col: df[col].fillna(df[col].max()),
        'elem_rating': lambda self,df,col: df[col].fillna(0),
        'middle_rating': lambda self,df,col: df[col].fillna(0),
        'high_rating': lambda self,df,col: df[col].fillna(0),
        'avg_elem_rating': lambda self,df,col: df[col].fillna(0),
        'avg_middle_rating': lambda self,df,col: df[col].fillna(0),
        'avg_high_rating': lambda self,df,col: df[col].fillna(0),
    }
    feature_engineer = {
        'sqft_price_zip_median': partial(get_aggregation, feature='sqft_price', group='zip', aggfunc='median'),
        'sold_price_zip_median': partial(get_aggregation, feature='sold_price', group='zip', aggfunc='median'),
        'sqft_zip_avg': partial(get_aggregation, feature='sqft', group='zip', aggfunc='mean'),
        'sqft_price_zip_avg': partial(get_aggregation, feature='sqft_price', group='zip', aggfunc='mean'),
        'sold_price_zip_avg': partial(get_aggregation, feature='sold_price', group='zip', aggfunc='mean'),
        'eval_zip_avg': partial(get_aggregation, feature='eval', group='zip', aggfunc='mean'),
        'sale_count_zip': partial(get_aggregation, feature='date', group='zip', aggfunc='count'),
        'prop_count_zip': partial(get_aggregation, feature='pin', group='zip', aggfunc='count'),
        'sqft_over_zip_avg': lambda self,df,col: df['sqft'] / df['sqft_zip_avg'],
        'eval_over_zip_avg': lambda self,df,col: df['eval'] / df['eval_zip_avg'],
        #'turnover_zip':
    }

    valid_criterias = {
        'sold_price': lambda df: df[(df['sold_price']>0)&(df['sold_price']<3000000)],
        'sqft_price': lambda df: df[(df['sqft_price']>0)&(df['sqft_price']<2000)],
        'sqft': lambda df: df[df['sqft']<10000],
        'num_bed': lambda df: df[df['num_bed']<10],
        'num_bath': lambda df: df[df['num_bath']<10],
        'lon': lambda df: df[df['lon'].notnull()],
       #'view': lambda df: df[df['view'].notnull()],
       #'pool': lambda df: df[df['pool'].notnull()],
       #'year_built': lambda df: df[df['year_built'].notnull()],
    }

    def __init__(self, df_transaction=None, df_property=None, **kwargs):
        self.df_transaction = df_transaction
        self.df_property = df_property
        self.target = kwargs.get('target', 'sold_price')

        if 'source' in kwargs:
            ds = kwargs['source']
            tran_view = kwargs.get('transaction', 'property_address_transactions')
            prop_view = kwargs.get('property', 'property_addresses')
            self.df_transaction = ds.get_view_df(tran_view)
            self.df_property = ds.get_view_df(prop_view)
        self.df_transaction['id'] = df_transaction.index

    def dataset(self, **kwargs):
        feature_set_all = Preprocess.features_delivered + Preprocess.features_underwork + [self.target] + ['id']
        feature_set = feature_set_all
        if kwargs.get('feature', 'all') == 'delivered':
            feature_set = Preprocess.features_delivered + [self.target]
        if kwargs.get('feature_set', None) is not None:
            feature_set = list(set(kwargs['feature_set'] + [self.target]))
        feature_set += ['id']

        exist_trans_columns = list(set(self.df_transaction.columns) & set(feature_set_all))
        df_ret = self.df_transaction[exist_trans_columns].copy(deep=True)

        trans_features = [f for f in feature_set_all if f in Preprocess.feature_transform]
        eng_features = [f for f in feature_set_all if f in Preprocess.feature_engineer]

        for f in trans_features:
            logger.debug('transforming {}'.format(f))
            func = Preprocess.feature_transform[f]
            if isinstance(func, types.FunctionType) or isinstance(func, partial):
                df_ret[f] = func(self, df_ret, f)
                if df_ret[f].dtype in ['float64', 'int64']:
                    df_ret[f] = df_ret[f].fillna(df_ret[f].mean())
            else:
                #df_ret[f] = df_ret[f].fillna(0).astype(func)
                df_ret[f] = df_ret[f].astype(func)

        if 'date' in kwargs:
            df_ret = self.filter_date(df_ret, kwargs['date'])

        for f in eng_features:
            logger.debug('making {}'.format(f))
            func = Preprocess.feature_engineer[f]
            if isinstance(func, partial):
                df_ret = func(self, df_ret, df_ret, f)
            else:
                df_ret[f] = func(self, df_ret, f)

        if kwargs.get('valid', False):
            df_ret = self.remove_invalid(df_ret, feature_set)

        if kwargs.get('clean', False):
            df_ret = self.remove_outlier(df_ret)

        feature_set_available = list(set(feature_set) & set(df_ret.columns))
        return df_ret.sort_values('date')[feature_set_available]

    def make_prop_transaction(self, begin, end):
        begin_date, end_date = pd.to_datetime(begin), pd.to_datetime(end)
        months = (end_date.year - begin_date.year) * 12 + end_date.month - begin_date.month
        #NOTE: hardcode test_window to 4
        test_window = 4
        df_transactions = []
        for m in range(0, int(months/test_window)):
            date = begin_date + pd.tseries.offsets.DateOffset(months=m*test_window)
            df_tran = self.df_property.copy(deep=True)
            df_tran['date'] = date
            df_transactions.append(df_tran)

        return pd.concat(df_transactions)

    def gen_dataset(self, **kwargs):
        logger.info('preprocess dataset - transaction: {}'.format(self.df_transaction.shape))
        feature_set_all = Preprocess.features_delivered + Preprocess.features_underwork + [self.target] + ['id']
        feature_set = feature_set_all
        if kwargs.get('feature', 'all') == 'delivered':
            feature_set = Preprocess.features_delivered + [self.target]
        if kwargs.get('feature_set', None) is not None:
            feature_set = list(set(kwargs['feature_set'] + [self.target]))
        feature_set += ['id']

        exist_trans_columns = list(set(self.df_transaction.columns) & set(feature_set_all))
        df_ret = self.df_transaction[exist_trans_columns].copy(deep=True)
        if 'date' in kwargs:
            df_ret = self.filter_date(df_ret, kwargs['date'])
        df_ret = df_ret.copy(deep=True)

        df_predict = None
        if 'date1' in kwargs:
            df_predict = self.make_prop_transaction(*kwargs['date1'])
            exist_prop_columns = list(set(df_predict.columns) & set(feature_set_all))
            df_predict = df_predict[exist_prop_columns].copy(deep=True)

        trans_features = [f for f in feature_set_all if f in Preprocess.feature_transform]
        eng_features = [f for f in feature_set_all if f in Preprocess.feature_engineer]

        for f in trans_features:
            logger.debug('transforming {}'.format(f))
            func = Preprocess.feature_transform[f]
            if isinstance(func, types.FunctionType) or isinstance(func, partial):
                df_ret[f] = func(self, df_ret, f)
                if df_ret[f].dtype in ['float64', 'int64']:
                    df_ret[f] = df_ret[f].fillna(df_ret[f].mean())
                if 'date1' in kwargs:
                    df_predict[f] = func(self, df_predict, f)
                    if df_predict[f].dtype in ['float64', 'int64']:
                        df_predict[f] = df_predict[f].fillna(df_predict[f].mean())
            else:
                #df_ret[f] = df_ret[f].fillna(0).astype(func)
                df_ret[f] = df_ret[f].astype(func)
                if 'date1' in kwargs: df_predict[f] = df_predict[f].astype(func)

        for f in eng_features:
            logger.debug('making {}'.format(f))
            func = Preprocess.feature_engineer[f]
            if isinstance(func, partial):
                df_ret = func(self, df_ret, df_ret, f)
                if 'date1' in kwargs: df_predict = func(self, df_ret, df_predict, f)
            else:
                df_ret[f] = func(self, df_ret, f)
                if 'date1' in kwargs: df_predict[f] = func(self, df_predict, f)

        if kwargs.get('valid', False):
            df_ret = self.remove_invalid(df_ret, feature_set)
            df_predict = self.remove_invalid(df_predict, feature_set)

        if kwargs.get('clean', False):
            df_ret = self.remove_outlier(df_ret)

        feature_set_available = list(set(feature_set) & set(df_ret.columns))
        if df_predict is None:
            return df_ret.sort_values('date')[feature_set_available], df_predict
        else:
            feature_set_predict = list(set(feature_set) & set(df_predict.columns))
            return df_ret.sort_values('date')[feature_set_available], df_predict[feature_set_predict]

    @classmethod
    def get_feature_list(cls, type='delivered'):
        if type == 'delivered':
            return copy(Preprocess.features_delivered)
        elif type == 'underwork':
            return copy(Preprocess.features_underwork)
        else:
            return Preprocess.features_delivered + Preprocess.features_underwork

    def get_valid_count(self):
        valid_counts = {}
        for f,func in Preprocess.valid_criterias.items():
            valid_counts[f] = func(self.df_transaction).shape[0]
        valid_counts['total'] = self.df_transaction.shape[0]
        return valid_counts

    def remove_invalid(self, df, features):
        df_ret = df
        for f,func in Preprocess.valid_criterias.items():
            if f not in df.columns: continue
            logger.debug("cleaning on criteria: {}".format(f))
            df_ret = func(df_ret)
        return df_ret

    def remove_outlier(self, df, upper=.75, lower=.75, column='sqft_price'):
        limits = self.get_zip_stats(df, upper=upper, lower=lower, column=column)
        zips = df['zip'].unique()
        dfs = []
        for stat in limits:
            zip = stat['zip']
            df_zip = df[df['zip']==zip]
            lower  = stat['lower']
            upper = stat['upper']
            dfs.append(df_zip[(df_zip[column]>lower)&(df_zip[column]<upper)])

        return pd.concat(dfs)

    def get_zip_stats(self, df, column='sqft_price', upper=1.5, lower=1.5):
        q75, q25 = np.percentile(df[column], [75 ,25])
        iqr = q75 - q25
        county_min = q25 - (iqr*upper)
        county_max = q75 + (iqr*lower)

        stats = []
        zips = df['zip'].unique()
        for zip in zips:
            df_zip = df[df['zip']==zip]
            mean = df_zip[column].mean()
            std = df_zip[column].std()
            median = df_zip[column].median()
            q75, q25 = np.percentile(df_zip[column], [75 ,25])
            iqr = q75 - q25
            zip_min = q25 - (iqr*upper)
            zip_max = q75 + (iqr*lower)
           #if df_zip.shape[0] < 100:
           #    zip_min = county_min
           #    zip_max = county_max
            stats.append({'zip': zip, 'lower':zip_min, 'upper':zip_max, 'iqr':iqr, 'mean':mean, 'std':std, 'median':median, 'count':df_zip.shape[0]})
        return stats

    def filter_date(self, df, date_range):
        if len(date_range) != 2:
            logger.error("date_range must be size of 2")
            exit(1)
        df['date'] = pd.to_datetime(df['date'])
        return df[(df['date'] >= date_range[0]) & (df['date'] < date_range[1])]

    def debug(self, df_check):
        cols_to_use = list(self.df_transaction.columns.difference(df_check.columns)) + ['id']
        df_ret = df_check.merge(self.df_transaction[cols_to_use], on='id')
        debug_columns = ['pin', 'str_no', 'street', 'st_type', 'city', 'zip', 'land_use_subcode'] + list(df_check.columns)
        return df_ret[debug_columns]
