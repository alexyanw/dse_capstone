import pandas as pd
import types
from functools import partial

__all__ = ['Preprocess']

def transform_YN(pp,df,col):
    return df[col].apply(lambda x: 1 if x=='Y' else 0)

def get_aggregation(pp, df, col, **kwargs):
    if 'feature' not in kwargs or 'group' not in kwargs or 'aggfunc' not in kwargs:
        print("Error: get_aggregation must specify feature,group,aggfunc in kwargs")
        exit(1)
    feature = kwargs['feature']
    group = kwargs['group']
    aggfunc = kwargs['aggfunc']
    suffix = '_' + aggfunc
    df_agg = df.groupby(group)[feature].agg(aggfunc).reset_index()
    df_agg = df_agg.rename(index=str, columns={feature: col})
    df_joined = df.merge(df_agg, on=group, suffixes=('', suffix))
    return df_joined

def get_property_aggregation(pp, df, col, **kwargs):
    if 'feature' not in kwargs or 'group' not in kwargs or 'aggfunc' not in kwargs:
        print("Error: get_aggregation must specify feature,group,aggfunc in kwargs")
        exit(1)
    if pp.df_property is None:
        print("Warning: feature '{}' is missing due to preprocess doesn't have df_property".format(col))
        return df
    feature = kwargs['feature']
    group = kwargs['group']
    aggfunc = kwargs['aggfunc']
    suffix = '_' + aggfunc
    df_agg = pp.df_property.groupby(group)[feature].agg(aggfunc).reset_index()
    df_agg = df_agg.rename(index=str, columns={feature: col})
    df_joined = df.merge(df_agg, on=group, suffixes=('', suffix))
    return df_joined

class Preprocess:
    target = 'sqft_price'
    features_delivered = [
        'sqft',
        'num_bed',
        'num_bath',
        'view',
        'pool',
        'sqft_zip_avg',
        'sqft_price_zip_avg',
        'sold_price_zip_avg',
    ]
    features_underwork = [
        'date',
        'street',
        'zip',
        'year_built',
        'sold_year',
        'sold_age',
        'sold_price',
        'sale_count_zip',
        'prop_count_zip',
    ]
    feature_transform = {
        'year_built': int,
        'view': transform_YN,
        'pool': transform_YN,
        'date': lambda self,df,col: pd.to_datetime(df[col]),
        'sold_year': lambda self,df,col: pd.to_datetime(df['date']).dt.year,
        'sold_age': lambda self,df,col: pd.to_datetime(df['date']).dt.year - df['year_built'],
    }
    feature_engineer = {
        'sqft_zip_avg': partial(get_aggregation, feature='sqft', group='zip', aggfunc='mean'),
        'sqft_price_zip_avg': partial(get_aggregation, feature='sqft_price', group='zip', aggfunc='mean'),
        'sold_price_zip_avg': partial(get_aggregation, feature='sold_price', group='zip', aggfunc='mean'),
        'sale_count_zip': partial(get_aggregation, feature='date', group='zip', aggfunc='count'),
        'prop_count_zip': partial(get_property_aggregation, feature='pin', group='zip', aggfunc='count'),
        #'turnover_zip':
    }

    valid_criterias = {
        'sold_price': lambda df: df[df['sold_price']>0],
        'sqft_price': lambda df: df[(df['sqft_price']>0)&(df['sqft_price']<2000)],
        'sqft': lambda df: df[df['sqft']<10000],
        'num_bed': lambda df: df[df['num_bed']<10],
        'num_bath': lambda df: df[df['num_bath']<10],
    }

    def __init__(self, df_transaction=None, df_property=None, **kwargs):
        self.df_transaction = df_transaction
        self.df_property = df_property

        if 'source' in kwargs:
            ds = kwargs['source']
            tran_view = kwargs.get('transaction', 'property_address_transactions')
            prop_view = kwargs.get('property', 'property_addresses')
            self.df_transaction = ds.get_view_df(tran_view)
            self.df_property = ds.get_view_df(prop_view)

    def dataset(self, **kwargs):
        feature_set_all = Preprocess.features_delivered + Preprocess.features_underwork + [Preprocess.target]
        feature_set = feature_set_all
        if kwargs.get('feature', 'all') == 'delivered':
            feature_set = Preprocess.features_delivered + [Preprocess.target]

        exist_columns = list(set(self.df_transaction.columns) & set(feature_set_all))
        df_ret = self.df_transaction[exist_columns].copy(deep=True)

        trans_features = [f for f in feature_set_all if f in Preprocess.feature_transform]
        eng_features = [f for f in feature_set_all if f in Preprocess.feature_engineer]

        for f in trans_features:
            print('transforming', f)
            func = Preprocess.feature_transform[f]
            if isinstance(func, types.FunctionType) or isinstance(func, partial):
                df_ret[f] = func(self, df_ret, f)
            else:
                df_ret[f] = df_ret[f].fillna(0).astype(func)

        if 'date' in kwargs:
            df_ret = self.filter_date(df_ret, kwargs['date'])

        for f in eng_features:
            print('making', f)
            func = Preprocess.feature_engineer[f]
            df_ret = func(self, df_ret, f)

        if kwargs.get('valid', False):
            df_ret = self.remove_invalid(df_ret)

        feature_set_available = list(set(feature_set) & set(df_ret.columns))
        return df_ret[feature_set_available]

    @classmethod
    def get_feature_list(cls, type='delivered'):
        if type == 'delivered':
            return Preprocess.features_delivered
        elif type == 'underwork':
            return Preprocess.features_underwork
        else:
            return Preprocess.features_delivered + Preprocess.features_underwork

    def get_valid_count(self):
        valid_counts = {}
        for f,func in Preprocess.valid_criterias.items():
            valid_counts[f] = func(self.df_transaction).shape[0]
        valid_counts['total'] = self.df_transaction.shape[0]
        return valid_counts

    def remove_invalid(self, df):
        df_ret = df
        for f,func in Preprocess.valid_criterias.items():
            df_ret = func(df_ret)
        return df_ret

    def filter_date(self, df, date_range):
        if len(date_range) != 2:
            print("Error: date_range must be size of 2")
            exit(1)
        return df[(df['date'] >= date_range[0]) & (df['date'] < date_range[1])]
