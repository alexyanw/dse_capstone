import pandas as pd
import types

__all__ = ['Preprocess']

def transform_YN(dfs,dft,col):
    return dfs[col].apply(lambda x: 1 if x=='Y' else 0)

def get_average_volumn(dfs, dft, region='zip'):
    return 0

def get_average_sqft_price(dfs, dft, region='zip'):
    return 0

class Preprocess:
    target = 'sqft_price'
    features_delivered = [
        'sqft',
        'num_bed',
        'num_bath',
        'view',
        'pool',
        'date',
    ]
    features_engineering = [
        'year_built',
        'sold_year',
        'sold_age',
        'sold_price',
    ]
    feature_func = {
        'year_built': int,
        'view': transform_YN,
        'pool': transform_YN,
        'date': lambda dfs,dft,col: pd.to_datetime(dfs[col]),
        'sold_year': lambda dfs,dft,col: pd.to_datetime(dfs['date']).dt.year,
        'sold_age': lambda dfs,dft,col: pd.to_datetime(dfs['date']).dt.year - dft['year_built'],
        'avg_volumn_street': get_average_volumn,
        'avg_sqft_price_street': get_average_sqft_price,
    }

    valid_criterias = {
        'sold_price': lambda df: df[df['sold_price']>0],
        'sqft_price': lambda df: df[(df['sqft_price']>0)&(df['sqft_price']<2000)],
        'sqft': lambda df: df[df['sqft']<10000],
        'num_bed': lambda df: df[df['num_bed']<10],
        'num_bath': lambda df: df[df['num_bath']<10],
    }

    def __init__(self, df, **kwargs):
        self.df = df

    def dataset(self, **kwargs):
        feature_set_all = Preprocess.features_delivered + Preprocess.features_engineering + [Preprocess.target]
        feature_set = feature_set_all
        if kwargs.get('feature', 'all') == 'delivered':
            feature_set = Preprocess.features_delivered + [Preprocess.target]

        exist_columns = list(set(self.df.columns) & set(feature_set_all))
        direct_features = [f for f in feature_set if f not in Preprocess.feature_func]
        engineer_features = [f for f in feature_set if f in Preprocess.feature_func]

        df_ret = self.df[exist_columns].copy(deep=True)
        for f in engineer_features:
            func = Preprocess.feature_func[f]
            print('converting', f)
            if isinstance(func, types.FunctionType):
                df_ret[f] = Preprocess.feature_func[f](self.df, df_ret, f)
            else:
                print(df_ret.columns)
                df_ret[f] = df_ret[f].fillna(0).astype(func)

        if kwargs.get('valid', False):
            df_ret = self.remove_invalid(df_ret)

        return df_ret[feature_set]

    def get_valid_count(self):
        valid_counts = {}
        for f,func in Preprocess.valid_criterias.items():
            valid_counts[f] = func(self.df).shape[0]
        valid_counts['total'] = self.df.shape[0]
        return valid_counts

    def remove_invalid(self, df):
        df_ret = df
        for f,func in Preprocess.valid_criterias.items():
            df_ret = func(df_ret)
        return df_ret
