import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_bar(data, title='', ylabel='value', ascending=None, total=None, column=None):
    ticks = []
    values = []
    if isinstance(data, dict):
        ticks = data.keys()
        values = data.values()
    elif isinstance(data, pd.core.series.Series):
        if ascending is not None: data = data.sort_values(ascending=ascending)
        ticks = list(data.index)
        values = list(data.values)
    elif isinstance(data, pd.core.frame.DataFrame):
        if ascending is not None: data = data.sort_values(by=column)
        ticks = list(data.index)
        values = list(data[column])

    x_pos = np.arange(len(ticks))

    fig, ax = plt.subplots(figsize=(18,8))
    rects = ax.bar(x_pos, values, align='center')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ticks, rotation='vertical')
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if not total: total = max(values)
    for i, v in enumerate(values):
        rect = rects[i]
        ax.text(rect.get_x() + rect.get_width()/2., 0.25 + rect.get_height(), "{:.2f}%".format(100*v/total), ha='center', va='bottom')
    plt.show()

def plot_feature_histogram(df, columns=None, title='histogram per feature'):
    if not columns:
        columns = df.columns
    columns = [col for col in columns if df[col].dtype in ['float64', 'int64']]
    rows = int((len(columns)+2)/3)
    fig, ax = plt.subplots(rows, 3, figsize=(15,3*rows))
    for i in range(rows):
        for j in range(3):
            idx = i*3+j
            if idx >= len(columns): break
            df.hist(column=columns[idx], bins=100, ax=ax[i,j])
    plt.show()

def plot_histogram(df, index_column, hist_column, title=None):
    if not title: title = "histgram " + hist_column
    df_indexed = df
    if index_column: df_indexed = df.set_index(index_column)
    ax = df_indexed[hist_column].plot(kind='bar', title = title, figsize=(15, 8), legend=True, fontsize=12, secondary_y=True)
    ax.set_xlabel(index_column, fontsize=12)
    ax.set_ylabel(hist_column, fontsize=12)
    plt.show()

def plot_histogram_group(df, index_column, hist_columns):
    fig, ax = plt.subplots(len(hist_columns), 1, figsize=(15,10))
    df_indexed = df.set_index(index_column)
    for i,col in enumerate(hist_columns):
        df_indexed[col].plot(ax=ax[i], kind='bar', title ="histgram " + col, legend=True, fontsize=12)
    ax[i].set_xlabel(index_column, fontsize=12)
    ax[i].set_ylabel(col, fontsize=12)
    plt.show()

def plot_histogram_group_scaled(df, index_column, hist_columns, title=None):
    plt.figure(figsize=(15, 6))
    fig, ax = plt.subplots()
    df_scaled = df.set_index(index_column)
    legends = hist_columns.copy()
    for i,col in enumerate(hist_columns):
        legends[i] += ' - max:' + str(df_scaled[col].max())
        df_scaled[col] /= float(df_scaled[col].max())

    df_scaled[hist_columns].plot.bar(figsize=(10,6), title=title, ax=ax)
    ax.legend(legends);
    plt.show()

def plot_curve(df, x_col, y_col):
    plt.figure(figsize=(10, 6))
    plt.plot(df[x_col], df[y_col])
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()

def plot_trends(df, columns, **kwargs):
    fig, ax = plt.subplots(figsize=(18,12))
    for col in columns:
        df[col].plot(ax=ax, kind='line', label=col)
    plt.legend(loc='best')
    title = kwargs.get('title', None)
    if not title:
        title = "trend over " + df.index.name
    plt.title(title)
    plt.show()

import operator
def plot_correlation(df_in, target, columns, plot=True):
    mean_values = df_in.mean(axis=0)
    df_in_new = df_in.fillna(mean_values)
    corrs = {}
    for col in columns:
        corrs[col] = np.corrcoef(df_in_new[col].values, df_in_new[target].values)[0,1]

    if not plot:
        return corrs

    columns_sorted = []
    values = []
    for key, value in sorted(corrs.items(), key=operator.itemgetter(1)):
        columns_sorted.append(key)
        values.append(value)

    idx = np.arange(len(corrs))
    width, height = 12, 0.3 * len(columns)
    fig, ax = plt.subplots(figsize=(width,height))
    rects = ax.barh(idx, np.array(values), color='b')
    ax.set_yticks(idx)
    ax.set_yticklabels(columns_sorted, rotation='horizontal')
    ax.set_xlabel("Correlation coefficient")
    ax.set_title("Correlation coefficient of the variables")
    plt.show()

    return corrs


def plot_joint_dist(df_in, col, target, uc=100, lc=0, ut=100, lt=0):
    ulimit = np.percentile(df_in[col].values, uc)
    llimit = np.percentile(df_in[col].values, lc)
    df1 = df_in[df_in[col]<=ulimit]
    df1 = df1[df1[col]>=llimit]

    ulimit = np.percentile(df_in[target].values, ut)
    llimit = np.percentile(df_in[target].values, lt)
    df2 = df1[df1[target]<=ulimit]
    df2 = df2[df2[target]>=llimit]

    plt.figure(figsize=(10,10))
    sns.jointplot(x=df2[col].values, y=df2[target].values, size=10, color=color[1])
    plt.ylabel(target, fontsize=12)
    plt.xlabel(col, fontsize=12)
    plt.title(col + ' Vs ' + target, fontsize=15)
    plt.show()

def plot_violin(df, x, y, order=None, title=None):
    plt.figure(figsize=(18,8))
    sns.violinplot(x=x, y=y, data=df, order=order)
    plt.xlabel(x, fontsize=15, rotation='vertical')
    plt.xticks(rotation=90)
    plt.ylabel(y, fontsize=15)
    if not title: title = "dist of {} per {}".format(y, x)
    plt.title(title)
    plt.show()

def plot_box(df, x, y, order=None, title=None):
    plt.figure(figsize=(18,8))
    ordered_columns = None
    if order:
        ordered_columns = list(df.groupby(x)[y].agg('median').sort_values().index)
    sns.boxplot(x=x, y=y, data=df, order=ordered_columns)
    plt.xlabel(x, fontsize=15, rotation='vertical')
    plt.xticks(rotation=90)
    plt.ylabel(y, fontsize=15)
    if not title: title = "boxplot of {} per {}".format(y, x)
    plt.title(title)
    plt.show()

def plot_missing(df):
    df_missing = df.isnull().sum(axis=0).reset_index()
    df_missing.columns = ['column_name', 'missing_count']
    df_missing = df_missing.sort_values(by='missing_count')

    idx = np.arange(df_missing.shape[0])
    width = 0.9
    fig, ax = plt.subplots(figsize=(12,18))
    rects = ax.barh(idx, df_missing.missing_count.values, color='blue')
    ax.set_yticks(idx)
    ax.set_yticklabels(df_missing.column_name.values, rotation='horizontal')
    ax.set_xlabel("Count of missing values")
    ax.set_title("Number of missing values in each column")
    plt.show()


def get_hist_on_price(engine, transaction, target, min_price, max_price, buckets, **kwargs):
    condition = ''
    if 'city' in kwargs:
        condition += "AND lower(city) = '{}'".format(kwargs['city'])
    if 'zip' in kwargs:
        condition += "AND zip = '{}'".format(kwargs['zip'])

    options = {'target':target, 'scope':transaction, 'begin':min_price, 'end':max_price, 'condition':condition, 'buckets':buckets}
    sql = '''
    WITH 
    trans AS (
    {scope}
    ),
    trans_valid AS (
        -- (1) Filter out the property with price > $2,000,000
        SELECT *
        FROM trans
        WHERE {target}::decimal(10,1) between {begin} and {end} {condition}
    ),
    p_minmax AS (
        -- (2) Find the min and max price of the current distribution
        SELECT MIN({target}::DECIMAL(10,1)) AS min_price,
           MAX({target}::DECIMAL(10,1)) AS max_price
        FROM trans_valid
    )
    -- (3) The data in the range min_price to max_price is divided into 10 + 1 buckets. Find the bucket to which the price of each product belongs to. And group them by buckets.
    SELECT WIDTH_BUCKET(sold_price::DECIMAL(10,1), min_price, max_price, {buckets}) AS bucket,
          MIN({target}::DECIMAL(10,1)) as min_price, MAX({target}::DECIMAL(10,1)) as max_price,
          COUNT(1) AS count,
          SUM({target}) AS total_value
    FROM trans_valid, p_minmax
    GROUP BY bucket
    ORDER BY bucket
    '''.format(**options)
    if 'debug' in kwargs:
        print(sql)
    return pd.read_sql_query(sql, engine)
