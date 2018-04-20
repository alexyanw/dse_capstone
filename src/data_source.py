import os,sys
import re
import logging,pprint
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from sqlalchemy import create_engine, text
from utils import *

__all__ = ['DataSource']

logger = logging.getLogger('dp')

class DataSource:
    supported_view = [
        'properties',
        'property_features',
        'transactions',
        'property_addresses',
        'property_address_transactions',
        'property_transaction_valid',
        'property_address_school_transactions',
    ]

    @classmethod
    def show_views(cls):
        logger.info('supported views: {}'.format(DataSource.supported_view))

    def __init__(self, **kwargs):
        dotenv_path = find_dotenv()
        self.rootdir = os.path.dirname(dotenv_path)
        load_dotenv(dotenv_path)

        if 'engine' in kwargs:
            self.engine = kwargs['engine']
        else:
            database_url = ''
            if 'host' in kwargs:
                database_url = "postgresql://{username}:{password}@{host}:{port}/{database}".format(**kwargs)
            else:
                database_url = os.environ.get("DATABASE_URL")

            logger.info("connect to database {}".format(database_url))
            self.engine = create_engine(database_url)

    def get_view_df(self, view):
        if view not in DataSource.supported_view:
            logger.error("view not supported")
            exit(1)

        sql = '''SELECT * FROM {}'''.format(view)

        return pd.read_sql_query(sql, self.engine)

    def update_views(self, **kwargs):
        sqls = []
        database_dir = os.path.join(self.rootdir, 'database')

        sql = kwargs.get('sql', None)
        if not sql:
            sql = os.path.join(database_dir, 'preprocess.sql')
        sqls.append(sql)

        if kwargs.get('materialize', False):
            sql = os.path.join(database_dir, 'materialize.sql')
            sqls.append(sql)

        for sql in sqls:
            logger.info("execute {}".format(sql))
            statement = ''
            with open(sql) as f:
                lines = f.readlines()
                statement = "".join([l for l in lines if not re.match(r'^ *--', l)])
            with self.engine.connect() as con:
                rs = con.execute(text(statement))

