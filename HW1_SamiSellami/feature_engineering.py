import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        df = X.copy()
        if 'ret' in df.columns:
            df = df.drop(columns=['ret'])
        # Do feature engineering as in transform
        for col in ['macro2', 'firm1', 'firm2', 'firm3']:
            if col in df.columns:
                df[f'{col}_lag1'] = df.groupby('firm_id')[col].shift(1)
                df[f'{col}_lag2'] = df.groupby('firm_id')[col].shift(2)
                df[f'{col}_lag3'] = df.groupby('firm_id')[col].shift(3)
                df[f'{col}_ma3'] = df.groupby('firm_id')[col].transform(lambda x: x.rolling(3).mean())
                df[f'{col}_ma5'] = df.groupby('firm_id')[col].transform(lambda x: x.rolling(5).mean())
                df[f'{col}_ma7'] = df.groupby('firm_id')[col].transform(lambda x: x.rolling(7).mean())
        if 'price' in df.columns:
            df['price_lag1'] = df.groupby('firm_id')['price'].shift(1)
            df['price_ma3_ratio'] = df['price'] / df.groupby('firm_id')['price'].transform(lambda x: x.rolling(3).mean())
            df['price_ma5_ratio'] = df['price'] / df.groupby('firm_id')['price'].transform(lambda x: x.rolling(5).mean())
            df['price_ma7_ratio'] = df['price'] / df.groupby('firm_id')['price'].transform(lambda x: x.rolling(7).mean())
        if 'macro1' in df.columns:
            for cat in ['Expansion', 'Contraction', 'Recovery', 'Peak', 'Trough']:
                df[f'macro1_{cat}'] = (df['macro1'] == cat).astype(int)
        if 'macro2' in df.columns:
            df['macro2^2'] = df['macro2'] ** 2
        for col in ['firm1', 'firm2', 'firm3']:
            if col in df.columns:
                df[f'{col}_rank'] = df.groupby('date')[col].rank(pct=True)
                df[f'{col}_cross_zscore'] = df.groupby('date')[col].transform(lambda x: (x - x.mean()) / x.std())
        if 'date' in df.columns:
            df['month'] = pd.to_datetime(df['date']).dt.month
        df = df.fillna(0)
        keep = ['firm_id']
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist() + keep
        num_cols = list(set(num_cols) & set(df.columns))
        df = df[num_cols]
        self._col_order = df.columns.tolist()
        return self

    def transform(self, X):
        df = X.copy()
        if 'ret' in df.columns:
            df = df.drop(columns=['ret'])
        for col in ['macro2', 'firm1', 'firm2', 'firm3']:
            if col in df.columns:
                df[f'{col}_lag1'] = df.groupby('firm_id')[col].shift(1)
                df[f'{col}_lag2'] = df.groupby('firm_id')[col].shift(2)
                df[f'{col}_lag3'] = df.groupby('firm_id')[col].shift(3)
                df[f'{col}_ma3'] = df.groupby('firm_id')[col].transform(lambda x: x.rolling(3).mean())
                df[f'{col}_ma5'] = df.groupby('firm_id')[col].transform(lambda x: x.rolling(5).mean())
                df[f'{col}_ma7'] = df.groupby('firm_id')[col].transform(lambda x: x.rolling(7).mean())
        if 'price' in df.columns:
            df['price_lag1'] = df.groupby('firm_id')['price'].shift(1)
            df['price_ma3_ratio'] = df['price'] / df.groupby('firm_id')['price'].transform(lambda x: x.rolling(3).mean())
            df['price_ma5_ratio'] = df['price'] / df.groupby('firm_id')['price'].transform(lambda x: x.rolling(5).mean())
            df['price_ma7_ratio'] = df['price'] / df.groupby('firm_id')['price'].transform(lambda x: x.rolling(7).mean())
        if 'macro1' in df.columns:
            for cat in ['Expansion', 'Contraction', 'Recovery', 'Peak', 'Trough']:
                df[f'macro1_{cat}'] = (df['macro1'] == cat).astype(int)
        if 'macro2' in df.columns:
            df['macro2^2'] = df['macro2'] ** 2
        for col in ['firm1', 'firm2', 'firm3']:
            if col in df.columns:
                df[f'{col}_rank'] = df.groupby('date')[col].rank(pct=True)
                df[f'{col}_cross_zscore'] = df.groupby('date')[col].transform(lambda x: (x - x.mean()) / x.std())
        if 'date' in df.columns:
            df['month'] = pd.to_datetime(df['date']).dt.month
        df = df.fillna(0)
        keep = ['firm_id']
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist() + keep
        num_cols = list(set(num_cols) & set(df.columns))
        df = df[num_cols]
        # Reorder columns to match fit
        if hasattr(self, '_col_order'):
            missing = [c for c in self._col_order if c not in df.columns]
            for c in missing:
                df[c] = 0
            df = df[self._col_order]
        return df 