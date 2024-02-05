import numpy as np
from typing import Dict, List

from keras.preprocessing.sequence import pad_sequences
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OrdinalEncoder


class FillEmpty(BaseEstimator, TransformerMixin):
    """Fill Empty Values with {laceholders"""

    def set_params(self, **kwargs):
        self.col_list: List[str] = kwargs.get("col_list", [])
        self.continuous_cols: List[str] = kwargs.get("continuous_cols", [])
        self.text_cols: List[str] = kwargs.get("text_cols", [])
        self.target_col: str = kwargs.get("target_col", [])
        return self

    def selected_cols(self) -> list[str]:
        return [
            *self.col_list,
            *self.continuous_cols,
            *self.text_cols,
            self.target_col,
        ]

    def transform(self, X, **kwargs):
        print("fill empty xform and remove unused columns")
        for col in self.col_list:
            X[col].fillna(value="missing", inplace=True)
        for col in self.continuous_cols:
            X[col].fillna(value=0.0, inplace=True)
        for col in self.text_cols:
            X[col].fillna(value="missing", inplace=True)
        return X.loc[:, self.selected_cols()]

    def fit(self, X, y=None, **kwargs):
        return self


class EncodeCategorical(BaseEstimator, TransformerMixin):
    """Encode Categorical Columns"""

    def __init__(self):
        self.label_encoders: Dict[str, OrdinalEncoder] = {}
        self.col_list: List[str] = []

    def set_params(self, **kwargs):
        self.col_list = kwargs.get("col_list", [])
        return self

    def fit(self, X, y=None, **kwargs):
        for col in self.col_list:
            print("fit column: ", col)
            self.label_encoders[col] = OrdinalEncoder(dtype=np.int64)
            self.label_encoders[col].fit(X[[col]])
        return self

    def transform(self, X, y=None, **kwargs):
        for col in self.col_list:
            print("transform column: ", col)
            X[col] = self.label_encoders[col].transform(X[[col]])
            print("after transform column: ", col)
        return X


class PrepForKeras(BaseEstimator, TransformerMixin):
    """Prepare Columns for Keras model input"""

    def __init__(self):
        self.dict_list = []
        self.col_list: List[str] = []
        self.continuous_cols: List[str] = []
        self.text_cols: List[str] = []
        self.max_dict: Dict[str, int] = {}

    def set_params(self, **kwargs):
        self.col_list = kwargs.get("col_list", [])
        self.continuous_cols = kwargs.get("continuous_cols", [])
        self.text_cols = kwargs.get("text_cols", None)
        return self

    def fit(self, X, y=None, **kwargs):
        for col in self.col_list:
            self.max_dict[col] = len(X[col].unique())
        return self

    def transform(self, X, y=None, **kwargs):
        dict_list = []
        for col in self.col_list:
            print("cat col is", col)
            dict_list.append(np.array(np.array(x) for x in X[col]))
        for col in self.text_cols:
            print("text col is", col)
            dict_list.append(pad_sequences(X[col], maxlen=self.max_dict[col]))
        for col in self.continuous_cols:
            print("cont col is", col)
            dict_list.append(np.array(X[col]))
        return np.array(dict_list)
