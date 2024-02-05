"""
Utilities package to help prep data
"""

from __future__ import annotations

import numpy as np
import os
import logging
from typing import Dict, Optional, Tuple
from enum import StrEnum
import yaml
from sklearn.model_selection import train_test_split

import pandas as pd


CONFIG_ENCODING = "utf8"


def load_config(config_file: Optional[str] = None) -> Dict:
    """Load config file and return Python dictionary

    Args:
        config_file: a yaml filename containing config parameters at the current path.
            If missing, will use config file based on current filename.
            e.g. file.py -> file_config.yml

    Returns:
        config: Python dictionary with config parms


    """
    current_path = os.getcwd()
    config_path = os.path.join(current_path, config_file)
    try:
        with open(config_path, "r", encoding=CONFIG_ENCODING) as f:
            config = yaml.safe_load(f)
        logging.debug("Loaded config from: %s", config_file)
        return config
    except Exception as error:
        error.add_note(f"Error reading the config file: {config_file}")
        raise


class ColumnType(StrEnum):
    """ColumnType"""

    CATEGORICAL = "categorical"
    CONTINUOUS = "continuous"
    TEXT = "text"

    @staticmethod
    def classify_column(
        col: pd.Series, categorical_max_count: int = 20, verbose=True
    ) -> ColumnType:
        """Classify a column"""
        if col.dtype in ["int64", "float64"]:
            return ColumnType.CONTINUOUS
        num_unique = len(col.unique())
        if verbose:
            print(f"The column '{col.name}' has {num_unique} unique values.")
        if num_unique <= categorical_max_count:
            return ColumnType.CATEGORICAL
        return ColumnType.TEXT

    @staticmethod
    def classify_columns(
        df: pd.DataFrame, categorical_max_count: int = 20, verbose=True
    ) -> Dict[str, ColumnType]:
        """
        Classify column in a dataframe.

        :param pd.DataFrame df: input pandas dataframe
        :param int categorical_max_count: If a non continuous column with more than
            this many unique values will be considered a TEXT instead of CATEGORYCAL.
        :param verbose: if true, will display the number of unique values by column
            names. (default: True)
        """
        return {
            col_name: ColumnType.classify_column(col, categorical_max_count, verbose)
            for col_name, col in df.items()
        }


def train_valid_test_split(
    X: np.Array, test_prop: float = 0.2, train_prop: float = 0.8
) -> Tuple[np.Array, np.Array, np.Array]:
    train_valid, test = train_test_split(X, test_size=test_prop)
    train, valid = train_test_split(train_valid, train_size=train_prop)
    return train, valid, test
