# datasets/__init__.py
"""
Dataset utilities for SNP package.

Provides access to example datasets used in experiments.
"""

import os
import pandas as pd

def get_housing_data():
    """
    Load California housing dataset.
    
    Returns:
        pd.DataFrame: Housing data with columns including 
                      'MedInc', 'Latitude', 'Longitude', 'MedHouseVal'
    
    Example:
        >>> import snpreg
        >>> df = snpreg.get_housing_data()
        >>> print(df.head())
    """
    data_path = os.path.join(os.path.dirname(__file__), 'housing.csv')
    return pd.read_csv(data_path)

__all__ = ['get_housing_data']
