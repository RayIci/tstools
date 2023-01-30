import pandas as pd
import numpy as np
import tstools.analysis as ts_analysis
import scipy.fftpack as sft
import math as math

def extract_column_as_dataframe(dataframe: pd.DataFrame, column: str):
    """
    Description
    -----------
    Extract a column from a pandas dataframe as a dataframe instead of a pandas Series
    
    Parameters
    ----------
    dataframe: pd.DataFrame
        The original padnas dataframe
    column: str
        The label of the column to extract

    Returns
    -------
    A pandas DataFrame containing the extracted column and indexed by the original index 

    """
    return pd.DataFrame(dataframe[column], index=dataframe.index)



def difference(series: pd.Series, diff_lag: int = 1):
    """
    Description
    -----------
    Extract a column from a pandas dataframe as a dataframe instead of a pandas Series
    
    Parameters
    ----------
    dataframe: pd.DataFrame
        The pandas series
    diff_lag: int
        the numbers of lags on which computing the difference (default is 1) 
        example: diff_lag = 1 compute the first difference between the current value and the value before

    Returns
    -------
    The series with the difference computed

    """
    return series.diff(diff_lag)[diff_lag:]



def count_null(dataset: pd.DataFrame):
    """
    Description
    -----------
    Display and return, for each column, the number of null values as a pandas DataFrame

    Parameters
    ----------
    dataset: pd.DataFrame
        The dataset as padas DataFrame
    disp:bool = True
        True or False if you want to display the numbers of null values

    Return
    ------
    The pandas Series conteining the numbers of null values for each column of the original dataset
    """

    # sum the numbers of null values for each column of the pandas dataframe
    null_values = dataset.isna().sum()

    # iterate over the null values (padnas Series) and insert 
    # them as a column in a new pandas DataFrame
    nullValues_df = pd.DataFrame()
    for idx in range(null_values.shape[0]):
        nullValues_df.insert(idx, null_values.index[idx], [null_values[idx]])
    
    return nullValues_df
