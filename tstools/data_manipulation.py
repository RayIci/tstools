import pandas as pd

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