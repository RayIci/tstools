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