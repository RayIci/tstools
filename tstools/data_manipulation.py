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


# i dati sono suddivisi a gruppi di 3 nel seso che se suddividessimo il dataset in gruppi di 3
# la prima colonna sarebbe relativa alla posizione sell'asse x di quel giunto, la seconda
# colonna sarebbe relativa alla posizione sull'asse y di quel giunto mentre l'ultima
# colonna sarebbe relativa alla likelihood
def rename_columns(dataset: pd.DataFrame):

    # punti di giunto
    joints = ['naso', 'torace', 'spalla_dx', 
    'gomito_dx',    'polso_dx',     'spalla_sx', 
    'gomito_sx',    'polso_sx',     'cresta_iliaca', 
    'anca_dx',      'ginocchio_dx', 'caviglia_dx', 
    'anca_sx',      'ginocchio_sx', 'caviglia_sx', 
    'occhio_dx',    'occhio_sx',    'zigomo_dx', 'zigomo_sx', 
    'piede_sx_1',   'piede_sx_2',   'piede_sx_3', 
    'piede_dx_1',   'piede_dx_2',   'piede_dx_3']

    # create the new names for the columns
    new_columns = []
    for idx, joint in enumerate(joints):
        new_columns.insert(idx*3 + 0, 'x_'+joint)
        new_columns.insert(idx*3 + 1, 'y_'+joint)
        new_columns.insert(idx*3 + 2, 'l_'+joint)

    # chek that the number of columns of the dataset match the new number of columns
    if dataset.shape[1] != len(new_columns):
        raise Exception('Il numero delle colonne del dataset di origine non è corretto')
    
    # set the new columns for the dataset
    dataset.columns = new_columns 
    
    
# Smoothing Dataframe
def smooth_dataframe(dataframe: pd.DataFrame):

    new_dataframe = pd.DataFrame()

    for idx, column_name in enumerate(dataframe):
        smoothed_series = ts_analysis.smooth_exponential(dataframe[column_name], 0.25).smoothed
        new_dataframe = pd.concat([new_dataframe, pd.DataFrame({column_name: smoothed_series})], axis=1)

    return new_dataframe


# Butterworth low pass filter
def low_pass_filter(X, n_order, cutoff_freq):
    return 1 / (1 + np.power(X/cutoff_freq, 2*n_order))


def intspace(range_len):
    space = []
    for i in range(range_len):
        space.append(i)

    return np.array(space)


# Filtering Dataframe
def filter_dataframe(dataframe: pd.DataFrame):
    new_dataframe = pd.DataFrame()

    for idx, column_name in enumerate(dataframe):
        series = delete_nan(dataframe[column_name])
        series_fft = sft.fft(series)
        series_fft = series_fft * low_pass_filter(intspace(len(series_fft)), 20, 70)
        series_ifft = np.abs( sft.ifft(series_fft) )
        
        new_dataframe = pd.concat([new_dataframe, pd.DataFrame({column_name: series_ifft})], axis=1)

    return new_dataframe

# Funzione che elimina i valori nulli da una singola lista
def delete_nan(series):
    new_series = []
    for idx, value in enumerate(series):
        if not math.isnan(value):
            new_series.append(value)
    return new_series

# Funzione che elimina i valori nulli dal dataset
def delete_nan_from_DataFrame(dataframe: pd.DataFrame):

    new_dataframe = pd.DataFrame()

    for idx, column_name in enumerate(dataframe.keys()):
        new_dataframe = pd.concat([new_dataframe, pd.DataFrame({column_name: delete_nan(dataframe[column_name])})], axis=1)

    return new_dataframe