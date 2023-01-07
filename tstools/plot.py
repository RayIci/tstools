import matplotlib.pyplot as plt 
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose


def plot_all(
    dataframe: pd.DataFrame, 
    plot_width : int = 20, 
    plot_height: int = 5, 
    tight_layout: bool = True
    ):
    
    """
    Description
    -----------
    Plot every column of the pandas DataFrame, more in the specific plot each column
    that can be described (columns returned by the pandas.DataFrame.describe() method).

    Parameters
    ----------
    dataFrame: pandas.DataFrame
        The pandas DataFrame to be plotted
    plot_width: int
        The width of the plot figure (default is 20)
    plot_height: int
        The height of a single plot (default is 5)
    tight_layout: bool
        Enable or disable the plot tight layout (default is True)
    
    """
    
    # get the numbers of columns
    col = dataframe.describe().shape[1]

    # determin the single plot size
    figsize = (plot_width, plot_height * col)

    # index used for the position of the plot on the figure
    plot_position = 1

    # iterate on each column that can be described and plot his graphic
    for col_name_index in dataframe.describe().columns.values.tolist():
        plt.subplot(col, 1, plot_position)
        plt.title(col_name_index)
        dataframe[col_name_index].plot(figsize=figsize)
        plot_position += 1

    if tight_layout:
        plt.tight_layout()

    plt.show()


def plot_multiple(
    series: list, 
    plot_width: int = 20,
    plot_height: int = 5,
    titles: list = [],
    tight_layout: bool = True
    ):

    """
    Description
    -----------
    Plot a list of series

    Parameters
    ----------
    series: list of pandas.Series
        the series to be plotted
    plot_width: int = 20
        the width of the plot
    plot_height: int = 5
        the height of the plot of a single seires
    titles: list = []
        the tilte list associated to each series
    tight_layout: bool = True
        True of Fase if you want a tight layout or not
    """

    # number of total plots
    n_plots = len(series)

    # set the total figure size
    plt.figure(figsize=(plot_width, plot_height * n_plots))

    # iterate over the series list and plot each graph in the right position
    # given by the index
    for idx, series in enumerate(series):
        
        plt.subplot(n_plots, 1, idx + 1)
        
        try:
            plt.title(titles[idx])
        except Exception as e:
            plt.title("")

        plt.plot(series)


    if tight_layout:
        plt.tight_layout()
    plt.show()


def plot_acf_pacf(
    series: pd.Series, 
    horizontal: bool = True, 
    figsize: tuple = None, 
    pacf_method: str = "ywm"
    ):

    """
    Description
    -----------
    Plot the autocorrelation and partial autocorrelation functions of the series

    Parameters
    ----------
    series: pandas.Series
        The series as pandas series
    horizontal: bool = True
        True of False if you want the plot to be plotted horizontally or vertically
    figsize: tuple = None
        The figsize relative to the whole plot 
    pacf_method: str = "ywm"
        The method in which the partial autocorrelation function is calculated see statsmodels [documentation](https://www.statsmodels.org/dev/generated/statsmodels.graphics.tsaplots.plot_pacf.html)
        for more details
    """

    nrow = 1
    ncol = 2
    if not horizontal:
        nrow, ncol = ncol, nrow

    if figsize == None: 
        if horizontal:
            figsize = (20, 5)
        else:
            figsize = (20, 15)

    f, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=figsize)

    plot_acf(series, ax=ax[0])
    plot_pacf(series, ax=ax[1], method=pacf_method)
    plt.show()


def plot_single(
    series: list | pd.Series, 
    figsize: tuple = (20, 5), 
    subplot: tuple = None, 
    title: str = "", 
    show: bool = True,
    labels: list | str = []
    ):

    """
    Description
    -----------
    Plot a single series of a list of series into a single plot

    Parameters
    ----------
    series: list
        The pandas Series or a list of pandas Series
    figsize: tuple = (20, 5)
        The figuresize of the plot
    subplot: tuple = None
        The subplot if needed, an example can be (1, 2, 1) this make the subplot of a row with 2 column in the position 1 (row, col, pos)
    title: str = ""
        The title of the plot
    show: bool = True
        True of False if you want to show the plot or not (you maybe want to defear the show of this plot)
    labels: list = []
        The label of the series the labels of the series as a list of strings
    """

    # set the figure size of the plot
    plt.figure(figsize=figsize)
    

    # check the subplot
    try:
        if subplot != None:
            plt.subplot(subplot[0], subplot[1], subplot[2])
    except Exception as e:
        raise Exception("please give 3 positional argument for the subplot\n(row, col, plot_pos)")

    # make the series a list of series if is not
    if not isinstance(series, list):
        series = [series]
    
    # make the labels a list of string if is not
    if not isinstance(labels, list):
        labels = [labels]

    # makes the actual plot
    for idx, series in enumerate(series):
        plt.plot(series, label= labels[idx] if len(labels) > idx else "")
    
    # set the title
    plt.title(title)
    if len(labels) > 0:
        plt.legend()

    if show:
        plt.show()


def plot_seasonalDecompose(
    series: pd.Series, 
    model:str="additive", 
    period:int=None
    ):

    if period == None:
        s_dec = seasonal_decompose(series, model=model)
    else:
        s_dec = seasonal_decompose(series, model=model, period=period)

    titles = [
        "observed",
        "trend",
        "seasonal",
        "resid"
    ]

    series = [
        s_dec.observed,
        s_dec.trend,
        s_dec.seasonal,
        s_dec.resid,
    ]

    plot_multiple(series, titles=titles)