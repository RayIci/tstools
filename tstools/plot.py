import matplotlib.pyplot as plt 
import pandas as pd

def plotAll(dataframe: pd.DataFrame, plot_width : int = 20, plot_height: int = 5, tight_layout: bool = True):
    
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