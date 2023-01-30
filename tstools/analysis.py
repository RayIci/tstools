import statsmodels.tsa.stattools as sts
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

from tstools import plot


"""
########################################################################
#########################        Tests         #########################
########################################################################
"""

def adfuller_test(series: pd.DataFrame, maxlag: int = None, canPrint: bool = True):
    
    """
    Description
    -----------
    Run the Dickey-Fuller Test for the stationarity of a time series

    Parameters
    ----------
    series: pandas.DataFrame
        The time series as pandas DataFrame
    maxlag: int
        Maximum lag which is included in test (default value of 12*(nobs/100)^{1/4} is used when None)
    canPrint: bool
        Enable or disable the print of the test result

    Return
    ------
    adf: float
        The test statistic.
    pvalue: float
        MacKinnon’s approximate p-value based on MacKinnon (1994, 2010).
    usedlag: int
        The number of lags used.
    nobs: int
        The number of observations used for the ADF regression and calculation of the critical values.
    critical values: dict
        Critical values for the test statistic at the 1 %, 5 %, and 10 % levels. Based on MacKinnon (2010).
    icbest: float
        The maximized information criterion if autolag is not None.
    resstore: ResultStore, optional
        A dummy class with results attached as attributes.

    """

    adf = sts.adfuller(series, maxlag=maxlag)
    
    if canPrint:
        isStationary = adf[1] < 0.05
        print("******[Advanced Dickey-Fuller test]******")
        print("*                                       *")
        print("*   - Test statistic: %s\t\t*" % round(adf[0], 4))
        print("*   - P value: %s\t\t\t*" % round(adf[1],4))
        print("*   - Lags used: %s\t\t\t*" % adf[2])
        print("*   - Critical values:\t\t\t*")
        print("*       1% : {}\t\t\t*".format(round(adf[4]["1%"], 4)))
        print("*       5% : {}\t\t\t*".format(round(adf[4]["5%"], 4)))
        print("*       10%: {}\t\t\t*".format(round(adf[4]["10%"], 4)))
        print("*   [--] time series is {}stationary\t*".format("" if isStationary else "not "))
        print("*                                       *")
        print("*****************************************")

    return adf



"""
########################################################################
########################   Anomaly Detenction   ########################
########################################################################
"""

def detect_anomalies(
    series: pd.Series, 
    method = "rci", 
    
    oci_window: int = 4, 
    oci_scale: int = 1.96,
    
    rci_scale: int = 3,
    rci_model: str = "additive", 
    rci_period: int = None
):
    """
    Description
    -----------
    Detect anomalies of a series

    Parameters
    ----------
    series: pd.Series
        The pandas series
    method = "rci"
        The method in which you want to dectect the anomalies:
        -   the "oci" method is based on confidence intervals for finding the anomalies
        -   the "rci" method is similar on the oci method but instead of searching the anomalies in the orignal seires the anomalies are detected in the residual
    
    oci_window: int = 4
        A window used for smoothing the series for the detection of anomalies
    oci_scale: int = 1.96
        The confidence level 
    
    rci_scale: int = 3
        The scale used for the resid anomalies detection, tune this parameter in order to obtain different results
    rci_model: str = "additive"
        The method in which the seasonal decomosition is computed, default is "additive" (can also be "multiplicative") see more [here](https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.seasonal_decompose.html) for more details and other methods
    rci_period: int = None
        The period used for computing the seasonal decomposition, if None is left the seasonal decomposition use his default value, see more [here](https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.seasonal_decompose.html)

    Returns
    -------
    The AnomalyResult class that contains everything you need
    """
    
    match(method):
        case "oci": # original confidence interval
            return __detect_anomalies_original_ci(series, oci_window, oci_scale)
        case "rci": # resid confidence interval
            return __detect_anomalies_resid_ci(series, rci_model, rci_period, rci_scale)
    

    raise Exception("Please select a valid method")


def __detect_anomalies_original_ci(series: pd.Series, window: int = 4, scale: int = 1.96):

    # if the window is negative raise an exception
    if window < 0:
        raise Exception("the value for the window cannot be negative")

    # create the rolling mean based on the window
    rolling_mean = series.rolling(window=window).mean()

    # compute the mean absolute error between the series and the rolling mean
    mean_absolute_error = np.abs(series[window:] - rolling_mean[window:]).sum() / series[window:].shape[0]

    # compute the standard deviation of the difference between the series and the rolling mean
    deviation = np.std(series[window:] - rolling_mean[window:])

    # compute the lower and upper bound
    upper_bound = rolling_mean + (mean_absolute_error + scale * deviation)
    lower_bound = rolling_mean - (mean_absolute_error + scale * deviation)

    # find the anomalies
    anomalies_top = series[upper_bound < series]
    anomalies_bot = series[lower_bound > series]
    anomalies = pd.concat([anomalies_top, anomalies_bot])

    return AnomalyResult(series, anomalies, upper_bound, lower_bound, "oci")


def __detect_anomalies_resid_ci(series: pd.Series, model: str = "additive", period: int = None, s_scale: int = 3):

    if period == None:
        sd = seasonal_decompose(series, model=model)
    else:
        sd = seasonal_decompose(series, model=model, period=period)

    resid_mean = sd.resid.mean()
    resid_std = sd.resid.std()
    
    # create the upper and lover bound for the confidence interval
    upper_bound = resid_mean + s_scale * resid_std
    lower_bound = resid_mean - s_scale * resid_std

    # find the anomalies
    anomalies_top = series[upper_bound < sd.resid]
    anomalies_bot = series[lower_bound > sd.resid]
    anomalies = pd.concat([anomalies_top, anomalies_bot])

    return AnomalyResult(series, anomalies, upper_bound, lower_bound, "rci")



"""
########################################################################
#########################      Smoothing       #########################
########################################################################
"""

def smooth_rollingMean(series: pd.Series, window: int):
    """
    Description
    -----------
    Smooth the series by using a window and computing the mean

    Parameters
    ----------
    series: pd.Series
        The pandas series
    window: int
        The window used for performing the moothing

    Returns
    -------
    The SmoothingResult class
    """
    rm = series.rolling(window=window).mean()
    return SmoothingResult(series, rm, "smoothing rolling mean", {"window":window})


# TODO: Test
def smooth_exponential(series: pd.Series, alpha: float):
    """
    Description
    -----------
    Smooth the series by using exponetial smoothing technique

    Parameters
    ----------
    series: pd.Series
        The pandas series
    alpha: float
        float between [0.0, 1.0], smoothing parameter

    Returns
    -------
    The SmoothingResult class
    """

    if alpha < 0 or alpha > 1:
        raise Exception("The alpha value bust be a number between 0 and 1 (current value: {})".format(alpha))

    result = [series[0]]
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n - 1])
    
    return_series = pd.Series(data=result, index=series.index)

    return SmoothingResult(series, return_series, "exponential smoothing", {'aplha':alpha})


# TODO: Test
def smooth_doubleExponential(series: pd.Series, alpha: float, beta: float):
    """
    Description
    -----------
    Smooth the series by using double exponetial smoothing technique

    Parameters
    ----------
    series: pd.Series
        The pandas series
    alpha: float
        float between [0.0, 1.0], smoothing parameter for level
    beta: float
        float between [0.0, 1.0], smoothing parameter for trend

    Returns
    -------
    The SmoothingResult class
    """

    if alpha < 0 or alpha > 1:
        raise Exception("The alpha value bust be a number between 0 and 1 (current value: {})".format(alpha))

    if beta < 0 or beta > 1:
        raise Exception("The beta value bust be a number between 0 and 1 (current value: {})".format(beta))
    
    result = [series[0]]
    for n in range(1, len(series)):
        if n == 1:
            level, trend = series[0], series[1] - series[0]

        if n >= len(series):  # forecasting
            value = result[-1]

        else:
            value = series[n]

        last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        result.append(level + trend)

    return_series = pd.Series(data=result, index=series.index)

    return SmoothingResult(series, return_series, "double exponential smoothing", {'aplha':alpha, 'beta':beta})


"""
########################################################################
########################    Result calsses     #########################
########################################################################
"""

class AnomalyResult:
    """ 
    Description
    -----------
    The result given from the anomalies analysis
    """
    
    
    def __init__(self, 
    _original_series: pd.Series, 
    _anomalies: pd.Series,
    _upper_bound: pd.Series,
    _lower_bound: pd.Series,
    _method: str
    ) -> None:
        self.__original = _original_series
        self.__anomalies = _anomalies
        self.__upper_bound = _upper_bound
        self.__lower_bound = _lower_bound
        self.__method = _method

    @property
    def original(self):
        return self.__original
    
    @property
    def anomalies(self):
        return self.__anomalies
    
    @property
    def upper_bound(self):
        return self.__upper_bound

    @property
    def lower_bound(self):
        return self.__lower_bound
    
    @property
    def method(self):
        return self.__method

    @property
    def n_anomalies(self):
        return self.__anomalies.shape[0]

    def plot(self, figsize=(20,5), tight_layout=True, grid=True, title=""):
        
        plt.figure(figsize=(20, 5))
        plt.title(title)

        # plot the upper and lower bound
        if self.__method != "rci":
            plt.plot(self.__upper_bound, "r--", label="Upper Bound / Lower Bound")
            plt.plot(self.__lower_bound, "r--")

        # plot the origianl series
        plt.plot(self.__original)

        # plot the anolaies
        plt.plot(self.__anomalies, "ro", markersize=10)

        if tight_layout:
            plt.tight_layout()
        plt.grid(grid)
        plt.show()


    def summary(self):
        print("[--] Number of anomalies detected: {}".format(self.n_anomalies))

class SmoothingResult():
    """
    Description
    -----------
    The result class returned by a smoothing function
    """

    def __init__(self,
        _original_series: pd.Series,
        _smoothed_series: pd.Series,
        _method: str,
        _params: dict = {}
    ):
        self.__original = _original_series
        self.__smoothed = _smoothed_series
        self.__params = _params
        self.__method = _method


    @property
    def params(self):
        return self.__params

    @property
    def original(self):
        return self.__original

    @property
    def smoothed(self):
        return self.__smoothed


    def plot_compare(self, title=""):
        """
        Description
        -----------
        Plot the original and the smoothed series all into one plot
        """

        if title == "":
            title = "{} ".format(self.__method) + "| params: {}".format(self.__params) if len(self.__params) > 0 else ""

        plot.plot_single(
            [self.__original, self.__smoothed],
            labels=["original", "smoothed"],
            title=title
        )


    def plot_original(self, title=""):
        """
        Description
        -----------
        Plot the original series
        """

        if title == "":
            title = "original series"

        plot.plot_single(
            [self.__original],
            title="original series"
        )


    def plot_smoothed(self):
        """
        Description
        -----------
        Plot the smoothed series
        """

        if title == "":
            title = "{} ".format(self.__method) + "| params: {}".format(self.__params) if len(self.__params) > 0 else ""

        plot.plot_single(
            [self.__smoothed],
            title=title
        )

