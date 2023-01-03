import statsmodels.tsa.stattools as sts
import pandas as pd

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