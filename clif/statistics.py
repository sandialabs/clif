import numpy as np
import xarray
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

"""A class containing functions for determining the stationarity of a given time series.

Stationarity means that the statistical properties of a time series i.e. mean, variance and covariance do not change over time.
Many statistical models require the series to be stationary to make effective and precise predictions.

Two statistical tests are used to check the stationarity of a time series - Augmented Dickey Fuller (“ADF”) test and Kwiatkowski-Phillips-Schmidt-Shin (“KPSS”) test.
"""


class StationarityTest:
    """Perform stationarity tests on time series data

    Uses statsmodels's unit root adfuller test and the kpss stationarity test.

    Parameters
    ----------
    test: {'adfuller','kpss'}, default = 'adfuller'
        Test type for stationarity

    pvalue: float, default=.01
        threshold to decide whether statistical tests are significant

    Examples
    --------
    >>> import numpy as np
    >>> from clif.statistics import StationarityTest
    >>> time_series = 1 + .01*np.sort(np.random.rand(100))
    >>> stest = StationarityTest()
    >>> stest.fit(time_series)
    >>> print(stest.is_stationary)

    """

    def __init__(self, test="adfuller", pvalue=0.01, params=None):
        self.test = test
        self.pvalue = pvalue
        self.params = params

    def _set_default_params(self):
        # use default parameters is params is None
        if self.params is None:
            if self.test == "adfuller":
                self.params = {"regression": "c", "autolag": "AIC"}
            elif self.test == "kpss":
                self.params = {"regression": "c", "nlags": "auto"}

    def fit(self, time_series):
        self._set_default_params()
        if self.test == "adfuller":
            self._fit_adfuller(time_series)
        if self.test == "kpss":
            self._fit_kpss(time_series)
        return self

    def _fit_adfuller(self, time_series):
        """
        Returns boolean to indicate whether series is stationary (True) or non-stationary (False)

        Augmented Dickey-Fuller test for stationarity (Null is non-stationary)
        p-value > 0.01: Accept the null hypothesis (H0), the data is non-stationary.
        p-value <= 0.01: Reject the null hypothesis (H0), the data is stationary.

        """
        adfuller_test = adfuller(time_series, **self.params)
        adfuller_p_statistic = adfuller_test[1]
        self.is_stationary = False  # Null hypothesis
        if adfuller_p_statistic <= self.pvalue:
            self.is_stationary = True
        return self

    def _fit_kpss(self, time_series):
        """
        Returns boolean to indicate whether series is stationary (True) or non-stationary (False)

        KPSS test for stationarity (Null is stationary)
        p-value > 0.01: Accept the null hypothesis (H0), the data is stationary.
        p-value <= 0.01: Reject the null hypothesis (H0), the data is non-stationary.
        """
        kpss_test = kpss(time_series, **self.params)
        kpss_p_statistic = kpss_test[1]
        self.is_stationary = True  # Null hypothesis
        if kpss_p_statistic <= self.pvalue:
            self.is_stationary = False
        return self


def StationarityTestOld(time_series, p_value_threshold, verbosity=1):
    """Tests whether the given time series is stationary with respect to the given p-value threshold.

    This uses both the Augmented Dickey Fuller (“ADF”) test and Kwiatkowski-Phillips-Schmidt-Shin (“KPSS”) test to determine stationarity. These are hypothsis tests in which the null and alternate hypotheses are opposites.
    If the ADF test fails to reject the null hypothesis, this may provide evidence that the series is non-stationary.
    If the KPSS test fails to reject the null hypothesis, this may provide evidence that the series is stationary.

    There are four cases of stationarity in this test:
    Case 1: Both tests conclude that the series is not stationary - The series is not stationary
    Case 2: Both tests conclude that the series is stationary - The series is stationary
    Case 3: KPSS indicates stationarity and ADF indicates non-stationarity - The series is trend stationary. Trend needs to be removed to make series strict stationary. The detrended series is checked for stationarity.
    Case 4: KPSS indicates non-stationarity and ADF indicates stationarity - The series is difference stationary. Differencing is to be used to make series stationary. The differenced series is checked for stationarity.

    Trend stationary: The mean trend is deterministic. Once the trend is estimated and removed from the data, the residual series is a stationary stochastic process.
    Difference stationary: The mean trend is stochastic. Differencing the series D times yields a stationary stochastic process.

    Parameters
    ----------
    time_series : numpy:array_like, 1d
        The data to be tested.
    p_value_threshold : float
        The hypothesis test threshold for the returned p-values.
    verbosity : int, optional
        Determines the level of verbosity. 1 explains the results and 2 print the ADF and KPSS p-values, by default 1

    Returns
    -------
    tuple
        A pair of booleans (adf_stationary, kpss_stationary) corresponding to whether the ADF test returns stationary or the KPSS test returns stationary, respectively.
    """

    adf_p_val = adfuller(time_series, autolag="AIC")[1]
    kpss_p_val = kpss(time_series, regression="c", nlags="auto")[1]

    # adf_reject means stationary
    # kpss_reject means not stationary
    adf_reject = adf_p_val <= p_value_threshold
    kpss_reject = kpss_p_val <= p_value_threshold
    adf_stationary = adf_reject
    kpss_stationary = ~kpss_reject

    if verbosity > 0:
        if verbosity > 1:
            print("ADF p-value:", adf_p_val)
            print("KPSS p-value:", kpss_p_val)
        if adf_stationary and kpss_stationary:
            print("Series is stationary.")
        elif not adf_stationary and kpss_stationary:
            print(
                "Series is trend stationary only. Trend needs to be removed to make the time series fully stationary."
            )
        elif adf_stationary and not kpss_stationary:
            print(
                "Series is difference stationary only. Seasonal differencing can be used to make the time series fully stationary. "
            )
        else:
            print("Series is not stationary.")

    return adf_stationary, kpss_stationary
