import numpy as np
import xarray
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

"""A class containing functions for determining the stationarity of a given time series.

Stationarity means that the statistical properties of a time series i.e. mean, variance and covariance do not change over time.
Many statistical models require the series to be stationary to make effective and precise predictions.
Two statistical tests are used to check the stationarity of a time series - Augmented Dickey Fuller (“ADF”) test and Kwiatkowski-Phillips-Schmidt-Shin (“KPSS”) test.

The ADF test is a unit root test. Unit roots are one cause for non-stationarity. 
The KPSS test is a test for "trend stationarity." 
"""


class StationarityTest:
    """Perform stationarity tests on time series data

    Uses statsmodels's unit root adfuller test and the kpss stationarity test.

    Parameters
    ----------
    test: str or list of: {'adfuller','kpss'}, default = 'adfuller'
        Test type for stationarity

    pvalue: float, default=.01
        threshold to decide whether statistical tests are significant

    Examples
    --------
    >>> from clif.statistics import StationarityTest
    >>> import numpy as np
    >>> rn = np.random.RandomState(2342)
    >>> n = 1000
    >>> time_series = 1 + 0.1 * np.sort(rn.rand(n))
    >>> print(StationarityTest(tests="adfuller").fit(time_series).is_stationary)
    >>> print(StationarityTest(tests=["adfuller", "kpss"]).fit(time_series).is_stationary)

    """

    def __init__(self, tests="adfuller", alpha=0.01, params=None):
        self.tests = tests
        self.alpha = alpha
        self.params = params
        self.is_stationary = []

    def _set_default_params(self):
        # use default parameters is params is None
        if self.params is None:
            if isinstance(self.tests, str):
                if self.tests == "adfuller":
                    self.params = {"regression": "c", "autolag": "AIC"}
                elif self.tests == "kpss":
                    self.params = {"regression": "c", "nlags": "auto"}
            elif isinstance(self.tests, list):
                self.params = []
                for t in self.tests:
                    if t == "adfuller":
                        self.params.append({"regression": "c", "autolag": "AIC"})
                    elif t == "kpss":
                        self.params.append({"regression": "c", "nlags": "auto"})

    def fit(self, time_series):
        self._set_default_params()
        if isinstance(self.tests, str):
            if self.tests == "adfuller":
                self.is_stationary = self._fit_adfuller(time_series, params=self.params)
            elif self.tests == "kpss":
                self.is_stationary = self._fit_kpss(time_series, params=self.params)
            return self
        elif isinstance(self.tests, list):
            for i, t in enumerate(self.tests):
                if t == "adfuller":
                    self.is_stationary.append(
                        self._fit_adfuller(time_series, params=self.params[i])
                    )
                elif t == "kpss":
                    self.is_stationary.append(
                        self._fit_kpss(time_series, params=self.params[i])
                    )
            return self
        else:
            raise TypeError("test param must be a string or a list of strings.")

    def _fit_adfuller(self, time_series, params=None):
        """
        Returns boolean to indicate whether series is stationary (True) or non-stationary (False)

        Augmented Dickey-Fuller test for stationarity (Null is non-stationary)
        p-value > 0.01: Fail to reject the null hypothesis (H0), the data has a unit-root and is likely nonstationary.
        p-value <= 0.01: Reject the null hypothesis (H0), the data is stationary.

        """
        adfuller_test = adfuller(time_series, **params)
        adfuller_p_value = adfuller_test[1]
        is_stationary = False  # Null hypothesis
        if adfuller_p_value <= self.alpha:
            is_stationary = True
        return is_stationary

    def _fit_kpss(self, time_series, params=None):
        """
        Returns boolean to indicate whether series is stationary (True) or non-stationary (False)

        KPSS test for stationarity (Null is stationary)
        p-value > 0.01: Fail to reject the null hypothesis (H0), the data is trend-stationary.
        p-value <= 0.01: Reject the null hypothesis (H0), the data has a unit-root and is likely nonstationary.
        """
        kpss_test = kpss(time_series, **params)
        kpss_p_value = kpss_test[1]
        is_stationary = True  # Null hypothesis
        if kpss_p_value <= self.alpha:
            is_stationary = False
        return is_stationary

    def interpreted_stationarity_tests(self, time_series):
        """Provides an interpreted stationarity testing result of the given time series.t

        Interpretation is mainly derived from these sources:
        https://www.statsmodels.org/stable/examples/notebooks/generated/stationarity_detrending_adf_kpss.html
        https://stats.stackexchange.com/a/235916/240083
        https://www.mathworks.com/help/econ/trend-stationary-vs-difference-stationary.html

        This uses the Augmented Dickey Fuller (“ADF”) test and Kwiatkowski-Phillips-Schmidt-Shin (“KPSS”) test to determine stationarity. These are hypothsis tests in which the null and alternate hypotheses are opposites.
        If the ADF test fails to reject the null hypothesis, this may provide evidence that the series is non-stationary.
        If the KPSS test fails to reject the null hypothesis, this may provide evidence that the series is stationary.

        There are four cases of stationarity in this test:
        Case 1: Both tests conclude that the series is not stationary - The series is not stationary.
        Case 2: Both tests conclude that the series is stationary - The series is stationary.
        Case 3: KPSS indicates stationarity and ADF indicates non-stationarity - The series is trend stationary. Trend needs to be removed to make series strict stationary.
        Case 4: KPSS indicates non-stationarity and ADF indicates stationarity - The series is difference stationary. Differencing is to be used to make series stationary.

        Trend stationary: The mean trend is deterministic. Once the trend is estimated and removed from the data, the residual series is a stationary stochastic process.
        Difference stationary: The mean trend is stochastic. Differencing the series D times yields a stationary stochastic process.

        Parameters
        ----------
        time_series : numpy:array_like, 1d
            The data to be tested.

        Returns
        -------
        tuple
            A pair of booleans (adf_stationary, kpss_stationary) corresponding to whether the ADF test returns stationary or the KPSS test returns stationary, respectively.
        """

        assert isinstance(
            self.tests, list
        ), "More than one test must be passed for interpretation."

        stationarity = dict(zip(self.tests, self.fit(time_series).is_stationary))
        adf_stationary = stationarity["adfuller"]
        kpss_stationary = stationarity["kpss"]

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

        return stationarity
