import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import statsmodels.api as sm

try:
    import clif
except:
    import sys

    sys.path.append("../")
    # from eof import fingerprints
    import clif

n = 1000
# stochastic_data[:, 0] is a stocastic gaussian process. This should always be stationary.
stochastic_data = np.random.randn(n, 3)
for t in range(n):
    stochastic_data[t, 1] = stochastic_data[t - 1, 1] * stochastic_data[t, 0]
    stochastic_data[t, 2] = stochastic_data[t - 1, 2] + stochastic_data[t, 0]
# trend_series is a linearly increasing function representing a simple changing mean.
#   It is difference stationary, so the ADF test return stationary and the KPSS test return nonstationary.
#   Due to the difference in the results from ADF test and KPSS test, it can be inferred that the series is trend stationary and not strict stationary.
trend_series = np.arange(0, n) / (n - 1)
# This dataset is useful because it is trend stationary, so it should make the ADF test return nonstationary and the KPSS test return stationary.
#   Due to the difference in the results from ADF test and KPSS test, it can be inferred that the series is trend stationary and not strict stationary.
sun_data = sm.datasets.sunspots.load_pandas().data["SUNACTIVITY"].values

time_series = sun_data  # 1 + 0.1 * (np.random.rand(100))

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

stest = clif.statistics.StationarityTest(test="adfuller", pvalue=0.01)
stest.fit(time_series)
print("adfuller stationary", stest.is_stationary)

stest = clif.statistics.StationarityTest(test="kpss", pvalue=0.01)
stest.fit(time_series)
print("kpss stationary", stest.is_stationary)

"""
Augmented Dickey-Fuller test for stationarity (Null is non-stationary)
p-value > 0.01: Accept the null hypothesis (H0), the data is non-stationary.
p-value <= 0.01: Reject the null hypothesis (H0), the data is stationary.
"""
adfuller_is_stationary = clif.statistics.adfuller_stationarity_test(time_series)

"""
KPSS test for stationarity (Null is stationary)
p-value > 0.01: Accept the null hypothesis (H0), the data is stationary.
p-value <= 0.01: Reject the null hypothesis (H0), the data is non-stationary.
"""
kpss_is_stationary = clif.statistics.kpss_stationarity_test(time_series)

print("adfuller stationarity", adfuller_is_stationary)
print("kpss stationarity", kpss_is_stationary)

# trend_series_result = clif.statistics.stationarity(trend_series, 0.01, verbosity=0)

# assert (
#     trend_series_result[0] == True and trend_series_result[1] == False
# ), "Test on linear trend is incorrect."

# stationary_data_result = clif.statistics.stationarity(
#     stochastic_data[:, 0], 0.01, verbosity=0
# )
# assert (
#     stationary_data_result[0] == True and stationary_data_result[1] == True
# ), "Test on stationary data is incorrect."

# # Not implemented:
# # heteroscedastic_result = clif.statistics.stationarity(
# #     stochastic_data[:, 1], 0.01, verbosity=0
# # )
# # assert (
# #     heteroscedastic_result[0] == True
# #     and heteroscedastic_result[1] == True
# #     and heteroscedastic_result[2] == False
# # ), "Test on heteroscedastic data is incorrect"

# seasonal_series_result = clif.statistics.stationarity(
#     stochastic_data[:, 2], 0.01, verbosity=0
# )
# assert (
#     seasonal_series_result[0] == False and seasonal_series_result[1] == False
# ), "Test on seasonal data is incorrect."

# sun_data_result = clif.statistics.stationarity(sun_data, 0.01, verbosity=0)
# assert (
#     sun_data_result[0] == False and sun_data_result[1] == True
# ), "Test on trend-only stationary data is incorrect."
