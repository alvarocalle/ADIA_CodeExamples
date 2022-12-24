import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX


def sarima_prediction(ser, steps_ahead=100, ci=0.95):
    """
    Forecast the close price n_forecasts steps ahead given the current trading window
    :param ser: pandas.Series (Close priced) for the current window
    :param steps_ahead: number of steps ahead to forecast
    :param ci: confidence interval of prediction
    :return:
    """

    stationary_df = np.log(ser) - np.log(ser.shift(1))

    model = SARIMAX(stationary_df.values, order=(3, 1, 0), seasonal_order=(2, 2, 0, 6))
    fit = model.fit(method='bfgs')

    # option 1: sarima prediction for current series length
    ####pred = fit.get_prediction(start=0, dynamic=False)
    # option 2: sarima forecast steps ahead in future
    pred = fit.get_forecast(steps=steps_ahead)

    pred_mean = pred.predicted_mean
    pred_ci = pred.conf_int(alpha=1-ci)

    return pred_mean, pred_ci


