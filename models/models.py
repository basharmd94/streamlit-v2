# models.py
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

def train_sarima(data, order, seasonal_order):
    model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    return model_fit

def update_sarima_with_params(data, order, seasonal_order, params):
    model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
    model.initialize_known(params['params'], params['other_results'])
    model_fit = model.smooth(params)
    return model_fit

def forecast_sarima(model_fit, steps):
    return model_fit.forecast(steps=steps)

def optimize_sarima(data, p_values, d_values, q_values, P_values, D_values, Q_values, s):
    best_score, best_order, best_seasonal_order = float('inf'), None, None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                for P in P_values:
                    for D in D_values:
                        for Q in Q_values:
                            order = (p, d, q)
                            seasonal_order = (P, D, Q, s)
                            try:
                                model = SARIMAX(data, 
                                                order=order, 
                                                seasonal_order=seasonal_order,
                                                enforce_stationarity=False, 
                                                enforce_invertibility=False)
                                model_fit = model.fit(disp=False, maxiter=50, tol=1e-2)
                                bic = model_fit.bic
                                if bic < best_score:
                                    best_score, best_order, best_seasonal_order = bic, order, seasonal_order
                            except:
                                continue
    return best_order, best_seasonal_order
