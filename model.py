import numpy as np
import sklearn.linear_model
import sklearn.neural_network
import sklearn.ensemble

def _get_inputs(inputs):
    pass

def _get_output():
    pass

def train(model, inputs):

    X_train = _get_inputs(inputs)  # predictors
    y_train = _get_output()        # predictand

    if model == 0:
        m = sklearn.linear_model.LinearRegression()
    elif model == 1:
        m = sklearn.neural_network.MLPRegressor()
    elif model == 2:
        m = sklearn.ensemble.RandomForestRegressor()
    else:
        return None

    m.fit(X_train, y_train)
    mae = np.mean((m2.predict(X_train)-y_train)**2)

    return m
