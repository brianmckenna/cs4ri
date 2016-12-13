import aiohttp
import base64
import datetime
import io
import json
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shelve
import sklearn.ensemble
import sklearn.linear_model
import sklearn.neural_network
import sklearn.svm

import requests

sns.set(style="darkgrid", palette="Set2")



# loads datasets
data_trn = np.load('data/train.npz')
data_tst = np.load('data/test.npz')





# SANDY FULL INDEX 23/00Z to 31/00Z
_dt = data_tst['dt']
print(_dt.shape)

# check all 6 minute intervals are there, add if not
add = []
t = _dt.min()
while True:
    if t > _dt.max():
        break
    if t not in _dt:
        add.append(t)
    t += datetime.timedelta(minutes=6)

# update _dt, SANDY FULL INDEX with every 6 minutes
dt = sorted(add + _dt.tolist())

idx_dt = range(len(data_tst['k004'].tolist()))

# index of existing values 
idx_tst = [dt.index(t) for t in _dt.tolist()] # TODO interpolate where None values
idx_add = [dt.index(t) for t in add]

# numpy this
dt = np.array(dt)
print(dt.shape)

# SANDY DISPLAY INDEX
def sandy(dt):
    mask = np.zeros(dt.shape, dtype=bool)
    _ = np.logical_and((dt >= datetime.datetime(2012,10,23,0,0,0)), (dt <= datetime.datetime(2012,10,31,0,0,0)))
    return np.logical_or(mask, _)
display_dt = dt[sandy(dt)]


# indexes to use
#idx_tst = range(len(data_tst['k004'].tolist()))

# masks values outside window
mask = data_trn['mask'][0] | data_trn['mask'][1] | data_trn['mask'][2]
idx0  = np.where(mask)
idx1 = np.where(data_trn['mask'][0])
idx2 = np.where(data_trn['mask'][1])
idx3 = np.where(data_trn['mask'][2])

# persisted objects
db = shelve.open('models.shelve')
forecasts = shelve.open('forecasts.shelve')


def _interpolate(idx_add, i, d):
    print(d.shape)
    print(len(idx_add))
    print(len(i))
    d[idx_add] = np.interp(idx_add, i, d[i])
    return d

def _get_inputs(data, input_keys, idx):
    inputs = []
    for k in input_keys:
        if k.endswith('3'):
            inputs.append(np.ma.masked_values(data[k+'u'], -9999.9)[idx])
            inputs.append(np.ma.masked_values(data[k+'v'], -9999.9)[idx])
        else:
            inputs.append(np.ma.masked_values(data[k], -9999.9)[idx])
    return np.array(inputs).T

def _get_output(data, idx):
    return data['k004'][idx]





async def forecast(cs4ri_id):
    (model, input_keys, m) = db.get(cs4ri_id, None)
    X_test = _get_inputs(data_tst, input_keys, idx_dt)  # predictors
    y_test = _get_output(data_tst, idx_dt)

    '''
    # create full array
    X_full = np.ones((dt.shape[0],X_test.shape[1]))*-99999.9
    y_full = np.ones(dt.shape)*-99999.9

    # set existing values
    X_full[idx_tst,:] = X_test
    y_full[idx_tst] = y_test
    
    # interpolate values
    X_full = _interpolate(idx_add, idx_tst, X_full)
    y_full = _interpolate(idx_add, idx_tst, X_full)

    print(X_test.shape, y_test.shape)
    print(X_full.shape, y_full.shape)
    '''

    predict = m.predict(X_test)
    forecasts[cs4ri_id] = predict.tolist()
    cmin, cmean, cmax = consensus()

    print(dt.shape, dt[100].isoformat())
    print(predict.shape, predict[100])
    print(cmean.shape, cmean[100])

    await data_fountain(predict, cmin, cmean, cmax, dt)

    return forecast_plot(predict, cmin, cmean, cmax)

def consensus():
    f = np.array([v for k,v in forecasts.items()])
    return [f.min(axis=0), f.mean(axis=0), f.max(axis=0)]

def train(cs4ri_id, model, input_keys):

    X_train = _get_inputs(data_trn, input_keys, idx0)  # predictors (idx0 all inputs)
    y_train = _get_output(data_trn, idx0)              # predictand

    # handle random request
    _model = model
    if model == 4:
        _model = np.random.randint(0,4)

    # determine model
    if _model == 0:
        m = sklearn.linear_model.LinearRegression()
    elif _model == 1:
        m = sklearn.neural_network.MLPRegressor()
    elif _model == 2:
        m = sklearn.svm.SVR()
    elif _model == 3:
        m = sklearn.ensemble.RandomForestRegressor()
    else:
        return None

    # train model
    m.fit(X_train, y_train)

    # statistics of training
    mae = np.mean((m.predict(X_train)-y_train)**2)

    # log the training 
    print("%.4f, %d, %d, %s" % (mae, _model, model, '|'.join(input_keys)))#, X_train.shape, y_train.shape)

    # three events
    X1_train = _get_inputs(data_trn, input_keys, idx1)  # predictors
    y1_train = _get_output(data_trn, idx1)              # predictand
    predict1 = m.predict(X1_train)
    mae1 = np.mean((predict1-y1_train)**2)

    X2_train = _get_inputs(data_trn, input_keys, idx2)  # predictors
    y2_train = _get_output(data_trn, idx2)              # predictand
    predict2 = m.predict(X2_train)
    mae2 = np.mean((predict2-y2_train)**2)

    X3_train = _get_inputs(data_trn, input_keys, idx3)  # predictors
    y3_train = _get_output(data_trn, idx3)              # predictand
    predict3 = m.predict(X3_train)
    mae3 = np.mean((predict3-y3_train)**2)

    # persist model for forecast (one per CS4RI id)
    db[cs4ri_id] = (model, input_keys, m)

    # returns the training plots for visual feedback
    return training_plots([
        (predict1, y1_train, mae1),
        (predict2, y2_train, mae2),
        (predict3, y3_train, mae3)
    ])


# -----------------------------------------------------------------------------
# DATA FOUNTAIN
# -----------------------------------------------------------------------------
async def data_fountain(predict, cmin, cmean, cmax, dt):

    _id = "SXajFa85CSzKjxphx" # TODO

    times = list(map(lambda l: l.isoformat(), dt.tolist()))
    df = {
            "data": {
                "consensus": {
                    "values": cmean.tolist(),
                    "units": "feet",
                    "type":  "timeSeries",
                    "times": times
                },
                "upper bound": {
                    "values": cmax.tolist(),
                    "units": "feet",
                    "type":  "timeSeries",
                    "times": times
                },
                "lower bound": {
                    "values": cmin.tolist(),
                    "units": "feet",
                    "type":  "timeSeries",
                    "times": times
                },
            },
            "id": "cs4ri",
            "title": "CS4RI Forecast Data Challenge",
    }

    async with aiohttp.ClientSession() as session:
        async with session.put('http://10.90.69.67:3000/api/v1/Data/%s' % _id, data=json.dumps(df), headers = {'content-type': 'application/json'}) as resp:
            pass
            #print(resp.status)


# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------

def training_plots(pv):

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)

    ax1.set_ylim([-3,7])
    ax1.plot(pv[0][1], 'k-', linewidth=0.5)
    ax1.plot(pv[0][0], 'r-', linewidth=0.25)
    #ax1.text(0.1, 0.92, '%.4f ft' % pv[0][2], horizontalalignment='center', verticalalignment='center', transform = ax1.transAxes)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), fontsize='9')

    ax2.plot(pv[1][1], 'k-', linewidth=0.5)
    ax2.plot(pv[1][0], 'r-', linewidth=0.25)
    #ax2.text(0.1, 0.92, '%.4f ft' % pv[1][2], horizontalalignment='center', verticalalignment='center', transform = ax2.transAxes)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), fontsize='9')

    ax3.plot(pv[2][1], 'k-', linewidth=0.5)
    ax3.plot(pv[2][0], 'r-', linewidth=0.25)
    #ax3.text(0.1, 0.92, '%.4f ft' % pv[2][2], horizontalalignment='center', verticalalignment='center', transform = ax3.transAxes)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), fontsize='9')

    # legends
    v_patch = matplotlib.patches.Patch(color='k', label='Measured')
    p_patch = matplotlib.patches.Patch(color='r', label='Predicted')
    ax1.legend(handles=[v_patch, p_patch])
    ax2.legend(handles=[v_patch, p_patch])
    ax3.legend(handles=[v_patch, p_patch])

    # image output
    dpi = 300
    fig.set_alpha(0)
    fig.set_figheight(2200/dpi)
    fig.set_figwidth(1300/dpi)
    with io.BytesIO() as _buffer:
        fig.savefig(_buffer, dpi=dpi, bbox_inches='tight', pad_inches=0.0)
        plt.close(fig)
        _buffer.seek(0)
        return base64.b64encode(_buffer.getvalue())


#def forecast_plot(predict):
def forecast_plot(predict, cmin, cmean, cmax):

    fig, ax1 = plt.subplots(1, sharex=True, sharey=True)

    ax1.set_ylim([-3,7])
    ax1.plot(predict, 'r-', linewidth=0.35)
    #ax1.plot(cmin,    'b-', linewidth=0.25)
    ax1.plot(cmean,   'k-', linewidth=0.25)
    #ax1.plot(cmax,    'g-', linewidth=0.25)
    ax1.axhline(y=predict.max(), color='b', linestyle='--', linewidth=0.25)
    ax1.text(0.2, 0.92, 'max predicted water level: %.4f ft' % predict.max(), fontsize='4', horizontalalignment='center', verticalalignment='center', transform = ax1.transAxes)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), fontsize='4')

    # image output
    dpi = 300
    fig.set_alpha(0)
    fig.set_figheight(400/dpi)
    fig.set_figwidth(800/dpi)
    with io.BytesIO() as _buffer:
        fig.savefig(_buffer, dpi=dpi, bbox_inches='tight', pad_inches=0.0)#, transparent=True)
        plt.close(fig)
        _buffer.seek(0)
        return base64.b64encode(_buffer.getvalue())

