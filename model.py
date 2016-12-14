import aiohttp
import asyncio
import base64
import datetime
import io
import json
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import operator
import seaborn as sns
import shelve
import sklearn.ensemble
import sklearn.linear_model
import sklearn.neural_network
import sklearn.svm

sns.set(style="darkgrid", palette="Set2")

ft_formatter = matplotlib.ticker.FormatStrFormatter('%d ft')

# loads datasets
data_trn = np.load('data/train_filled.npz')
data_tst = np.load('data/test_filled.npz')

# index of existing values 
dt = data_tst['dt']

# SANDY DISPLAY INDEX
def sandy(dt):
    mask = np.zeros(dt.shape, dtype=bool)
    _ = np.logical_and((dt >= datetime.datetime(2012,10,23,0,0,0)), (dt <= datetime.datetime(2012,10,31,0,0,0)))
    return np.logical_or(mask, _)

idx_tst = sandy(dt)

# masks values on training set (3 data sets)
mask = data_trn['mask'][0] | data_trn['mask'][1] | data_trn['mask'][2]
idx0 = np.where(mask)
idx1 = np.where(data_trn['mask'][0])
idx2 = np.where(data_trn['mask'][1])
idx3 = np.where(data_trn['mask'][2])

# persisted objects
db = shelve.open('models.shelve')
forecasts = shelve.open('forecasts.shelve')


def _get_inputs(data, input_keys, idx):
    inputs = []
    for k in input_keys:
        if k.endswith('3'):
            #inputs.append(np.ma.masked_values(data[k+'u'], -9999.9)[idx])
            #inputs.append(np.ma.masked_values(data[k+'v'], -9999.9)[idx])
            inputs.append(data[k+'u'][idx])
            inputs.append(data[k+'v'][idx])
        else:
            #inputs.append(np.ma.masked_values(data[k], -9999.9)[idx])
            inputs.append(data[k][idx])
    return np.array(inputs).T

def _get_output(data, idx):
    return data['k004'][idx]



obs = _get_output(data_tst, idx_tst)


async def forecast(cs4ri_id):

    (model, input_keys, m) = db.get(cs4ri_id, None)

    X_test = _get_inputs(data_tst, input_keys, idx_tst)  # predictors
    y_test = _get_output(data_tst, idx_tst)

    predict = m.predict(X_test)
    forecasts[cs4ri_id] = predict.tolist()
    cmin, cmean, cmax = consensus()

    #print(dt[idx_tst].shape, dt[idx_tst][100].isoformat())
    #print(predict.shape, predict[100])
    #print(cmean.shape, cmean[100])

    await data_fountain(cmin, cmean, cmax, dt)

    return forecast_plot(dt[idx_tst], predict, cmin, cmean, cmax)

async def results():
    mask_idx = await time_offset()
    r = {}
    for k,v in forecasts.items():
        mae = np.mean((v[mask_idx]-obs[mask_idx])**2)
        r[k] = mae
    cmin, cmean, cmax = consensus()
    r['AVERAGE FORECAST (CONSENSUS)'] = np.mean((cmean[mask_idx]-obs[mask_idx])**2)
    return sorted(r.items(), key=operator.itemgetter(1))

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
#async def data_fountain(predict, cmin, cmean, cmax, dt):
async def data_fountain(cmin, cmean, cmax, dt):

    _id = "2aHZgZ6KiiDgwxuLe" # TODO

    mask_idx = await time_offset()
    
    # FOR OBS (time based) ALSO COMPUTE LEADERBOARD HERE (MAX AND MAE)
    _obs = obs.tolist()
    for i in range(mask_idx, len(_obs)):
        _obs[i] = None

    times = list(map(lambda l: l.isoformat(), dt.tolist()))
    df = {
            "data": {
                "times": times,
                "waterLevel": {
                    "values": _obs,
                    "units": "feet",
                    "type":  "timeSeries",
                    "times": times
                },
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

    #print(json.dumps(df))
    async with aiohttp.ClientSession() as session:
        #async with session.post('http://70.181.21.199:3000/api/v1/Data', data=json.dumps(df), headers = {'content-type': 'application/json'}) as resp:
        async with session.put('http://70.181.21.199:3000/api/v1/Data/%s' % _id, data=json.dumps(df), headers = {'content-type': 'application/json'}) as resp:
            print(resp.status)
            #print(await resp.text())
            #pass

        df = {'data': {'waterLevel':cmean.max()*0.3048}}
        #print(json.dumps(df))
        async with session.put('http://70.181.21.199:3000/api/v1/Unity/5LAcJtBgzGz4kFHnM', data=json.dumps(df), headers = {'content-type': 'application/json'}) as resp:
            print(resp.status)
            #print(await resp.text())
            #pass


# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------

def training_plots(pv):

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)

    ax1.set_ylim([-3,9])
    ax1.plot(pv[0][1], 'k-', linewidth=0.5)
    ax1.plot(pv[0][0], 'r-', linewidth=0.25)
    ax1.yaxis.set_major_formatter(ft_formatter)

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
    ax1.legend(loc="upper left", handles=[v_patch, p_patch])
    ax2.legend(loc="upper left", handles=[v_patch, p_patch])
    ax3.legend(loc="upper left", handles=[v_patch, p_patch])

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
def forecast_plot(dt, predict, cmin, cmean, cmax):

    fig, ax1 = plt.subplots(1, sharex=True, sharey=True)

    ax1.set_ylim([-3,9])
    ax1.yaxis.set_major_formatter(ft_formatter)
    ax1.plot(dt, predict, 'r-', linewidth=0.35)
    ax1.plot(dt, cmean,   'k-', linewidth=0.25)
    ax1.axhline(y=predict.max(), color='b', linestyle='--', linewidth=0.25)

    #y = ((predict.max()+3)/12)+0.1
    y = -0.025
    ax1.text(0.8, y, 'max predicted water level: %.4f ft' % predict.max(), color='b', fontsize='4', horizontalalignment='center', verticalalignment='center', transform = ax1.transAxes)
    plt.setp(ax1.get_xticklabels(), fontsize='4', rotation=30, horizontalalignment='right')
    plt.setp(ax1.get_yticklabels(), fontsize='4')

    # legends
    v_patch = matplotlib.patches.Patch(color='k', label='Average Forecast')
    p_patch = matplotlib.patches.Patch(color='r', label='Your Forecast')
    ax1.legend(loc="upper left", fontsize='4', handles=[v_patch, p_patch])

    # image output
    dpi = 300
    fig.set_alpha(0)
    fig.set_figheight(400/dpi)
    fig.set_figwidth(1000/dpi)
    with io.BytesIO() as _buffer:
        fig.savefig(_buffer, dpi=dpi, bbox_inches='tight', pad_inches=0.0)#, transparent=True)
        plt.close(fig)
        _buffer.seek(0)
        return base64.b64encode(_buffer.getvalue())


async def time_offset():
    sandy_seconds = 691200.0
    four_hours_seconds = 14400.0
    running_seconds = (datetime.datetime.now() - datetime.datetime(2016,12,14,8,0,0)).total_seconds()
    sandy_minutes = ((sandy_seconds/four_hours_seconds)*running_seconds)/60
    return max(int(sandy_minutes/6),10)

async def update_obs():
    for i in range(0,30000):
        print(await time_offset())
        #print(i)
        #await obs_data_fountain()
        await asyncio.sleep(10)
