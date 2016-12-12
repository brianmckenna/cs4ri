import base64
import io
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shelve
import sklearn.ensemble
import sklearn.linear_model
import sklearn.neural_network
import sklearn.svm

sns.set(style="darkgrid", palette="Set2")



# run at LOAD
data = np.load('data/train.npz')
test_data = np.load('data/test.npz')
idx_ = range(len(test_data['k004'].tolist()))

# masks values outside window
mask = data['mask'][0] | data['mask'][1] | data['mask'][2]
idx0  = np.where(mask)
idx1 = np.where(data['mask'][0])
idx2 = np.where(data['mask'][1])
idx3 = np.where(data['mask'][2])

db = shelve.open('models.shelve')

def _get_inputs(input_keys, idx):
    inputs = []
    for k in input_keys:
        if k.endswith('3'):
            inputs.append(np.ma.masked_values(data[k+'u'], -9999.9)[idx])
            inputs.append(np.ma.masked_values(data[k+'v'], -9999.9)[idx])
        else:
            inputs.append(np.ma.masked_values(data[k], -9999.9)[idx])
    return np.array(inputs).T

def _get_output(idx):
    return data['k004'][idx]

def forecast(cs4ri_id):

    (model, input_keys, m) = db.get(cs4ri_id, None)

    X_test = _get_inputs(input_keys, idx_)  # predictors
    y_test = _get_output(idx_)  

    predict = m.predict(X_test)
    mae = np.mean((predict-y_test)**2)
    print(mae)
    print(predict)
    print(predict.shape)

    return forecast_plot(predict)

def train(cs4ri_id, model, input_keys):

    X_train = _get_inputs(input_keys, idx0)  # predictors (idx0 all inputs)
    y_train = _get_output(idx0)               # predictand

    # handle random request
    _model = model
    if model == 4:
        _model = np.random.randint(0,4)

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

    m.fit(X_train, y_train)
    mae = np.mean((m.predict(X_train)-y_train)**2)

    mae0 = np.mean((m.predict(X_train)-y_train)**2)
    print(mae, _model, model, input_keys, X_train.shape, y_train.shape)

    # three events
    X1_train = _get_inputs(input_keys, idx1)  # predictors
    y1_train = _get_output(idx1)              # predictand
    predict1 = m.predict(X1_train)
    #print(predict1.shape)
    mae1 = np.mean((predict1-y1_train)**2)
    #print(mae1)

    X2_train = _get_inputs(input_keys, idx2)  # predictors
    y2_train = _get_output(idx2)              # predictand
    predict2 = m.predict(X2_train)
    #print(predict2.shape)
    mae2 = np.mean((predict2-y2_train)**2)
    #print(mae2)

    X3_train = _get_inputs(input_keys, idx3)  # predictors
    y3_train = _get_output(idx3)              # predictand
    predict3 = m.predict(X3_train)
    #print(predict3.shape)
    mae3 = np.mean((predict3-y3_train)**2)
    #print(mae3)

    # persist model for forecast
    db[cs4ri_id] = (model, input_keys, m)

    return training_plots([
        (predict1, y1_train, mae1),
        (predict2, y2_train, mae2),
        (predict3, y3_train, mae3)
    ])



def training_plots(pv):

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)

    #ax1 = plt.subplot(311)
    ax1.plot(pv[0][1], 'k-', linewidth=0.5)
    ax1.plot(pv[0][0], 'r-', linewidth=0.25)
    ax1.text(0.1, 0.92, '%.4f ft' % pv[0][2], horizontalalignment='center', verticalalignment='center', transform = ax1.transAxes)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), fontsize='8')

    #ax2 = plt.subplot(312, sharex=ax1, sharey=ax1)
    ax2.plot(pv[1][1], 'k-', linewidth=0.5)
    ax2.plot(pv[1][0], 'r-', linewidth=0.25)
    ax2.text(0.1, 0.92, '%.4f ft' % pv[1][2], horizontalalignment='center', verticalalignment='center', transform = ax2.transAxes)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), fontsize='8')

    #ax3 = plt.subplot(313, sharex=ax1, sharey=ax1)
    ax3.plot(pv[2][1], 'k-', linewidth=0.5)
    ax3.plot(pv[2][0], 'r-', linewidth=0.25)
    ax3.text(0.1, 0.92, '%.4f ft' % pv[2][2], horizontalalignment='center', verticalalignment='center', transform = ax3.transAxes)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), fontsize='8')

    dpi = 300
    fig.set_alpha(0)
    fig.set_figheight(2200/dpi)
    fig.set_figwidth(1300/dpi)
    #fig.set_figheight(500/dpi)
    #fig.set_figwidth(300/dpi)

    ax1.set_ylim([-2,6])

    with io.BytesIO() as _buffer:
        fig.savefig(_buffer, dpi=dpi, bbox_inches='tight', pad_inches=0.0)#, transparent=True)
        plt.close(fig)
        _buffer.seek(0)
        return base64.b64encode(_buffer.getvalue())

def forecast_plot(predict):

    fig, ax1 = plt.subplots(1, sharex=True, sharey=True)

    ax1.plot(predict, 'r-', linewidth=0.25)
    ax1.axhline(y=predict.max(), color='b', linestyle='--', linewidth=0.25)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), fontsize='4')

    dpi = 300
    fig.set_alpha(0)
    fig.set_figheight(400/dpi)
    fig.set_figwidth(700/dpi)
    #fig.set_figheight(500/dpi)
    #fig.set_figwidth(300/dpi)

    ax1.set_ylim([-2,6])

    with io.BytesIO() as _buffer:
        fig.savefig(_buffer, dpi=dpi, bbox_inches='tight', pad_inches=0.0)#, transparent=True)
        plt.close(fig)
        _buffer.seek(0)
        return base64.b64encode(_buffer.getvalue())

