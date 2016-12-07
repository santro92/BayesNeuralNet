from __future__ import division
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l2
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.cross_validation import train_test_split

np.random.seed(seed = 2456)
def get_params(params):
    """
    Values in params are numpy array, we need to get a list out of it
    :param params:
    :return: a new dict
    """
    parsed_params = {}
    for k, v in params.iteritems():
        vv = v
        if isinstance(v, np.ndarray):
            vv = v.tolist()
            if len(vv) == 1:
                vv = vv[0]
            if isinstance(v, basestring) and len(v) == 0:
                vv = None
        elif isinstance(v, list) and len(v) == 1:
            vv = vv[0]

        if isinstance(vv, basestring) and len(vv) == 0:
            vv = None

        parsed_params[k] = vv
    return parsed_params


def build_nn(neurons, input_dim, activation1, activation2, w_init, errFunc, wd, dropout):
    model = Sequential()
    model.add(Dense(neurons, input_dim=input_dim, init=w_init, W_regularizer=l2(wd), activation=activation1))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation=activation2))
    model.compile(loss=errFunc, optimizer='rmsprop')
    return model


def z_score_inputs(Inputs, Mean, StdDev):
    Inputs = np.divide((Inputs - Mean), StdDev)
    return Inputs


def find_best_model(train_data, train_labels, param_grid):
    model = KerasRegressor(build_fn=build_nn)
    regressor = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=10, n_jobs=1)
    reg_results = regressor.fit(train_data, train_labels)

    print("Best AUC score: %f" % reg_results.best_score_)
    print("Best Parameters: %s" % reg_results.best_params_)

    return reg_results.best_estimator_


def get_data():

    trainData = pd.read_csv('./data/train.csv')
    trainData = trainData.replace(-1, np.nan)
    trainData = trainData.fillna(trainData.mean())
    trainData = trainData.as_matrix()

    trainLabel = pd.read_table('./data/train_labels.txt', header=None)
    trainLabel = trainLabel.as_matrix()

    [x_train, x_val, y_train, y_val] = train_test_split(trainData, trainLabel, test_size=0.30)

    trainMean = np.mean(x_train, axis=0)
    trainStd = np.array(list(np.std(x_train,axis=0,dtype=np.float32)))
    x_train = z_score_inputs(x_train, trainMean, trainStd)
    x_val = z_score_inputs(x_val, trainMean, trainStd)

    return x_train, y_train, x_val, y_val

def ret_params_model(x_train, y_train, x_val, y_val):

    param_grid = {'neurons': range(10,101,10), 'input_dim': [x_train.shape[1]], 'activation1': ['sigmoid','relu'], 'activation2': ['relu'], 'w_init': ['glorot_uniform','glorot_normal'],
                    'errFunc': ['mse'], 'wd': [0,0.0001,0.001,0.01], 'dropout': [x / 100.0 for x in range(25,76,5)],'nb_epoch':range(30,101,10)}
    model = find_best_model(x_train, y_train, param_grid)
    model.fit(x_train, y_train,
              nb_epoch=500,
              batch_size=32,
              shuffle=True)
    val_loss = mean_squared_error(y_val, model.predict(x_val))
#               validation_data=(x_val, y_val))
    return val_loss

def fit_model_test(params):

    x_train, y_train,x_val,y_val = get_data()
    model = build_nn(params['neurons'] , x_train.shape[1], params['activation1'],params['activation2'],params['w_init'],'mse', params['wd'], params['dropout'])
    model.fit(x_train, y_train,
              nb_epoch=params['nb_epochs'],
              batch_size=32,
              shuffle=True)
#               validation_data=(x_val, y_val))

    val_loss = mean_squared_error(y_val, model.predict(x_val))
    print(val_loss)
    return model, val_loss

def main(job_id, params):
    print params
    params = get_params(params)
    _, val_loss = fit_model_test(params)
    return val_loss

if __name__ == "__main__": 
    
    x_train, y_train, x_val, y_val = get_data()
    print ret_params_model(x_train, y_train, x_val, y_val)

    # x,_,_,_ = get_data()
#     param_grid = {'neurons': 20, 'input_dim': x.shape[1], 'activation1': 'tanh','activation2':'relu', 'w_init': 'glorot_normal','errFunc': 'mse', 'wd': 0.01, 'dropout': 0.5,'nb_epoch':20}
#     fit_model_test(param_grid)
