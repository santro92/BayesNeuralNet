import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l2
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.cross_validation import train_test_split


def build_nn(neurons, input_dim, activation, optimizer, errFunc, wd, dropout):
    model = Sequential()
    model.add(Dense(neurons[0], input_dim=input_dim, init='glorot_uniform', W_regularizer=l2(wd), activation=activation[0]))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation=activation[1]))
    model.compile(loss=errFunc, optimizer=optimizer)
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


def train_and_test():

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

    param_grid = {'neurons': [[50],[40],[30],[20]], 'input_dim': [trainData.shape[1]], 'activation': [['tanh','relu']], 'optimizer': ['rmsprop'],
                  'errFunc': ['mse'], 'wd': [0,0.0001,0.001,0.01], 'dropout': [0.25,0.5],'nb_epoch':[20,30,40,50]}
    model = find_best_model(x_train, y_train, param_grid)
    # model.fit(x_train, y_train,
    #           nb_epoch=500,
    #           batch_size=32,
    #           shuffle=True,
    #           validation_data=(x_val, y_val))

def main(job_id, params):
    print params
    return np.random.randn()

if __name__ == "__main__":
    train_and_test()