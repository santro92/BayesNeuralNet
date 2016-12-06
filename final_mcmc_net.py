from __future__ import division
import pandas as pd
from scipy.stats import gamma, lognorm
from elliptical_slice import *
from slice_update import *
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error


class Data:
    'Class for data object'
    
    def __init__(self, trainFile, trainLabelFile, valRatio):
        self.trainFile = trainFile
        self.trainLabelFile = trainLabelFile
        self.valRatio = valRatio
        
    def inputPreprocess(self):
        trainData = pd.read_csv(self.trainFile)
        trainLabel = pd.read_table(self.trainLabelFile,header = None)
        # The number of input features  

        trainData = trainData.replace(-1,np.nan)
        self.features = trainData.shape[1];
        trainData = trainData.fillna(trainData.mean())
        self.trainInputs = trainData.as_matrix()
        self.trainTargets = trainLabel.as_matrix()
        [self.trainInputs, self.valInputs, self.target, self.valTargets] = train_test_split(self.trainInputs,self.trainTargets ,test_size = self.valRatio)
        self.trainMean = np.mean(self.trainInputs,axis=0)
        tnStd = np.std(self.trainInputs,axis=0,dtype = np.float32)
        self.trainStd = np.array(list(tnStd))
        self.input = z_score_inputs(self.trainInputs,self.trainMean,self.trainStd);
        self.valInputs = z_score_inputs(self.valInputs,self.trainMean,self.trainStd);
        
    def select_subset(self, start, number):
        d = Data('./data/train.csv','./data/train_labels.txt',0.30)
        d.input = self.input[start:start+number,:]
        d.target = self.target[start:start+number,:]
        return d

def z_score_inputs(Inputs, Mean, StdDev):
    """ 
        Pre-process inputs by making it mean zero and unit standard deviation
    """
    Inputs = np.divide((Inputs-Mean),StdDev)
    return Inputs

class Weights:
    """ Class that contains weight vectors for each layer"""
    
    def __init__(self,input_dim = 451, hidden_neurons = 32, output_dim =1):
        # Sets gaussian prior to weight vector
        self.Wl1 = np.random.normal(0, .1, (input_dim, hidden_neurons))
        self.Wl2 = np.random.normal(0, .1, (hidden_neurons, output_dim))
        

def sigmoid(input, w):
# Returns the output of the neuron after applying sigmoid activation
    z = np.dot(w.T,input.T).T
    return 1/(1 + np.exp(z))


def relu(input,w):
    z = np.dot(w.T,input.T).T
    return np.maximum(0,z)
    
    
def predict(w_obj, data_obj):
    # Give output of one feedforward pass given the network
    y_1 = sigmoid(data_obj.input, w_obj.Wl1)
    y_2 = relu(y_1, w_obj.Wl2)
    return y_2
        
        
def tau_prior(a, b):
    return np.random.gamma(a,b)


def tau_llh(tau, prior_a, prior_b, data_obj, w_obj):
    n = data_obj.input.shape[0]
    a = prior_a + n/2
    b = prior_b + np.sum((data_obj.target - predict(w_obj, data_obj)),axis = 0)
    return gamma.logpdf(tau, a = a, scale = 1/b)
    
    
def obs_llh(w_vec, idx, layer, data_obj, w_obj, tau):
    if layer == 1:
        w_obj.Wl1[idx,:] = w_vec
    else:
        w_obj.Wl2[idx,:] = w_vec

    mu = predict(w_obj, data_obj)
    s = np.sqrt(1/tau)
#     print data_obj.target/np.exp(mu)
    print data_obj.target.shape
    print (data_obj.target/np.exp(mu))
    return lognorm.logpdf(data_obj.target/np.exp(mu), s)
    
    
def mcmc_fit(data_obj, w_obj, tau, prior_a, prior_b, mcmc_iter = 1000, sigma0=.1):
    
    for iter in range(mcmc_iter):
        ct = 0
        print iter
        for w_mat in [w_obj.Wl1, w_obj.Wl2]:
            ct+=1 
            for neuron in range(w_mat.shape[0]):
                prior = np.random.normal(0, sigma0, w_mat.shape[1])
#                 print prior.shape
#                 print w_mat[neuron,:].shape
                print "layer = %d" %ct
                print neuron
                print w_mat[neuron,:]
#                 print obs_llh(w_mat[neuron,:],neuron, ct, data_obj, w_obj, tau)
                w_mat[neuron,:], next_llh = elliptical_slice(w_mat[neuron,:],prior,obs_llh,pdf_params=(neuron, ct, data_obj, w_obj, tau),cur_lnpdf=None,angle_range=None)
#                 print w_mat[neuron,:]
#                 print tau
        tau, llh_val = slice_update(tau, tau_llh, bounds = [0,10],pdf_params=(prior_a, prior_b, data_obj, w_obj))
        
    return w_obj, tau
        
        
def main():

    pri_a = 2
    pri_b = 2
    tau = tau_prior(pri_a, pri_b)
    w_obj = Weights()
    data_obj = Data('./data/train.csv','./data/train_labels.txt',0.30);
    data_obj.inputPreprocess();
    
    for i in range(data_obj.input.shape[0]):
        d_obj = data_obj.select_subset(i,1)
        w_obj, tau = mcmc_fit(d_obj, w_obj, tau, pri_a, pri_b)
        
    val_pred = predict(w_obj, data_obj.valInputs)
    val_loss = mean_squared_error(data_obj.valTargets,val_pred)
    return val_loss
    
if __name__ == "__main__":
    main()