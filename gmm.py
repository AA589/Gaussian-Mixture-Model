import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal as mvn
import copy
from sklearn.cluster import KMeans

class custom_gmm():
    #initialize gmm for dataset and k classes
    def __init__(self, data, k, cov_init=1, kmeans=True):
        self.data = copy.deepcopy(data)
        self.k = k
        self.n = len(self.data)
        self.f = len(self.data[0])
        #run kmeans and use centroids for decent mu initialization
        if kmeans:
            kmeans = KMeans(n_clusters=k).fit(self.data)
            self.mus = kmeans.cluster_centers_
        else:
            self.mus = self.data[np.random.choice(np.arange(self.n), self.k)]
        #initialize k covariance matrices as identity matrices
        self.covs = np.array([np.asmatrix(np.identity(self.f)) for i in range(self.k)]) * cov_init
        #intialize all mixes as equal
        self.mixes = np.ones(self.k) / self.k
        #initialize class likelihood probabilities as empty
        self.probs = np.empty((self.n, self.k))
        
    #assign probabilities each data point is in each class
    def e_step(self):
        s = 0
        for i in range(self.k):
            p = mvn.pdf(self.data, self.mus[i,:].ravel(), self.covs[i,:]) * self.mixes[i]
            s += p
            self.probs[:,i] = p
        self.probs = self.probs / s[:,None]
        
    #update mixer, mu, and cov matrix for each class
    def m_step(self):
        for i in range(self.k):
            #update mixer
            s = self.probs[:,i].sum()
            self.mixes[i] = s / self.n 
            #update mu
            new_mu = self.data.T @ self.probs[:, i]
            self.mus[i] = new_mu.reshape(1,self.f) / s
            #update cov
            dif = self.data - self.mus[i]
            diag = np.diag(np.diag(np.diag(self.probs[:,i])))
            new_cov = (diag @ dif).T @ dif
            self.covs[i] = new_cov / s
            
    #get log likelihood of model at its current stage
    def log_like(self):
        like = 0
        for i in range(self.k):
            like += mvn.pdf(self.data, self.mus[i,:].ravel(), self.covs[i,:]) * self.mixes[i]
        ll = np.log(like).sum()
        return ll
    
    #fit k gaussians to data until convergence within threshold
    def fit(self, thresh = 0.0001):
        like = 1
        prev_like = 0
        #perform e step then m step until log likelihood converges
        while np.abs(like - prev_like) > thresh:
            prev_like = self.log_like()
            self.e_step()
            self.m_step()
            like = self.log_like()
            
    #predict class and probability of new data
    def predict(self, data):
        s = 0
        probs = np.asmatrix(np.empty((len(data), self.k)))
        for i in range(self.k):
            p = mvn.pdf(data, self.mus[i,:].ravel(), self.covs[i,:])
            s += p[:,None]
            probs[:,i] = p[:,None]
        return pd.DataFrame(np.hstack((np.argmax(probs, axis=1), np.max(probs, axis=1) / s)))
    
            

    #return class predictions for each data point  
    def get_labs(self):
        return pd.DataFrame(np.argmax(self.probs, axis=1))
    
    #return whether a data point is an anomaly or not
    #Checks if assignment prob is less than a condifence
    def get_anoms(self, conf=0.05):
        labs = []
        Ps = np.max(self.probs, axis=1)
        for p in Ps:
            if p < conf:
                labs.append(1)
            else:
                labs.append(0)
        return pd.DataFrame(labs)