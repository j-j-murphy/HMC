#!/usr/bin/env python
# coding: utf-8

# In[217]:


import numpy as np
import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad 
import matplotlib.pyplot as plt


# In[218]:


#building HMC object
class Hamiltonian_Monte_Carlo():

    def __init__(self, init):
        #initialise empty list to store values of x
        self.chain=[]
        #integers for tracking acceptance probability
        self.accepted, self.rejected, self.total = 0, 0, 0
        #initial values of position and momentum
        self.x_old, self.x_new, self.v_old, self.v_new = init, 0, 0, 0

    def leapfrog(self, epsilon, L, M, gradient):
        #copies x_old to x_new so x_new can be propagated through leapfrog
        self.x_new = self.x_old.copy()
        x_dim = self.x_new.shape[0]
        #draw from normal distribution for momentum value
        self.v_old = np.random.multivariate_normal(mean = np.zeros(x_dim),
                                                   cov = M
                                                  ).reshape(x_dim,1)
        self.v_new = np.subtract(self.v_old, 0.5 * epsilon * -gradient(self.x_new))
        
        for _ in range(L - 1):
            self.x_new = np.add(self.x_new, epsilon * self.v_new)
            self.v_new = np.subtract(self.v_new, epsilon * -gradient(self.x_new))
            
        self.x_new = np.add(self.x_new, epsilon * self.v_new)
        self.v_new = np.subtract(self.v_new, 0.5 * epsilon * -gradient(self.x_new))
        
    def acceptance(self, log_prob):
        self.total += 1
        #hamiltonian prior to leapfrog
        H_old = self.H(self.x_old, self.v_old, log_prob)
        #hamiltonian post leapfrog
        H_new = self.H(self.x_new, self.v_new, log_prob)
        Metropolis_Hastings = -(H_new-H_old)        
        
        if np.log(np.random.random()) < min(0, Metropolis_Hastings):
            self.accepted += 1
            self.x_old = self.x_new
            #appending x to MC
            self.chain.append(self.x_old)
        else:
            self.rejected += 1
                    
    #cop out to work out Metropolis-Hastings acceptance value
    def H(self, x, v, log_prob):
        E = -log_prob(x)
        K = 0.5 * np.sum(v ** 2)
        return(K+E)
    
    def prior_M_adapt(self, num_prior_samples, prior_var):
        num_samples = len(self.chain)
        current_var = np.cov(np.hstack(self.chain))
        #print(current_var)
        M = (num_samples*current_var+num_prior_samples*prior_var)/(num_samples+num_prior_samples)
        return(M)
    


# In[219]:


def build_HMC_chain(HMC, epsilon, L, M, n_iter, log_prob,
                    num_prior_samples=0, prior_var=0, prior_M=False):#, log_prob_gradient):
    gradient = grad(log_prob)
    #gradient = -grad(log_prob)
    for _ in range (n_iter):
        print(_)
        if prior_M==True:
            #fix current var
            if len(HMC.chain)<2:
                M=prior_var
            else:
                M = HMC.prior_M_adapt(num_prior_samples, prior_var)
            print(M)
        HMC.leapfrog(epsilon, L, M, gradient)
        HMC.acceptance(log_prob)


# In[220]:


HMC = Hamiltonian_Monte_Carlo(np.array([[5],
                                       [1]]))


# In[221]:


def log_prob(x):
    return -0.5 * np.sum(x ** 2.0)


# In[222]:


build_HMC_chain(HMC, epsilon=1.5, L=10, 
                M = np.array([[1,0],
                             [0,1]]),
                n_iter=1000, log_prob=log_prob,
               num_prior_samples=100, prior_var=np.array([[0.001,0],
                     [0,1]]), prior_M=True)#, 
                #log_prob_gradient=log_prob_gradient)


# In[223]:


HMC.total


# In[224]:


HMC.accepted


# In[225]:


HMC.rejected


# In[81]:


def plot_contour(x_start, x_end, num_points):
    x = np.linspace(x_start, x_end, num_points)
    X, Y = np.meshgrid(x, x)
    flattened_X = X.flatten()
    flattened_Y = Y.flatten()
    #getting the coordinates we want to graph across 
    coords = []
    for i in range(len(flattened_X)):
        xs = flattened_X[i]
        ys = flattened_Y[i]
        coords.append(np.array([xs, ys]))
    return (X, Y)


# In[185]:


prior_var = np.eye(50)


# In[201]:


init = np.ones(50).reshape(50,1)*2


# In[202]:


HMC = Hamiltonian_Monte_Carlo(init)


# In[205]:


build_HMC_chain(HMC, epsilon=0.1, L=10, 
                M = np.array([[1,5],
                             [5,1]]),
                n_iter=10000, log_prob=log_prob,
               num_prior_samples=10000, prior_var=prior_var, prior_M=True)


# In[206]:


HMC.accepted


# In[ ]:




