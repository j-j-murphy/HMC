#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np


# In[46]:


#building HMC object
class Hamiltonian_Monte_Carlo():
    def __init__(self, init):
        #initialise empty list to store values of x
        self.chain=[]
        #integers for tracking acceptance probability
        self.accepted, self.rejected, self.total = 0, 0, 0
        #initial values of position and momentum
        self.x_old, self.x_new, self.v_old, self.v_new = init, 0, 0, 0

    def leapfrog(self, epsilon, L, gradient):
        #copies x_old to x_new so x_new can be propagated through leapfrog
        self.x_new = self.x_old.copy()
        #draw from normal distribution for momentum value
        self.v_old = np.random.normal(size=self.x_new.shape)
        self.v_new = np.subtract(self.v_old, 0.5 * epsilon * gradient(self.x_new))
        
        for _ in range(L - 1):
            self.x_new = np.add(self.x_new, epsilon * self.v_new)
            self.v_new = np.subtract(self.v_new, epsilon * gradient(self.x_new))
            
        self.x_new = np.add(self.x_new, epsilon * self.v_new)
        self.v_new = np.subtract(self.v_new, 0.5 * epsilon * gradient(self.x_new))
        
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
        else:
            self.rejected += 1
            
        #appending x to MC
        self.chain.append(self.x_old)
        
    #cop out to work out Metropolis-Hastings acceptance value
    def H(self, x, v, log_prob):
        E = -log_prob(x)
        K = 0.5 * np.sum(v ** 2)
        return(K+E)


# In[63]:


def build_HMC_chain(HMC, epsilon, L, n_iter, log_prob, log_prob_gradient):
    def gradient(x): return -log_prob_gradient(x)
    
    for _ in range (n_iter):
        HMC.leapfrog(epsilon, L, gradient)
        HMC.acceptance(log_prob)


# In[60]:


HMC = Hamiltonian_Monte_Carlo(np.array([[5],
                                       [1]]))


# In[61]:


def log_prob_gradient(x): 
    return  -x


# In[62]:


def log_prob(x):
    return -0.5 * np.sum(x ** 2)


# In[64]:


build_HMC_chain(HMC, epsilon=1.5, L=10, n_iter=100, log_prob=log_prob, log_prob_gradient=log_prob_gradient)


# In[65]:


HMC.total


# In[66]:


HMC.accepted


# In[67]:


HMC.rejected


# In[ ]:




