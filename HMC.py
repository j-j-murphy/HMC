#!/usr/bin/env python
# coding: utf-8

# In[34]:


np.random.seed(0)


# In[35]:


import numpy as np


# In[36]:


def leapfrog(x, v, gradient, timestep, trajectory_length):
    print("__________________")
    print("iter")
    print("x is :", x)
    print("v is :", v)
    print("*******")
    v -= 0.5 * timestep * gradient(x)
    for _ in range(trajectory_length - 1):
        print("x is :", x)
        print("v is :", v)
        print("Grad x", gradient(x))
        x += timestep * v
        v -= timestep * gradient(x)
    x += timestep * v
    v -= 0.5 * timestep * gradient(x)
    print("x is :", x)
    print("v is :", v)
    return x, v


# In[45]:


def sample_HMC(x_old, log_prob, log_prob_gradient, timestep, trajectory_length):
    # switch to physics mode!
    def E(x): return -log_prob(x)
    def gradient(x): return -log_prob_gradient(x)
    def K(v): return 0.5 * np.sum(v ** 2)
    def H(x, v): return K(v) + E(x)

    # Metropolis acceptance probability, implemented in "logarithmic space"
    # for numerical stability:
    def log_p_acc(x_new, v_new, x_old, v_old):
        return min(0, -(H(x_new, v_new) - H(x_old, v_old)))

    # give a random kick to particle by drawing its momentum from p(v)
    v_old = np.array([0.5, 0.5])

    # approximately calculate position x_new and momentum v_new after
    # time trajectory_length  * timestep
    x_new, v_new = leapfrog(x_old.copy(), v_old.copy(), gradient, 
                            timestep, trajectory_length)

    # accept / reject based on Metropolis criterion
    accept = np.log(np.random.random()) < log_p_acc(x_new, v_new, x_old, v_old)
    print("MH:", log_p_acc(x_new, v_new, x_old, v_old))

    # we consider only the position x (meaning, we marginalize out v)
    if accept:
        return accept, x_new
    else:
        return accept, x_old


# In[46]:


def build_HMC_chain(init, timestep, trajectory_length, n_total, log_prob, gradient):
    n_accepted = 0
    chain = [init]

    for _ in range(n_total):
        accept, state = sample_HMC(chain[-1].copy(), log_prob, gradient,
                                   timestep, trajectory_length)
        chain.append(state)
        n_accepted += accept

    acceptance_rate = n_accepted / float(n_total)

    return chain, acceptance_rate


# In[47]:


def log_prob(x): return -0.5 * np.sum(x ** 2)


# In[48]:


def log_prob_gradient(x): return -x


# In[49]:


chain, acceptance_rate = build_HMC_chain(np.array([5.0, 1.0]), 1.5, 10, 100,
                                         log_prob, log_prob_gradient)
print("Acceptance rate: {:.3f}".format(acceptance_rate))


# In[ ]:





# In[44]:


np.random.random()


# In[ ]:




