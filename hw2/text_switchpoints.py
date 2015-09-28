#!/usr/bin/python

# Alex Pine (akp258@nyu.edu)
# Inference and Representation
# September 27 2015

import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import theano.tensor as t

# TODO do I need an __all__ var?

# here is the trick
@pm.theano.compile.ops.as_op(itypes=[t.lscalar, t.lscalar, t.dscalar, t.dscalar,
                                     t.dscalar],
                             otypes=[t.dvector])
def rateFunc(first_switchpoint, second_switchpoint, first_mean, second_mean,
             third_mean):
    ''' Concatenate Poisson means '''
    out = np.empty(n_count_data)
    out[:first_switchpoint] = first_mean
    out[first_switchpoint:second_switchpoint] = second_mean
    out[second_switchpoint:] = third_mean
    return out


count_data = np.loadtxt('text_data.csv')
n_count_data = len(count_data)

# Data plotting code
# plt.bar(np.arange(n_count_data), count_data, color="#348ABD")
# plt.xlabel("Time (days)")
# plt.ylabel("count of text-msgs received")
# plt.title("Did the user's texting habits change over time?")
# plt.xlim(0, n_count_data);

with pm.Model() as text_model:
    first_switchpoint = pm.DiscreteUniform('first_switchpoint', lower=0,
                                           upper=n_count_data)
    # TODO force the second after the first? Hangs when I do this
    second_switchpoint = pm.DiscreteUniform('second_switchpoint',
                                            lower=0,
                                            upper=n_count_data) 

    # NOTE the setting of alpha seems to make a huge difference? not a good sign
    alpha = 1.0 / count_data.mean()
    first_mean = pm.Exponential('first_mean', lam=alpha)
    second_mean = pm.Exponential('second_mean', lam=alpha)
    third_mean = pm.Exponential('third_mean', lam=alpha)

    #TODO    rate = pm.switch(first_switchpoint >= count_data, first_mean, second_mean)
    rate = rateFunc(first_switchpoint, second_switchpoint, first_mean, second_mean,
                    third_mean)

    text_count = pm.Poisson('text_count', rate, observed=count_data)
    
    #TODO    step1 = pm.NUTS([first_mean, second_mean, third_mean])
    step1 = pm.Slice([first_mean, second_mean, third_mean])
    # Use Metropolis for switchpoint, and missing values since it accomodates discrete variables
    step2 = pm.Metropolis([first_switchpoint, second_switchpoint])
    trace = pm.sample(10000, step=[step1, step2])
    pm.summary(trace)
    pm.traceplot(trace)
    plt.show()

    
# NOTE: tentative results
# switchpoint 1: about day 33
# switchpoint 2: about day 61
# rate 1: about 10.67 per day
# rate 2: about 14.67 per day
# rate 3: about 10 per day



