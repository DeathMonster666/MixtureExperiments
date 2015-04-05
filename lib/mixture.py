import theano
import theano.tensor as T
import random
import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt

#Mixture of gaussians.  

#Mu is M dimensional vector, sigma is M dimensional vector
#C is M dimensional vector

#Compute p.  Maximize p for data.  

def gmmSample(mu, sigma, C, size): 

    assert len(mu) == len(sigma)
    assert len(sigma) == len(C)

    samples = []

    for i in range(0, size): 

        component = np.random.multinomial(1,C,size=1)[0].tolist().index(1)

        muPick = mu[component]
        sigmaPick = sigma[component]
        cPick = C[component]

        samples += [np.random.normal(muPick, sigmaPick)]

    return samples

numComponents = 50

samples = gmmSample([3.0,8.0,-8.0,-3.0], [0.5, 0.5,0.5,0.5], [0.25, 0.25, 0.25, 0.25], 1000)

learning_rate = 0.01

mu = theano.shared(np.random.normal(0,0.1, size = numComponents))
exp_sigma = theano.shared(0.0 * np.random.normal(0,0.1, size = numComponents))
C = theano.shared(np.random.normal(0,0.1,size = numComponents))

sigma = T.exp(exp_sigma)

params = [mu, C, exp_sigma]

#The true value for the sample
y = T.scalar()

#srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
#mask = srng.binomial(n=1, p=0.5, size=mu.shape)
mask = 1.0

normal_likelihood = mask * T.nnet.softmax(C) * (1.0 / sigma) * T.exp(-1.0 * ((y - mu)**2) / (2 * sigma**2))

print "likelihood dim", normal_likelihood.ndim

log_likelihood = T.log(T.sum(normal_likelihood))

updates = {}

for param in params: 
    updates[param] = param - learning_rate * T.grad(-1.0 * log_likelihood, param)

train = theano.function([y], outputs = [log_likelihood], updates = updates)

observe = theano.function([], outputs = [mu, sigma, T.nnet.softmax(C)[0]])

for i in range(0, 10000): 

    sample = samples[random.randint(0, len(samples) - 1)]

    likelihood = train(sample)

    print "likelihood", likelihood

    mu, sigma, C = observe()

    print "mu", mu
    print "sigma", sigma
    print "C", C


generatedSamples = gmmSample(mu, sigma, C, size = 2000)

bins = np.arange(-10, 10, 0.5)
plt.hist(samples, bins = bins, normed = True, alpha = 0.5, label = "observed")
plt.hist(generatedSamples, bins = bins, normed = True, alpha = 0.5, label = "generated")
plt.legend()
plt.show()


