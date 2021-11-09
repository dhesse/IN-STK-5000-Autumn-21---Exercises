import numpy
from scipy.stats import poisson, gamma

class Ex11:

    """We will solve various tasks using a data set containing
    synthetic data drawn from a poisson distribution. You have to write the code
    to complete the tasks. The data set is created for you, no need to do
    anything with this. As in the classroom, we will run a number of simulations
    in parallel.
    """

    N = 1_000 # time steps
    sims = 1_000 # number of parallel simulations
    thetas = numpy.array([3, 3.5, 4])
    k = thetas.shape[0]

    def __init__(self):
        self.observations = poisson(self.thetas)\
            .rvs((self.N, self.sims, self.k))
        
        self.alphas = numpy.ones((self.sims, self.k))
        self.betas = numpy.ones((self.sims, self.k))
        self.idx = numpy.arange(self.sims)
        

    def task1(self):
        """Implement the first step in Thompson sampling using a Gamma
        distributed prior starting with unit shape and rate. Return the selected
        action."""
        theta_hat = gamma(self.alphas, scale=1/self.betas)\
            .rvs((self.sims, self.k))
        action = numpy.argmax(theta_hat, axis=1)
        return action
    
    def task2(self, action, t):
        """Given an action, and a time step t, return the reward."""
        return self.observations[t,self.idx,action]
    
    def task3(self, action, reward):
        """Given a reward, update the prior."""
        self.alphas[self.idx, action] += reward
        self.betas[self.idx, action] += 1
    
    def task4(self, action):
        """Given an action, calculate the per-period regret."""
        return self.thetas.max() - self.thetas[action]
