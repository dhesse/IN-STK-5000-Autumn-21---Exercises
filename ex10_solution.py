import numpy

class Ex10:

    """We will solve various tasks using a data set containing
    the '8 schools' data set. Eight schools were selected to have students
    undergo training for their SATS testing and calculated effects and 
    standard deviation. You have to write the code to complete the tasks. 
    The data set is loaded for you, no need to do anything with this.
    """

    num_schools = 8
    treatment_effects = numpy.array(
       [28, 8, -3, 7, -1, 1, 18, 12], dtype=numpy.float32)  # treatment effects
    treatment_stddevs = numpy.array(
        [15, 10, 16, 11, 9, 11, 10, 18], dtype=numpy.float32)  # treatment SE

    def task1(self):
        """Create a JointDistributionNamedAutoBatched model that has
        mu ~ Normal(0, 7)
        tau ~ LogNormal(2, 0.5)
        theta_i ~ Normal(mu, tau), i = 1,...,8
        treatment_effects_i ~ Normal(theta_i, treatment_stddevs_i), i=1,...,8
        """
        pass
    
    def task2(self):
        """Return the log prob function for the model created in task1,
        accepting the parameters mu, tau, and theta, assuming
        self.treatment_effects to be the observed effects."""
        pass
    
    def task3(self):
        """Perform an MCMC estimate of the model parameters. Use a
        HamiltonianMonteCarlo kernel with 5 leapfrog steps and step size 0.9.
        Run 4 chains in parallel, with 1000 measurements and 500 burn-in steps.
        As starting parameters, use a sample from the model of size 4.
        Return a tuple containing the potential scale reduction, the fraction of
        samples for mu that are strictly greater than zero, and an array
        indicating the fraction of samples for each theta_i that are strictly
        greater than zero."""
        pass