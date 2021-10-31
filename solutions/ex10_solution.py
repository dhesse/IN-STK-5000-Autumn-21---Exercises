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

        import tensorflow_probability as tfp
        import tensorflow as tf

        dist = tfp.distributions
        return dist.JointDistributionNamedAutoBatched({
            'mu': dist.Normal(0, 7),
            'tau': dist.LogNormal(2, .5),
            'theta': lambda mu, tau: dist.Independent(
                dist.Normal(tf.ones(self.num_schools)*mu, tau),
                reinterpreted_batch_ndims=1),
            'y': lambda theta: dist.Independent(
                dist.Normal(theta, self.treatment_stddevs),
                reinterpreted_batch_ndims=1)})
    
    def task2(self):
        """Return the log prob function for the model created in task1,
        accepting the parameters mu, tau, and theta, assuming
        self.treatment_effects to be the observed effects."""
        model = self.task1()
        def log_prob(mu, tau, theta):
            return model.log_prob({
                'mu': mu,
                'tau': tau,
                'theta': theta,
                'y': self.treatment_effects})
        return log_prob
    
    def task3(self):
        """Perform an MCMC estimate of the model parameters. Use a
        HamiltonianMonteCarlo kernel with 5 leapfrog steps and step size 0.9.
        Run 4 chains in parallel, with 1000 measurements and 500 burn-in steps.
        As starting parameters, use a sample from the model of size 4.
        Return a tuple containing the potential scale reduction, the fraction of
        samples for mu that are strictly greater than zero, and an array
        indicating the fraction of samples for each theta_i that are strictly
        greater than zero."""
        model = self.task1()
        log_prob = self.task2()

        import tensorflow_probability as tfp
        import tensorflow as tf

        @tf.function
        def sample_chain(nobs, nburn, nchains):
            sample = model.sample(nchains)
            kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=log_prob,
                step_size=0.9,
                num_leapfrog_steps=5)
            return tfp.mcmc.sample_chain(
                num_results=nobs,
                num_burnin_steps=nburn,
                kernel=kernel,
                current_state=[
                    sample['mu'],
                    sample['tau'],
                    sample['theta']]
            )
        tr, kr = sample_chain(1000, 500, 4)
        return (
            tfp.mcmc.potential_scale_reduction(tr),
            (tr[0].numpy().reshape(-1) > 0).mean(),
            (tf.reshape(tr[-1], (-1, tr[-1].shape[-1])).numpy() > 0
             ).mean(axis = 0)
        )
