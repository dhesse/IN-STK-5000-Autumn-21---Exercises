import pandas as pd


class Ex6:

    """We will solve various tasks using a data set containing information
    about patients suffering from heart disease.
    You have to complete the code to complete the tasks. The 
    data set is loaded for you, no need to do anything with this.

    More information about the data can be found in the UCI ML repository,
    https://archive.ics.uci.edu/ml/datasets/Heart+Disease
    """

    d = pd.read_csv("heart.csv.gz")

    def task1(self, theta, p):
        """Return an sklearn FunctionTransformer that implements the randomized
        mechanism. Each response shall be truthful with probability theta. If
        the true value is not revealed, the random response shall be drawn from
        a Bernoulli distribution with P(X = 1) = p and P(X = 0) = 1-p."""

        pass

    def task2(self, theta, p):
        """Return an sklearn ColumnTransformer that will transform a dataset
        as follows:

        The 'chol' column, representing a patient's cholesterol level shall be
        scaled to have zero mean and unit stanard deviation.

        The 'sex' column, representing the patient's gender shall because of
        its sensitive nature be randomized with parameters theta and p, as 
        explained in task 1."""

        pass

    def task3(self, theta, p):
        """Incorporate the transformer created in task 2 to make a
        classification pipeline using a support vector classifier (default
        constructor arguments). Run a 10-fold cross validation on the heart
        disease data and return the results.

        The classification target is simply called 'target' in the data,
        indicating presence (1) or absence (0) of hear disease."""

        pass
