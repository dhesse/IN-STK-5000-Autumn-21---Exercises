import pandas as pd

class Ex7:

    """We will solve various tasks using a data set containing
    patients suffering from heart disease, again. You have to write the code to
    complete the tasks. The data set is loaded for you, no need to do anything
    with this.
    """

    d = pd.read_csv("heart.csv.gz")

    def task1(self):
        """Laplae mechanism.
        
        Return a function that implements the Laplace mechanism. It will accept
        three arguments, the answer, the sensitivity, and the privacy parameter
        epsilon, in that order, and return the randomized response. Use numpy's
        Laplace function for reproducibility."""

        import numpy
        def laplace(answer, sensitivity, epsilon):
            return answer + numpy.random.laplace(0, sensitivity/epsilon)

        return laplace

    def task2(self):
        """Randomized query.
        
        Using the Laplace mechanism from the previous task, setting sensitivity
        and epsilon to 1, answer the question how many of the patients over 60
        years of agesuffer from heart disease."""

        laplace = self.task1()
        return laplace(self.d[self.d['age'] > 60]['target'].sum(), 1, 1)

    def task3(self, epsilon):
        """Private regression pipeline.
        
        Return a pipeline that scales the full input data to zero mean and unit
        variance and fits a LogisticRegression from the diffprivlib package.
        Since we scale the data we can use 1 as the data norm.
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from diffprivlib.models import LogisticRegression

        return Pipeline([('scale', StandardScaler()),
         ('clf', LogisticRegression(data_norm=1, epsilon=epsilon))])
    
    def task4(self):
        """Explore model performance.
        
        Calculate average cross validation scores (5 folds), using the full
        dataset, of the model pipeline created in the previous task. Scan 
        epsilon values between 0.1 and 5, in 10 even steps.
        
        You are encouraged to also plot the results, and explore the variance
        in performance."""
        import numpy
        from sklearn.model_selection import cross_val_score
        es = numpy.linspace(.1, 5, 10)
        return [cross_val_score(self.task3(e),
                    self.d.drop(columns='target'), self.d['target']).mean()
                    for e in es]
