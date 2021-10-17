import pandas as pd

class Ex8:

    """We will solve various tasks using a data set containing
    COMPAS data (https://github.com/propublica/compas-analysis).
    It contains information about defendants that appeared before a
    US court, together with an assessment of their risk to commit another
    crime. Propublica has added a column giving us information if an 
    individual did indeed commit another crime in the coming 2 years.
    You have to write the code to complete the tasks. The 
    data set is loaded for you, no need to do anything with this.
    """

    d = pd.read_csv("compas-scores-two-years-filtered.csv.gz")

    def __init__(self):
        # you might want to calculate re-used quantities here ...
        pass

    def task1(self):
        """Given the predicted outcome yhat (corresponding to the 'score_text'
        field being 'Low', and the sensitive variable z (corresponding to 
        the 'race' column being equal to 'African-American', what is the
        maximum likelihood estimate of the probability P(yhat | z = 1)?"""
        pass
    
    def task2(self):
        """With the definitions from task 1, what is the MLE of the probability
        P(yhat | z = 0)?"""
        pass
    
    def task3(self):
        """With the results from task 1 and task 2, calculate the p-percent
        score."""
        pass
    
    def task4(self):
        """With the definitions from task 1, and the outcome y, which we will
        take as the 'two_year_recid' column (did re-offend within 2 years) being
        equal to 0 (i.e. did not re-offend). Calculate the MLE for
        P(yhat | z = 1, y = 1)."""
        pass
    
    def task5(self):
        """With the definitions from task 1, and the outcome y, which we will
        take as the 'two_year_recid' column (did re-offend within 2 years) being
        equal to 0 (i.e. did not re-offend). Calculate the MLE for
        P(yhat | z = 0, y = 1)."""
        pass
    
    def task6(self):
        """With the results from task 4 and task 5, calculate the equal
        opportunity score."""
        pass

        
