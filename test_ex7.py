import unittest
import numpy
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from diffprivlib.models import LogisticRegression

from ex7_solution import Ex7

class TestEx7(unittest.TestCase):

    def test_task1(self):
        laplace = Ex7().task1()
        actual = laplace(0, 1, 1)
        self.assertAlmostEqual(actual, -0.9596187264010436)

    def test_task2(self):
        actual = Ex7().task2()
        self.assertAlmostEqual(actual, 44.28000169828538)
    
    def test_task3(self):
        model = Ex7().task3(1)
        self.assertIsInstance(model, Pipeline)
        self.assertIsInstance(model.steps[0][1], StandardScaler)
        self.assertIsInstance(model.steps[1][1], LogisticRegression)
    
    def test_task4(self):
        # diffprivlib's global_seed doesn't seem to work from an import, so
        # it's a bit hard to test this ...
        pass

if __name__ == "__main__":
    numpy.random.seed(1234)
    unittest.main()