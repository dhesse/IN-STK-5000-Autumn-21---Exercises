import unittest
import numpy
import pandas

from ex6_solution import Ex6

class TestEx6(unittest.TestCase):

    def test_task1(self):

        trf = Ex6().task1(0.4, 0.6)
        numpy.random.seed(123)
        actual = trf.fit_transform(numpy.ones(200)).mean()
        self.assertAlmostEqual(actual, 0.725)


    def test_task2(self):
        trf = Ex6().task2(0.4, 0.6)
        numpy.random.seed(123)
        df = pandas.DataFrame(
            {'sex': numpy.ones(100),
            'chol': numpy.random.uniform(0, 100, 100)})
        actual = trf.fit_transform(df)
        mean = actual.mean(axis=0)
        std = actual.std(axis=0)
        self.assertAlmostEqual(mean[0], 0.74)
        self.assertAlmostEqual(mean[1], 0)
        self.assertAlmostEqual(std[1], 1)
    
    def test_task3(self):
        numpy.random.seed(123)
        actual = Ex6().task3(0.7, 0.3)
        self.assertAlmostEqual(actual.mean(), 0.58419354838)


if __name__ == "__main__":
    unittest.main()