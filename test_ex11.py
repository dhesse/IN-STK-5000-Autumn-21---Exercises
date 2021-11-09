import unittest
import numpy

from ex11_solution import Ex11

class TestEx11(unittest.TestCase):

    def test_task1(self):
        Ex11.sims = 10
        numpy.random.seed(123)
        actual = Ex11().task1()
        expected = numpy.array([2, 1, 2, 1, 0, 1, 2, 2, 0, 2])
        numpy.testing.assert_array_equal(actual, expected)
    
    def test_task2(self):
        Ex11.sims = 10
        numpy.random.seed(123)
        actual = Ex11().task2(0, 10)
        expected = numpy.array([4, 2, 2, 4, 4, 3, 3, 4, 6, 0])
        numpy.testing.assert_array_equal(actual, expected)

    def test_task3(self):
        Ex11.sims = 2
        numpy.random.seed(123)
        sim = Ex11()
        reward = sim.task2(0, 10)
        sim.task3(0, reward)
        expected = numpy.array([[4., 1., 1.],
                                [2., 1., 1.]])
        numpy.testing.assert_array_equal(sim.alphas, expected)
        expected = numpy.array([[2., 1., 1.],
                                [2., 1., 1.]])
        numpy.testing.assert_array_equal(sim.betas, expected)
    
    def test_task4(self):
        Ex11.sims = 10
        numpy.random.seed(123)
        actual = Ex11().task4([0]*Ex11.sims)
        expected = numpy.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
        numpy.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
