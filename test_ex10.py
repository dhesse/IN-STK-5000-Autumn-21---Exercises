import unittest
import tensorflow as tf
import numpy

from ex10_solution import Ex10

class TestEx10(unittest.TestCase):

    def test_task1(self):
        model = Ex10().task1()
        tf.random.set_seed(123)
        expected = numpy.array([-62.35868 , -67.963844])
        actual = model.log_prob(model.sample(2))
        numpy.testing.assert_array_almost_equal(expected, actual)

    def test_task2(self):
        log_prob = Ex10().task2()
        tf.random.set_seed(123)
        sample = Ex10().task1().sample(2)
        del sample['y']
        actual = log_prob(**sample).numpy()
        expected = numpy.array([-65.62223 , -74.086105])
        numpy.testing.assert_array_almost_equal(expected, actual)
    

    def test_task3(self):
        tf.random.set_seed(123)
        sr, mu_f, theta_f = Ex10().task3()
        self.assertAlmostEqual(sr[0], 1.0387455)
        self.assertAlmostEqual(sr[1], 1.0026323)
        numpy.testing.assert_array_almost_equal(
            sr[2], [1.0485951, 1.0156175, 1.0309933,
         1.0458852, 1.0165746, 1.0113516, 1.0085162, 1.0350327])
        self.assertAlmostEqual(mu_f, 0.93825)
        numpy.testing.assert_array_almost_equal(theta_f,
         [0.93075, 0.85125, 0.77925, 0.8635 ,
         0.71625, 0.78525, 0.94675, 0.8495])


if __name__ == "__main__":
    unittest.main()
