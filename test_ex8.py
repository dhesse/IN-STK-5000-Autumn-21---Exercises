import unittest

from ex8_solution import Ex8

class TestEx8(unittest.TestCase):

    def test_task1(self):
        actual = Ex8().task1()
        expected = 0.41179653679653677
        self.assertAlmostEqual(actual, expected)
    
    def test_task2(self):
        actual = Ex8().task2()
        expected = 0.6519967400162999
        self.assertAlmostEqual(actual, expected)
    
    def test_task3(self):
        actual = Ex8().task3()
        expected = 0.6315929383116883
        self.assertAlmostEqual(actual, expected)

    def test_task4(self):
        actual = Ex8().task4()
        expected = 0.5515320334261838
        self.assertAlmostEqual(actual, expected)

    def test_task5(self):
        actual = Ex8().task5()
        expected = 0.7654569892473119
        self.assertAlmostEqual(actual, expected)

    def test_task6(self):
        actual = Ex8().task6()
        expected = 0.7205264844057607
        self.assertAlmostEqual(actual, expected)

if __name__ == "__main__":
    unittest.main()
