import unittest

import ex1_solution

class TestEx1(unittest.TestCase):

    def test_task1(self):

        l = list("abc")
        for n, item in enumerate(l):
            self.assertEqual(ex1_solution.task1(l, n), item)
        self.assertEqual(ex1_solution.task1(l, len(l)), None)

if __name__ == "__main__":
    unittest.main()

        