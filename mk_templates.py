import argparse

def mk_solution_template(k: int, data_desc: str, csv_file: str) -> None:
    return f"""import pandas as pd

class Ex{k}:

    \"""We will solve various tasks using a data set containing
    {data_desc}. You have to write the code to complete the tasks. The 
    data set is loaded for you, no need to do anything with this.
    \"""

    d = pd.read_csv("{csv_file}")

    def task1(self):
        pass
"""

def mk_test_template(k: int) -> None:
    return f"""import unittest

from ex{k}_solution import Ex{k}

class TestEx{k}(unittest.TestCase):

    def test_task1(self):
        pass

if __name__ == "__main__":
    unittest.main()
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('k', type=int)
    parser.add_argument('data_desc')
    parser.add_argument('csv_file')
    args = parser.parse_args()
    with open(f"ex{args.k}_solution.py", "x") as ex_f:
        ex_f.write(mk_solution_template(args.k, args.data_desc, args.csv_file))
    with open(f"test_ex{args.k}.py", "x") as te_f:
        te_f.write(mk_test_template(args.k))
