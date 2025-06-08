import unittest
import numpy as np
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.part_1 import (  # noqa: E402
    create_gd,
    create_gk,
    max_degree,
    number_of_connectivity_components,
    size_max_clique,
    size_max_independent_set,
)


class TestStub(unittest.TestCase):
    def test_always_pass(self):
        self.assertTrue(True, "Always pass")

    def test_size_max_independent_set(self):
        x2 = np.array([1, 2, 5, 6])
        n = len(x2)
        d = 1
        gd = create_gd(x2, n, d)
        assert size_max_independent_set(n, gd) == 2

    def test_number_of_connectivity_components(self):
        x2 = np.array([1, 2, 5, 6])
        n = len(x2)
        d = 1
        gd = create_gd(x2, n, d)
        assert number_of_connectivity_components(gd) == 2

    def test_max_degree(self):
        x = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        n = len(x)
        k = 2
        gk = create_gk(x, n, k)
        assert max_degree(n, gk) == 2

    def test_size_max_clique(self):
        x = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 2]])
        n = len(x)
        k = 2
        gk = create_gk(x, n, k)
        assert size_max_clique(gk) >= 2

    def test_empty_graph(self):
        gd = np.zeros((0, 0), dtype=bool)
        assert size_max_independent_set(0, gd) == 0
        assert number_of_connectivity_components(gd) == 0

        gk = np.zeros((0, 0), dtype=bool)
        assert max_degree(0, gk) == 0
        assert size_max_clique(gk) == 0

    def test_single_node(self):
        gd = np.zeros((1, 1), dtype=bool)
        assert size_max_independent_set(1, gd) == 1
        assert number_of_connectivity_components(gd) == 1

        gk = np.zeros((1, 1), dtype=bool)
        assert max_degree(1, gk) == 0
        assert size_max_clique(gk) == 1

    def test_complete_graph(self):
        n = 4
        x2 = np.array([1, 1, 1, 1])
        d = 0.1
        gd = create_gd(x2, n, d)
        assert size_max_independent_set(n, gd) == 1
        assert number_of_connectivity_components(gd) == 1

        x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        k = 3
        gk = create_gk(x, n, k)
        assert max_degree(n, gk) == n - 1
        assert size_max_clique(gk) == n


if __name__ == "__main__":
    unittest.main()
