"""确定性数学校验 + 固定种子的抽样诊断；不是定理检验。"""
import math
from fractions import Fraction
import unittest

import numpy as np

from simulation import (LOG2, conditional_labels, digit_probability, edge_probability,
                        edge_tail, empirical_normal_w1, initial_state, next_digit,
                        outdegrees_from_digits, simulate_gauss)


def cylinder_probability(word):
    # 从连分数收敛分数独立计算柱集端点，再积分 Gauss 密度。
    p0, p1, q0, q1 = 1, 0, 0, 1
    for a in word:
        p0, p1, q0, q1 = p1, a*p1+p0, q1, a*q1+q0
    left, right = sorted([Fraction(p1, q1), Fraction(p1+p0, q1+q0)])
    return math.log(float((1+right)/(1+left))) / LOG2


class ModelTests(unittest.TestCase):
    def test_count_duplicates_self_loop_and_transition_index(self):
        digits = [2, 1, 2, 3, 2, 1, 2, 2]
        self.assertEqual(outdegrees_from_digits(digits, 2, [1, 3, 5, 7]), [1, 2, 2, 3])
        self.assertEqual(outdegrees_from_digits([3, 1, 3, 1, 3, 3], 3, [5]), [2])
        self.assertEqual(outdegrees_from_digits([1, 1], 1, [1]), [1])

    def test_edge_formula_against_cylinder(self):
        for u in [1, 2, 5, 10]:
            for v in [1, 2, 7, 50]:
                self.assertAlmostEqual(float(edge_probability(u, v)), cylinder_probability([u, v]), places=14)

    def test_tail_identity(self):
        for u in [1, 2, 5]:
            self.assertAlmostEqual(edge_tail(u, 1), digit_probability(u), places=14)
            for k in [1, 5, 100]:
                self.assertAlmostEqual(edge_tail(u, k)-edge_tail(u, k+1), float(edge_probability(u, k)), places=14)

    def test_conditional_sampler_joint_law_by_quadrature(self):
        # 不使用 Monte Carlo：积分生成器的柱集概率，和端点公式比较。
        nodes, weights = np.polynomial.legendre.leggauss(128)
        y0 = (nodes+1)/2
        for word in [[1], [2], [1, 1], [1, 7], [2, 1], [1, 2, 1], [3, 1, 3, 3]]:
            y = y0.copy()
            density = 1/(LOG2*(1+y))
            for a in word:
                density *= (1+y)/((a+y)*(a+1+y))
                y = 1/(a+y)
            actual = float(weights @ density)/2
            self.assertAlmostEqual(actual, cylinder_probability(word), places=13)

    def test_generated_single_pair_and_triple_probabilities(self):
        repetitions = 200000
        rng = np.random.default_rng(87123)
        y = initial_state(rng, repetitions)
        digits = []
        for _ in range(4):
            a, y = next_digit(y, rng)
            digits.append(a)
        for word in [[1], [5], [1, 1], [1, 2], [2, 1], [1, 2, 1], [2, 3, 1]]:
            mask = np.ones(repetitions, dtype=bool)
            for a, observed in zip(word, digits):
                mask &= observed == a
            expected = cylinder_probability(word)
            se = math.sqrt(expected*(1-expected)/repetitions)
            self.assertLess(abs(float(mask.mean())-expected), 6*se+1/repetitions)
        # 后续位置也应保持同样的单数字边缘分布。
        for observed in digits[1:]:
            expected = digit_probability(1)
            self.assertLess(abs(float((observed == 1).mean())-expected),
                            6*math.sqrt(expected*(1-expected)/repetitions))

    def test_proxy_conditional_tail(self):
        rng = np.random.default_rng(9417)
        n = 200000
        for u in [1, 5]:
            labels = conditional_labels(u, rng, n)
            for k in [2, 10, 100]:
                expected = edge_tail(u, k)/digit_probability(u)
                se = math.sqrt(expected*(1-expected)/n)
                self.assertLess(abs(float((labels >= k).mean())-expected), 6*se+1/n)

    def test_simulation_reproducibility_bounds_and_m1(self):
        kwargs = dict(vertices=[1, 2], checkpoints=[1, 5, 20], repetitions=20000, progress=False)
        a = simulate_gauss(**kwargs, rng=np.random.default_rng(123))
        b = simulate_gauss(**kwargs, rng=np.random.default_rng(123))
        for u in a:
            np.testing.assert_array_equal(a[u], b[u])
            self.assertTrue(np.all(np.diff(a[u], axis=0) >= 0))
            self.assertTrue(np.all(a[u] <= np.array([1, 5, 20])[:, None]))
            p = digit_probability(u)
            self.assertLess(abs(a[u][0].mean()-p), 6*math.sqrt(p*(1-p)/20000))

    def test_wasserstein_including_normal_tails(self):
        self.assertAlmostEqual(empirical_normal_w1([0]), math.sqrt(2/math.pi), places=12)
        self.assertAlmostEqual(empirical_normal_w1([-2, 0, 1]),
                               empirical_normal_w1([-1, 0, 2]), places=12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
