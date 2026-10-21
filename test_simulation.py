"""确定性数学校验 + 固定种子的抽样诊断；不是定理检验。"""
import math
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

from simulation import (LOG2, conditional_labels, digit_probability, edge_probability,
                        edge_tail, empirical_normal_w1, initial_state, next_digit,
                        outdegrees_from_digits, simulate_gauss, summarize,
                        parse_positive_values, parse_options)


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


class UserInputTests(unittest.TestCase):
    def test_values_ranges_duplicates_and_invalid_input(self):
        self.assertEqual(parse_positive_values('1-10'), list(range(1, 11)))
        self.assertEqual(parse_positive_values('5，1,3 3; 7-9'), [1, 3, 5, 7, 8, 9])
        for text in ['', '0', '-1', '3-1', '1.5', 'abc', '1-100000000']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_positive_values(text)

    def test_interactive_retry_and_selected_count(self):
        with tempfile.TemporaryDirectory() as folder:
            responses = ['0', '1-10', '30,10,30', '1', '20']
            with patch('builtins.input', side_effect=responses), patch('sys.stdout', new_callable=io.StringIO):
                args = parse_options(['--interactive', '--output', folder])
            self.assertEqual(args.u, list(range(1, 11)))
            self.assertEqual(args.m, [10, 30])
            self.assertEqual(args.repetitions, 20)

    def test_no_arguments_prompt_only_in_terminal(self):
        with patch('sys.stdin.isatty', return_value=True), patch('builtins.input', side_effect=['3', '', '10']), \
                patch('sys.stdout', new_callable=io.StringIO):
            args = parse_options([])
        self.assertEqual(args.u, [3])
        self.assertEqual(args.repetitions, 10)
        with patch('sys.stdin.isatty', return_value=False), patch('builtins.input') as input_mock:
            args = parse_options([])
        input_mock.assert_not_called()
        self.assertEqual(args.u, [1, 2, 5])
        self.assertFalse(args.output.exists())
        self.assertEqual(args.output.parent, Path(__file__).resolve().parent/'results')

    def test_cli_is_noninteractive_and_existing_results_are_protected(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('builtins.input') as input_mock:
                args = parse_options(['--u', '9', '2', '9', '--m', '30', '10', '--output', folder])
            input_mock.assert_not_called()
            self.assertEqual(args.u, [2, 9])
            self.assertEqual(args.m, [10, 30])
            saved = Path(folder)/'existing.txt'
            saved.write_text('keep', encoding='utf-8')
            with patch('sys.stderr', new_callable=io.StringIO), self.assertRaises(SystemExit) as error:
                parse_options(['--output', folder])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(saved.read_text(encoding='utf-8'), 'keep')


class ComparisonTests(unittest.TestCase):
    def test_curves_include_all_vertices_with_individual_normalization(self):
        from plotting import build_comparison_curves
        vertices, m = list(range(1, 11)), [10, 30]
        data = {u: np.array([[0, u, 2*u], [u, 2*u, 3*u]], dtype=float) for u in vertices}
        samples = {'m': np.array(m), **{f'gauss_u{u}': d for u, d in data.items()}}
        curves = build_comparison_curves(samples, vertices, 'gauss', summarize(data, 'gauss', m))
        self.assertEqual([label for label, _ in curves['mean']], [f'u={u}' for u in vertices])
        self.assertEqual(len(curves['cdf']), 11)
        for i, u in enumerate(vertices):
            alpha = math.sqrt(math.pi/(math.log(2)*u*(u+1)))
            self.assertAlmostEqual(curves['mean_ratio'][i][1][0][1], u/(alpha*math.sqrt(10)))
            self.assertAlmostEqual(curves['paths'][i][1][1][1], u/(alpha*math.sqrt(30)))
        proxy_samples = {'m': np.array(m), **{f'poisson_proxy_u{u}': d for u, d in data.items()}}
        proxy = build_comparison_curves(proxy_samples, vertices, 'poisson_proxy',
                                       summarize(data, 'poisson_proxy', m))
        self.assertNotIn('paths', proxy)  # independent proxy checkpoints must not be presented as nested paths

    def test_ten_vertex_interactive_report_end_to_end(self):
        with tempfile.TemporaryDirectory() as folder:
            command = [sys.executable, str(Path(__file__).with_name('simulation.py')),
                       '--interactive', '--output', folder]
            result = subprocess.run(command, input='1-10\n10 30\n20\n', text=True,
                                    encoding='utf-8', capture_output=True, timeout=120,
                                    env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            output = Path(folder)
            metadata = json.loads((output/'metadata.json').read_text(encoding='utf-8'))
            self.assertEqual(metadata['vertices'], list(range(1, 11)))
            self.assertEqual(metadata['vertex_count'], 10)
            report = (output/'report.html').read_text(encoding='utf-8')
            for src in re.findall(r'<img src="([^"]+)"', report):
                self.assertTrue((output/src).is_file(), src)
            for model in ['gauss', 'poisson_proxy']:
                svg = ET.parse(output/'figures'/f'compare_{model}_cdf.svg')
                labels = set(svg.getroot().itertext())
                for u in range(1, 11):
                    self.assertIn(f'u={u}', labels)
                self.assertIn('Standard normal', labels)
            self.assertGreater(float(svg.getroot().attrib['height']), 460)

    def test_single_vertex_single_checkpoint_report(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, str(Path(__file__).with_name('simulation.py')),
                                     '--u', '3', '--m', '10', '--repetitions', '5', '--skip-proxy',
                                     '--output', folder], text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertTrue((Path(folder)/'figures'/'compare_gauss_mean.svg').is_file())
            self.assertFalse((Path(folder)/'figures'/'compare_poisson_proxy_cdf.svg').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
