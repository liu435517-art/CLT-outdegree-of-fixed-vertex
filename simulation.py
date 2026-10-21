"""连分数图固定顶点出度实验。运行方法见 README.md。

默认模拟平稳 Gauss 连分数数字，保留相邻数字依赖。
一个 m 对应 m 次转移，即 m+1 个数字；重复终点只计一次。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
from statistics import NormalDist
import time

import numpy as np

LOG2 = math.log(2.0)
NORMAL = NormalDist()


def constants(u):
    c = 1.0 / (LOG2 * u * (u + 1))
    alpha = math.sqrt(math.pi * c)
    variance_coefficient = (math.sqrt(2) - 1) * alpha
    return c, alpha, variance_coefficient


def digit_probability(u):
    return math.log1p(1.0 / (u * (u + 2))) / LOG2


def edge_probability(u, v):
    v = np.asarray(v, dtype=float)
    return np.log1p(1.0 / ((u * v + u + 1) * ((u + 1) * v + 1))) / LOG2


def edge_tail(u, k):
    """P(a_1=u, a_2>=k)，不是条件概率。"""
    return math.log1p(1.0 / (u * ((u + 1) * k + 1))) / LOG2


def open_uniform(rng, size):
    # NumPy random 返回 [0,1)；拒绝零，不把极小值裁剪成固定阈值。
    r = rng.random(size)
    zeros = r == 0
    while np.any(zeros):
        r[zeros] = rng.random(int(zeros.sum()))
        zeros = r == 0
    return r


def initial_state(rng, size):
    # Gauss 分布的逆 CDF：F(y)=log(1+y)/log(2)。
    return np.expm1(LOG2 * open_uniform(rng, size))


def next_digit(y, rng):
    """条件抽样：P(a=k|y)=(1+y)/((k+y)(k+1+y))。

    更新 y'=1/(a+y)。这里 y 是保存过去信息的状态，不是 Gauss
    映射中的未来坐标 x。该构造在精确算术下生成正确的平稳数字律。
    """
    a_float = np.floor((1.0 + y) / open_uniform(rng, y.size) - y)
    if np.any(a_float < 1) or np.any(a_float >= 2**63):
        raise ArithmeticError('Digit out of int64 range; no silent truncation allowed.')
    a = a_float.astype(np.int64)
    return a, 1.0 / (a_float + y)


def outdegrees_from_digits(digits, u, checkpoints):
    """用于小例子和独立校验的直观实现。"""
    if max(checkpoints) >= len(digits):
        raise ValueError('m transitions require at least m+1 digits')
    seen, result = set(), []
    wanted = set(checkpoints)
    for t, (a, b) in enumerate(zip(digits[:-1], digits[1:]), start=1):
        if a == u:
            seen.add(int(b))
        if t in wanted:
            result.append(len(seen))
    return result


def simulate_gauss(vertices, checkpoints, repetitions, rng, progress=True):
    """轨道之间独立；同一轨道的不同检查点保留前缀关系。

    只存已发现的终点，不创建巨大邻接矩阵，不截断终点标签。
    """
    y = initial_state(rng, repetitions)
    previous, y = next_digit(y, rng)
    seen = {u: [set() for _ in range(repetitions)] for u in vertices}
    data = {u: np.empty((len(checkpoints), repetitions), dtype=np.int64)
            for u in vertices}
    check_index = {m: j for j, m in enumerate(checkpoints)}
    for t in range(1, checkpoints[-1] + 1):
        current, y = next_digit(y, rng)
        for u in vertices:
            for i in np.flatnonzero(previous == u):
                seen[u][i].add(int(current[i]))
        previous = current
        if t in check_index:
            j = check_index[t]
            for u in vertices:
                data[u][j] = [len(s) for s in seen[u]]
            if progress:
                print(f'Gauss model: m={t:,}, independent paths={repetitions}', flush=True)
    return data


def conditional_labels(u, rng, size):
    """完整条件分布 q_v/P(a_1=u) 的逆尾概率抽样，无标签截断。"""
    z = np.expm1(open_uniform(rng, size) * math.log1p(1.0 / (u * (u + 2))))
    labels = np.floor((1.0 / (u * z) - 1.0) / (u + 1))
    if np.any(labels < 1) or np.any(labels >= 2**63):
        raise ArithmeticError('Proxy label out of range; no silent truncation allowed.')
    return labels.astype(np.int64)


def simulate_proxy(vertices, checkpoints, repetitions, seed_sequence):
    """独立 Poisson 计数的占盒数。每个 (u,m) 独立模拟。

    K~Poi(m*p_u)，给 K 个点独立抽终点 q_v/p_u，去重。
    Poisson 分裂给出独立 N_v~Poi(m*q_v)。总点数与出度不同。
    """
    streams = iter(seed_sequence.spawn(len(vertices) * len(checkpoints)))
    data = {}
    for u in vertices:
        data[u] = np.empty((len(checkpoints), repetitions), dtype=np.int64)
        for j, m in enumerate(checkpoints):
            rng = np.random.default_rng(next(streams))
            counts = rng.poisson(m * digit_probability(u), repetitions)
            for i, count in enumerate(counts):
                data[u][j, i] = np.unique(conditional_labels(u, rng, int(count))).size
        print(f'Poisson occupancy proxy: u={u}', flush=True)
    return data


def empirical_normal_w1(values):
    """经验分布与 N(0,1) 的一维 W1，按分位数积分（含两侧无穷尾）。

    这是有限 Monte Carlo 样本的距离，不是真实分布距离的无偏估计。
    """
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    boundaries = np.arange(n + 1) / n
    q = np.array([NORMAL.inv_cdf(float(p)) for p in boundaries[1:-1]])
    density = np.r_[0.0, np.exp(-q*q/2) / math.sqrt(2*math.pi), 0.0]
    cdf = np.array([NORMAL.cdf(float(v)) for v in x])
    a, b = boundaries[:-1], boundaries[1:]
    c = np.clip(cdf, a, b)
    phi_c = np.where(cdf <= a, density[:-1],
                     np.where(cdf >= b, density[1:], np.exp(-x*x/2)/math.sqrt(2*math.pi)))
    return max(0.0, float(np.sum(x*(2*c-a-b) + 2*phi_c-density[:-1]-density[1:])))


def summarize(data, model, checkpoints):
    rows = []
    for u, values in data.items():
        _, alpha, var_c = constants(u)
        for m, d in zip(checkpoints, values):
            theoretical_mean, theoretical_variance = alpha*math.sqrt(m), var_c*math.sqrt(m)
            z = (d-theoretical_mean) / math.sqrt(theoretical_variance)
            empirical_variance = float(d.var(ddof=1))
            rows.append(dict(model=model, u=u, m=m, repetitions=len(d),
                             mean=float(d.mean()), variance=empirical_variance,
                             mean_mc_se=math.sqrt(empirical_variance/len(d)),
                             mean_main_term=theoretical_mean,
                             variance_main_term=theoretical_variance,
                             mean_ratio=float(d.mean())/theoretical_mean,
                             variance_ratio=empirical_variance/theoretical_variance,
                             z_mean=float(z.mean()), z_variance=float(z.var(ddof=1)),
                             empirical_w1_to_normal=empirical_normal_w1(z)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--u', nargs='+', type=int, default=[1, 2, 5], help='fixed vertices')
    parser.add_argument('--m', nargs='+', type=int, default=[100, 300, 1000, 3000, 10000, 30000],
                        help='numbers of transitions, not numbers of digits')
    parser.add_argument('--repetitions', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20261021)
    parser.add_argument('--output', type=Path, default=Path('results/demo'))
    parser.add_argument('--skip-proxy', action='store_true')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    vertices, checkpoints = sorted(set(args.u)), sorted(set(args.m))
    if min(vertices + checkpoints) < 1 or args.repetitions < 2 or args.seed < 0:
        parser.error('u,m must be positive; repetitions>=2; seed>=0')
    # 防止第二次运行不小心覆盖前一次实验。
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory is not empty; choose a new --output directory.')
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    real_seed, proxy_seed = np.random.SeedSequence(args.seed).spawn(2)
    models = {'gauss': simulate_gauss(vertices, checkpoints, args.repetitions,
                                      np.random.default_rng(real_seed))}
    if not args.skip_proxy:
        models['poisson_proxy'] = simulate_proxy(vertices, checkpoints, args.repetitions, proxy_seed)
    rows = [row for name, data in models.items() for row in summarize(data, name, checkpoints)]
    with (args.output/'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # 完整保存每条轨道在各检查点的出度；之后画图不需要重跑随机实验。
    np.savez_compressed(args.output/'samples.npz', m=np.array(checkpoints),
                        **{f'{name}_u{u}': a for name, data in models.items() for u, a in data.items()})
    code_files = ['simulation.py', 'plotting.py']
    metadata = dict(seed=args.seed, vertices=vertices, checkpoints=checkpoints,
                    repetitions=args.repetitions, python=platform.python_version(),
                    numpy=np.__version__, bit_generator='PCG64',
                    sampler='stationary Gauss natural-extension conditional sampler',
                    arithmetic='float64; finite PRNG precision, no explicit label cutoff',
                    elapsed_seconds=time.perf_counter()-start,
                    source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                   for name in code_files})
    (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    if not args.no_plots:
        from plotting import make_report
        make_report(args.output)
    print(f'Finished in {time.perf_counter()-start:.1f}s. Results: {args.output.resolve()}')


if __name__ == '__main__':
    main()
