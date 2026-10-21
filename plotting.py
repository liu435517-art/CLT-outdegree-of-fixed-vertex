"""用 ReportLab 的标准图表组件绘制可缩放 SVG，并生成离线中文报告。

也可独立运行：python plotting.py results/demo
"""
import csv
import hashlib
import html
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys

import numpy as np
import reportlab
from reportlab.graphics import renderSVG
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.graphics.widgets.markers import makeMarker
from reportlab.lib.colors import HexColor

from simulation import constants

COLORS = ['#176B9D', '#E48A28', '#238A68', '#B44C66', '#8263AB',
          '#67788A', '#A37600', '#007F86', '#884E2F', '#5954BD']
NORMAL = NormalDist()


def chart(path, title, curves, xlabel, ylabel, ylim=None):
    """curves = [(legend, [(x,y), ...]), ...]。"""
    # 图例在绘图区上方；随曲线数量自动加高，十个 u 也不会压住曲线。
    legend_rows = max(1, math.ceil(len(curves)/3))
    height = 440 + 20*legend_rows
    drawing = Drawing(850, height)
    drawing.add(Rect(0, 0, 850, height, fillColor=HexColor('#FFFFFF'), strokeColor=None))
    drawing.add(String(65, height-28, title, fontName='Helvetica-Bold', fontSize=16,
                       fillColor=HexColor('#16334A')))
    plot = LinePlot()
    plot.x, plot.y, plot.width, plot.height = 75, 75, 690, 280
    plot.data = [points for _, points in curves]
    plot.joinedLines = 1
    plot.xValueAxis.labels.fontSize = 10
    plot.yValueAxis.labels.fontSize = 10
    plot.xValueAxis.labelTextFormat = '%.2f'
    plot.yValueAxis.labelTextFormat = '%.2f'
    plot.xValueAxis.maximumTicks = 7
    plot.yValueAxis.maximumTicks = 7
    plot.yValueAxis.visibleGrid = 1
    plot.yValueAxis.gridStrokeColor = HexColor('#E4EAF0')
    all_x = [x for _, points in curves for x, _ in points]
    if all_x and min(all_x) == max(all_x):
        plot.xValueAxis.valueMin = all_x[0]-0.5
        plot.xValueAxis.valueMax = all_x[0]+0.5
    if ylim is not None:
        plot.yValueAxis.valueMin, plot.yValueAxis.valueMax = ylim
    data_index = 0
    dash_patterns = [None, [6, 3], [2, 3], [8, 3, 2, 3]]
    for i, (label, points) in enumerate(curves):
        reference = label.startswith(('Reference', 'Limit', 'Asymptotic', 'Standard normal'))
        if reference:
            color, dash = HexColor('#202B33'), [6, 3]
        else:
            color = HexColor(COLORS[data_index % len(COLORS)])
            dash = dash_patterns[(data_index // len(COLORS)) % len(dash_patterns)]
            data_index += 1
        plot.lines[i].strokeColor = color
        plot.lines[i].strokeWidth = 1.7
        if dash:
            plot.lines[i].strokeDashArray = dash
        if len(points) == 1 or len(set(points)) == 1:
            plot.lines[i].symbol = makeMarker('FilledCircle', size=5, fillColor=color, strokeColor=color)
        lx, ly = 75 + (i % 3)*250, height-60 - (i // 3)*20
        drawing.add(Line(lx, ly, lx+22, ly, strokeColor=color, strokeWidth=2, strokeDashArray=dash))
        drawing.add(String(lx+29, ly-3, label, fontSize=10))
    drawing.add(plot)
    drawing.add(String(420, 27, xlabel, textAnchor='middle', fontSize=11))
    drawing.add(String(75, 365, ylabel, fontSize=10))
    renderSVG.drawToFile(drawing, str(path))
    return drawing


def empirical_cdf(z, xmin, xmax):
    values, counts = np.unique(z, return_counts=True)
    points, before = [(xmin, 0.0)], 0.0
    for value, count in zip(values, counts):
        after = before + count/len(z)
        points.extend([(float(value), before), (float(value), after)])
        before = after
    points.append((xmax, 1.0))
    return points


def build_comparison_curves(samples, vertices, model, summary_rows):
    """同一模型中一条曲线对应一个 u；各 u 分别使用自己的理论标准化常数。"""
    m = samples['m']
    x = np.log10(m)
    curves = {key: [] for key in ['mean', 'mean_ratio', 'variance_ratio', 'w1', 'cdf', 'qq']}
    if model == 'gauss':
        curves['paths'] = []
    w1 = {(int(row['u']), int(row['m'])): float(row['empirical_w1_to_normal'])
          for row in summary_rows if row['model'] == model}
    standardized = {}
    for u in vertices:
        d = samples[f'{model}_u{u}']
        _, alpha, var_c = constants(u)
        label = f'u={u}'
        mean, variance = d.mean(axis=1), d.var(axis=1, ddof=1)
        curves['mean'].append((label, list(zip(x, mean))))
        curves['mean_ratio'].append((label, list(zip(x, mean/(alpha*np.sqrt(m))))))
        curves['variance_ratio'].append((label, list(zip(x, variance/(var_c*np.sqrt(m))))))
        curves['w1'].append((label, [(float(t), w1[u, int(n)]) for t, n in zip(x, m)]))
        if model == 'gauss':
            curves['paths'].append((label, list(zip(x, d[:, 0]/(alpha*np.sqrt(m))))))
        z = (d[-1]-alpha*math.sqrt(int(m[-1]))) / math.sqrt(var_c*math.sqrt(int(m[-1])))
        standardized[u] = z
        probabilities = (np.arange(len(z))+0.5)/len(z)
        normal_q = [NORMAL.inv_cdf(float(p)) for p in probabilities]
        curves['qq'].append((label, list(zip(normal_q, np.sort(z)))))
    for key in ['mean_ratio', 'variance_ratio', 'paths']:
        if key in curves:
            curves[key].append(('Reference = 1', [(float(x[0]), 1), (float(x[-1]), 1)]))
    low = min(-4.0, *(float(z.min()) for z in standardized.values()))
    high = max(4.0, *(float(z.max()) for z in standardized.values()))
    for u, z in standardized.items():
        curves['cdf'].append((f'u={u}', empirical_cdf(z, low, high)))
    curves['cdf'].append(('Standard normal', [(float(t), NORMAL.cdf(float(t)))
                                            for t in np.linspace(low, high, 250)]))
    qlow = min(points[0][0] for _, points in curves['qq'])
    qhigh = max(points[-1][0] for _, points in curves['qq'])
    curves['qq'].append(('Reference y = x', [(qlow, qlow), (qhigh, qhigh)]))
    return curves


def comparison_report(samples, vertices, rows, figures):
    parts = ['<h2>所有 u 同图对比</h2>',
             '<p>每条彩色曲线代表一个 u，相同 u 在各汇总图中使用相同颜色。'
             '超过十个 u 后，颜色与线型组合区分曲线；黑色虚线是理论参考线。'
             '真实模型和代理模型分开作图。</p>']
    specs = [
        ('mean', 'Mean outdegree', 'log10(m)', 'Empirical mean of D_m',
         '不同 u 的平均出度，纵轴保留原始尺度。'),
        ('mean_ratio', 'Normalized mean', 'log10(m)', 'Mean / (alpha_u * sqrt(m))',
         '每个 u 的均值分别除以自己的渐近主项，参考值为 1。'),
        ('variance_ratio', 'Normalized variance', 'log10(m)', 'Variance / (sigma_u^2 * sqrt(m))',
         '每个 u 的样本方差分别除以自己的渐近主项，参考值为 1。'),
        ('paths', 'Nested prefixes: first path', 'log10(m)', 'D_m / (alpha_u * sqrt(m))',
         '同一条模拟轨道上不同 u 的出度；各 m 保留轨道前缀关系。此图只显示第一条轨道，不是样本均值。'),
        ('w1', 'Empirical normal W1', 'log10(m)', 'Empirical W1 distance',
         '各 u 的标准化经验分布到标准正态的 W1 距离，包含有限轨道数的抽样误差。'),
        ('cdf', 'CLT empirical CDF', 'Standardized outdegree Z', 'Cumulative probability',
         '在最大 m 处比较各 u 的经验分布。每个 u 使用自己的理论中心和尺度。'),
        ('qq', 'Normal Q-Q', 'Theoretical standard normal quantile', 'Observed standardized quantile',
         '在最大 m 处比较各 u 的正态 Q-Q 曲线，黑色虚线是 y=x。'),
    ]
    for model, name in [('gauss', 'Gauss'), ('poisson_proxy', 'Poisson proxy')]:
        if f'{model}_u{vertices[0]}' not in samples:
            continue
        curves = build_comparison_curves(samples, vertices, model, rows)
        parts.append(f'<h3>{name}：{len(vertices)} 个 u</h3>')
        for key, title, xlabel, ylabel, caption in specs:
            if key not in curves:
                continue
            filename = f'compare_{model}_{key}.svg'
            suffix = f' | m={int(samples["m"][-1]):,}' if key in ('cdf', 'qq') else ''
            chart(figures/filename, f'{name} | {title}{suffix}', curves[key], xlabel, ylabel,
                  (0, 1) if key == 'cdf' else None)
            parts.append(f'<figure><img src="figures/{filename}"><figcaption>{caption}</figcaption></figure>')
    return parts


def make_report(output):
    output = Path(output)
    metadata = json.loads((output/'metadata.json').read_text(encoding='utf-8'))
    with (output/'summary.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    figures = output/'figures'
    figures.mkdir(exist_ok=True)
    content = []
    with np.load(output/'samples.npz') as samples:
        m = samples['m']
        x = np.log10(m)
        content.extend(comparison_report(samples, metadata['vertices'], rows, figures))
        content.append('<h2>各 u 的详细结果</h2><p>下面保留逐个 u 的真实模型与代理模型对照。</p>')
        for u in metadata['vertices']:
            _, alpha, var_c = constants(u)
            real = samples[f'gauss_u{u}']
            proxy_key = f'poisson_proxy_u{u}'
            proxy = samples[proxy_key] if proxy_key in samples else None
            curves = []
            for name, d in [('Gauss', real), ('Proxy', proxy)]:
                if d is not None:
                    curves.extend([(f'{name} mean / main term', list(zip(x, d.mean(axis=1)/(alpha*np.sqrt(m))))),
                                   (f'{name} variance / main term', list(zip(x, d.var(axis=1, ddof=1)/(var_c*np.sqrt(m)))))])
            curves.append(('Asymptotic target = 1', [(float(x[0]), 1), (float(x[-1]), 1)]))
            chart(figures/f'u{u}_moments.svg', f'Mean and variance scaling | u={u}', curves,
                  'log10(m): 2 = 100, 3 = 1,000, 4 = 10,000', 'Ratio to asymptotic main term')

            paths = [(f'Path {i+1}', list(zip(x, real[:, i]/(alpha*np.sqrt(m)))))
                     for i in range(min(5, real.shape[1]))]
            paths.append(('Limit = 1', [(float(x[0]), 1), (float(x[-1]), 1)]))
            chart(figures/f'u{u}_paths.svg', f'Nested path prefixes | u={u}', paths,
                  'log10(m)', 'D_m / (alpha_u * sqrt(m))')

            last_m = int(m[-1])
            scale = math.sqrt(var_c*math.sqrt(last_m))
            z = (real[-1]-alpha*math.sqrt(last_m))/scale
            zp = (proxy[-1]-alpha*math.sqrt(last_m))/scale if proxy is not None else None
            low = min(-4.0, float(z.min()), float(zp.min()) if zp is not None else -4.0)
            high = max(4.0, float(z.max()), float(zp.max()) if zp is not None else 4.0)
            grid = np.linspace(low, high, 250)
            cdfs = [('Gauss model', empirical_cdf(z, low, high))]
            if zp is not None:
                cdfs.append(('Poisson occupancy proxy', empirical_cdf(zp, low, high)))
            cdfs.append(('Standard normal', [(float(t), NORMAL.cdf(float(t))) for t in grid]))
            chart(figures/f'u{u}_cdf.svg', f'CLT empirical CDF | u={u}, m={last_m:,}', cdfs,
                  'Z = (D_m - alpha_u * sqrt(m)) / (sigma_u * m^(1/4))', 'Cumulative probability', (0, 1))
            probabilities = (np.arange(len(z)) + 0.5)/len(z)
            normal_q = np.array([NORMAL.inv_cdf(float(p)) for p in probabilities])
            qq = [('Gauss quantiles', list(zip(normal_q, np.sort(z))))]
            if zp is not None:
                qq.append(('Proxy quantiles', list(zip(normal_q, np.sort(zp)))))
            qq.append(('Reference y = x', [(float(normal_q[0]), float(normal_q[0])),
                                          (float(normal_q[-1]), float(normal_q[-1]))]))
            chart(figures/f'u{u}_qq.svg', f'Normal Q-Q | u={u}, m={last_m:,}', qq,
                  'Theoretical standard normal quantile', 'Observed standardized outdegree quantile')
            content.append(f'<h2>固定顶点 u={u}</h2><p>α={alpha:.6f}，σ²={var_c:.6f}。</p>')
            for suffix, caption in [('paths', '前五条独立轨道的前缀：曲线接近 1 是强律的有限样本展示，不能证明几乎处处收敛。'),
                                    ('moments', '均值与方差除以各自的渐近主项。小样本偏差和 Monte Carlo 波动都可能使比值偏离 1。'),
                                    ('cdf', '按理论常数中心化、标准化；没有用样本均值和样本标准差强行校正。'),
                                    ('qq', '点接近对角线表示分位数接近正态；尾部的抽样波动更明显。')]:
                content.append(f'<figure><img src="figures/u{u}_{suffix}.svg"><figcaption>{caption}</figcaption></figure>')
            content.append('<table><tr><th>模型</th><th>m</th><th>均值比值</th><th>方差比值</th><th>Z 均值</th><th>经验 W₁</th></tr>')
            for row in rows:
                if int(row['u']) == u:
                    content.append('<tr>'+''.join(f'<td>{html.escape(v)}</td>' for v in
                                   [row['model'], row['m']]+[f'{float(row[k]):.4f}' for k in
                                    ['mean_ratio', 'variance_ratio', 'z_mean', 'empirical_w1_to_normal']])+'</tr>')
            content.append('</table>')
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>连分数图出度：数值实验</title><style>
body{max-width:1000px;margin:40px auto;padding:0 25px;background:#f5f8fb;color:#193247;font:16px/1.75 system-ui,sans-serif}
h1{font-size:30px}h2{margin-top:55px}figure{margin:24px 0;background:white;padding:12px;border-radius:12px}
img{width:100%;height:auto}figcaption{padding:0 30px 15px;color:#42566b}table{border-collapse:collapse;width:100%;background:white}
td,th{border-bottom:1px solid #dde5ed;padding:8px;text-align:right}th:first-child,td:first-child{text-align:left}
.note{background:#e8f0f8;padding:18px;border-radius:10px}code{background:#e3eaf1;padding:2px 5px}
</style><h1>连分数图固定顶点出度：数值实验</h1>'''
    document += f'<p>共 {len(metadata["vertices"])} 个 u：{metadata["vertices"]}。</p>'
    document += f'<p>独立轨道数：{metadata["repetitions"]}；随机种子：{metadata["seed"]}；转移次数：{metadata["checkpoints"]}。</p>'
    document += '''<div class="note">本实验以用户提供的讲稿为理论目标，采用平稳 Gauss 条件抽样生成有依赖的数字。
实验用于检查数值表现，不构成对定理或证明的认证。代理模型是独立 Poisson 计数的占盒数，不是 Poisson 出度。
不同 m 的真实样本来自同一批轨道的前缀；跨 m、跨顶点的统计量不是相互独立的。</div>'''
    document += ''.join(content)
    document += '''<h2>如何理解误差</h2><p>经验 W₁ 是有限样本的经验分布到标准正态的距离。
它包含 Monte Carlo 误差，可能不随 m 单调下降，不能据此认定理论 W₁ 速率已被验证。
summary.csv 中 mean_mc_se 是独立轨道样本均值的 Monte Carlo 标准误，不是渐近公式的误差界。
当前报告未提供方差或 W₁ 的置信区间。若用于论文，需要增加重复次数、独立种子批次及适当的不确定性分析。</p>
<p>原始出度在 samples.npz；汇总在 summary.csv；参数、环境和代码哈希在 metadata.json。
本页可以离线打开，图形是可缩放 SVG。</p></html>'''
    (output/'report.html').write_text(document, encoding='utf-8')
    # 重画时只更新绘图实现的记录，不改变原始模拟代码的哈希。
    metadata['source_sha256']['plotting.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata['reportlab'] = reportlab.Version
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    make_report(Path(sys.argv[1]))
