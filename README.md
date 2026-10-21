# 连分数图固定顶点出度：强大数定律与中心极限定理的数值实验

这是依据用户提供的《连分数图出度与 Stein 方法本科生讲课稿》编写的第一版实验。
目标是展示理论的有限样本表现、检查模型与实现，不是通过模拟证明定理，也不是对讲稿证明的完整审稿。

如果下载的版本包含示例结果，可用浏览器打开 `results/demo/report.html`；没有示例时按下方步骤运行生成报告。
报告离线可用，SVG 图片保存在报告同级的 `figures` 文件夹。

**点击运行后可以自己选择 u。** 在 VS Code 的终端运行 `simulation.py`，无命令行参数且终端支持输入时，会依次询问 u、转移次数 m 和独立轨道数。

```text
选择 u：1-10
转移次数 m：100 300 1000 3000 10000
独立轨道数 repetitions：500
```

`1-10` 表示从 1 到 10 的十个顶点；也可输入 `1 3 5` 或 `1,3,5`，重复值会去重，程序显示实际数量。
每项直接按回车可使用显示的默认值。没有指定 `--output` 时，每次运行自动在脚本旁的 `results/run-日期-时间/` 下新建目录，不必反复修改源码中的路径。
运行结束后用浏览器打开终端打印的结果目录内的 `report.html`。

报告先展示**所有选定 u 的同图对比**：平均出度、归一化均值、归一化方差、第一条轨道的前缀、经验 W1、CDF 和 Q-Q。
每条彩色曲线代表一个 u，真实模型与代理模型分开画；代理模型不同 m 独立，故不绘制代理模型的前缀路径图。
十个 u 使用十种颜色，更多 u 使用颜色和线型组合；图例高度自动调整。后面继续保留每个 u 的详细图表。
路径汇总图比较同一条模拟轨道上不同 u 的出度，不能把不同 u 当作独立轨道。各 u 的标准化使用其各自的理论常数。

## 1. 我们模拟什么

从 Gauss 平稳分布生成有依赖的连分数数字 `a1,a2,...`。
固定顶点 `u`，观察前 `m` 次相邻转移 `(a1,a2),...,(am,a[m+1])`。
`D_m` 是其中以 `u` 为起点的**不同终点数**，重复边只计一次，自环计入。

例如数字 `2,1,2,3,2,1,2,2` 有 7 次转移，顶点 2 的终点集合是 `{1,2,3}`，所以出度是 3。

讲稿的常数为

$$c_u=\frac{1}{(\log2)u(u+1)},\quad
\alpha_u=\sqrt{\pi c_u},\quad
\sigma_u^2=(\sqrt2-1)\alpha_u.$$

实验查看：

$$D_m/\sqrt m\longrightarrow\alpha_u,$$

$$\mathbb E D_m=\alpha_u\sqrt m+O_u(1),\qquad
\operatorname{Var}(D_m)\sim\sigma_u^2\sqrt m,$$

$$Z_m=\frac{D_m-\alpha_u\sqrt m}{\sigma_u m^{1/4}}
\Rightarrow N(0,1).$$

这里 `sigma_u` 是 `sigma_u²` 的正平方根。`m` 是转移次数，不是数字个数。
每一个 `u` 都固定；同时画 `u=1,2,5` 不表示检验了 `u` 随 `m` 增长的结论。

## 2. 文件分别有什么用

| 文件 | 作用 |
|---|---|
| `simulation.py` | 生成真实模型和 Poisson 占盒代理，统计结果 |
| `plotting.py` | 从已保存的样本生成 SVG 图和离线报告 |
| `test_simulation.py` | 检查出度计数、概率公式、生成器和可重复性 |
| `requirements.txt` | 所需 Python 包 |
| `METHODS.md` | 解释条件生成器为何保留真实依赖，以及数值限制 |
| `results/demo/` | 已执行的一轮示例结果 |

本项目用 CPU 即可运行，不需要 CMake、make、Docker 或 GPU。
随机模拟由 NumPy 完成，静态 SVG 科研图由 ReportLab 图表组件生成。

## 3. 在 Windows 上运行

先在 **VS Code 的本地窗口**打开这个项目文件夹，再选择“终端 → 新建终端”。
终端当前目录应当是能看到 `simulation.py` 的目录。
这里使用你自己的 Windows 电脑，无需进入之前运行 CT 的远程容器。

如果当前选中的 Python 已经安装依赖，可以直接点击 `simulation.py` 右上角的“在终端中运行 Python 文件”，根据提示输入。
若运行环境没有可交互终端，可显式运行 `python simulation.py --interactive`；批处理可用 `--no-interactive`。
只要提供普通命令行参数（如 `--u`），就不会自动询问输入，原来的命令行用法仍然可用。

第一次运行，创建单独的 Python 环境并安装依赖：

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果电脑找不到 `py`，而 `python --version` 能显示 Python 3.10 或更新版本，
则把第一行的 `py` 换成 `python`。依赖安装需要联网。以后用下面的命令运行即可。

先做一个很小的实验：

```powershell
.\.venv\Scripts\python.exe simulation.py --u 1 --m 100 1000 10000 --repetitions 300 --output results/my-first-run
```

完成后，用浏览器打开 `results/my-first-run/report.html`。

| 参数 | 含义 |
|---|---|
| `--u 1` | 研究固定顶点 1 |
| `--m 100 1000 10000` | 在同一轨道的这三个前缀长度记录出度 |
| `--repetitions 300` | 生成 300 条独立轨道 |
| `--seed 20261021` | 控制伪随机序列，默认已经设置 |
| `--output ...` | 本轮结果文件夹，必须为空或尚不存在 |

例如，通过命令行选择十个 u（无需手工修改 Python 文件）：

```powershell
.\.venv\Scripts\python.exe simulation.py --u 1 2 3 4 5 6 7 8 9 10 --m 100 300 1000 3000 10000 --repetitions 500
```

`1-10` 简写用于交互输入；命令行 `--u` 后使用空格分隔的整数。自动结果目录使用本机时间命名。

重现附带示例的参数（另选目录，避免覆盖）：

```powershell
.\.venv\Scripts\python.exe simulation.py --output results/demo-repeat
```

默认参数仍是 `u=1,2,5`，`m=100,300,1000,3000,10000,30000`，独立轨道 1000 条。
同一环境、相同参数和种子可以重现结果；环境版本记录在 `metadata.json`。
没有固定不同版本 NumPy 间的逐位一致性承诺。

想增加样本量与重复次数，可以例如运行：

```powershell
.\.venv\Scripts\python.exe simulation.py --u 1 2 5 --m 1000 10000 100000 --repetitions 3000 --seed 42 --output results/larger-seed42
```

增大 `m` 观察渐近行为；增大 `repetitions` 减少 Monte Carlo 波动。这是两个不同的操作。
运行耗时、内存随参数增加；终端会在每个检查点显示进度。

## 4. 如何看图

1. **路径图**：前五条轨道的 `D_m/(alpha_u*sqrt(m))`，参考线是 1。
   每条曲线的不同点来自同一条轨道，不能为每个 `m` 重新独立模拟后冒充样本路径。
2. **均值和方差图**：分别除以理论渐近主项，观察比值是否接近 1。
   小 `m` 时主项不是精确均值或精确方差。
3. **CDF 图**：标准化出度的经验分布函数与标准正态比较，同时展示代理模型。
4. **Q-Q 图**：经验分位数与正态分位数比较。接近对角线表示分布较接近。

所有 CLT 图使用理论 `alpha_u` 和 `sigma_u`。
程序**没有**用样本均值、样本标准差把偏移与尺度偏差消掉。
标准化后的有限样本均值不一定是 0，有限样本方差不一定是 1。

报告与 CSV 中 `empirical_w1_to_normal` 是经验分布到标准正态的 W1 距离，
包括有限重复次数造成的随机误差，不能直接当成真实分布的 W1，也不能用它声称验证了
讲稿里的 `O((log m)^2/m^(1/4))` 理论界。理论上界不是精确的等价式。

`mean_mc_se` 是样本均值的 Monte Carlo 标准误。
第一版未计算方差和 W1 的置信区间；论文级实验应增加独立种子批次和适当的误差分析。
同一批轨道在不同 `m`、不同 `u` 的结果相互关联，不应当作额外独立重复。

## 5. 为什么不能简单独立抽连分数数字

真实连分数数字不独立。程序使用平稳自然扩张导出的条件抽样：
初始隐状态 `Y0` 服从 Gauss 分布；给定状态 `y` 后抽数字，然后更新状态。
每一步虽然使用新随机数，但条件分布依赖之前留下的状态。
它在精确算术下生成与讲稿模型相同的有限维数字分布，数值实现使用 float64。
它不追踪某个用户给定初始实数的确定轨道。

这样避免长期迭代 `x = 1/x - floor(1/x)` 放大舍入误差；完整推导见 `METHODS.md`。
不把数字裁剪到任意最大值，不把独立代理冒充真实模型。

代理模型通过总 Poisson 点数及随机标签抽样，等价于各标签独立 Poisson 计数后去重，
而非直接生成一个 Poisson 出度。代理在不同 `(u,m)` 上独立生成，不用来画强律样本路径。

## 6. 保存了什么，怎样重画

- `samples.npz`：各 `u,m` 下每条轨道的出度；未保存所有原始连分数数字。
- `summary.csv`：均值、方差、主项比值、标准化均值、经验 W1 等。可用 Excel 打开。
- `metadata.json`：参数、种子、环境版本及模拟/绘图代码 SHA-256。
- `figures/*.svg`：可缩放图，图内用英文标签避免中文字体缺失。
- `report.html`：中文解释与图形。

只重新画图，无需重跑模拟：

```powershell
.\.venv\Scripts\python.exe plotting.py results/my-first-run
```

运行实现检查：

```powershell
.\.venv\Scripts\python.exe -m unittest -v
```

测试包括讲稿的计数例子、从柱集端点独立计算的概率、条件抽样的联合分布积分、
20 万条独立短轨道的单数字/双数字/三数字频率，以及同种子复现。
通过测试不等于证明渐近定理。

## 7. 放进你刚建的 GitHub 仓库

先上传 `simulation.py`、`plotting.py`、`test_simulation.py`、`requirements.txt`、`METHODS.md`，
再用本 README 的内容更新仓库说明。
网页操作可用 **Add file → Upload files**，完成后提交修改。
同名文件先查看差异，避免不小心覆盖已有内容。

`.gitignore` 排除 `.venv`、Python 缓存及 `results/`，防止默认把大量实验文件提交。
如果想分享一个精选示例，可以另建 `examples/demo/`，放入报告、相关 SVG、CSV 和参数记录；
报告与 `figures` 的相对位置要保持一致。GitHub 普通文件页面展示 HTML 源码，
下载后用浏览器打开才是完整报告。

本次只创建本地项目，没有向 GitHub 推送，也没有修改你已经创建的远程仓库。
讲稿 PDF 没有复制进项目；项目的公开范围与许可证由作者决定。

学习记录: 已成功克隆仓库， 开始练习Git。