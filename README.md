# A股月频多因子量化项目

本项目直接读取 `data/*.zip`，不解压原始文件。第一版基线策略使用：

- 动量：过去 12 个月收益，跳过最近 1 个月（12-1 momentum）
- 价值：PB、PE、PS 的倒数综合
- 质量：ROE、ROA、毛利率、现金利润比、低杠杆（严格按财报公告日生效）
- 风格：低波动、小市值、一个月反转、低非流动性
- 股票池：A股、非 ST、正常交易、上市满 120 个交易日、月均成交额不低于横截面 20% 分位
- 组合：每月等权持有综合得分最高的 10%，信号滞后一个月，单边交易成本默认 20bp
- 增强：证监会行业内标准化；前10%建仓、跌出前15%才卖出；组合波动率超过15%时动态降仓
- 基准：沪深300（000300）

## 运行

```powershell
python quant_project.py --start 2016-01-01 --end 2025-12-31
```

首次运行会扫描压缩包，时间较长。结果写入 `output/`：

- `monthly_panel.csv.gz`：月频股票面板
- `backtest_monthly.csv`：策略与基准月收益、净值、换手率
- `metrics.csv`：年化收益、波动率、夏普、最大回撤等
- `factor_ic.csv`：各因子的月度截面 IC 与 ICIR
- `nav.png`：净值曲线（环境已安装 matplotlib 时生成）

快速验证可先运行：

```powershell
python quant_project.py --start 2022-01-01 --end 2025-12-31 --no-quality
```

## 研究口径

月末收盘后形成信号，下一个月持有。估值数据取月末最后可得值；财务指标只有在 `Annodt` 公告日之后才能进入模型，因此不会用报告期末日期直接回填。当前版本为研究基线，不代表实盘收益；涨跌停无法成交、冲击成本和组合容量仍需在后续版本中精细建模。

## 机器学习走步回测

```powershell
python ml_quant.py --start 2018-01-01 --train-months 60
```

机器学习版本使用行业内横截面因子、平方项和交互项，按季度进行走步式多项式岭回归。正则强度仅用训练窗口末尾12个月验证集选择，结果写入 `ml_backtest_monthly.csv`、`ml_metrics.csv` 和 `ml_model_log.csv`。

稳健性检验运行 `python robustness.py`，输出年度、成本、参数敏感性及区块自助法结果。

## 新版：TabM 排序集成（2026-07）

`advanced_quant.py` 面向月度截面选股加入了两类较新的表格学习方法：

- **TabM 式参数高效集成**：共享主要权重、保留成员专属缩放参数，在一次前向计算中产生多个弱学习器；模型采用平滑回归损失与成对排序损失共同训练。
- **TabR 式历史相似样本检索**：只从预测月之前的训练窗口中寻找近邻，由相似历史股票月份的收益排名提供局部修正。这里是适合 CPU 回测的轻量研究实现，并非官方 TabR 的逐行复刻。
- **稳定锚点**：把 84 个月窗口的多项式岭回归作为低方差基准，三类模型的权重只用训练窗口末尾 12 个月验证集的月均 Rank IC 决定。
- **新增动态因子**：3/6 月动量、动量加速度、6/12 月收益稳定性、规模变化、换手代理、极端收益质量、价值与质量综合项。所有变量均只使用信号月及以前数据。

参考：[TabM（ICLR 2025）](https://openreview.net/forum?id=Sd4wYYOhmY)、[TabM 官方代码](https://github.com/yandex-research/tabm)、[TabR（ICLR 2024）](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4ef594af0d9a519db8fb292452c461fa-Abstract-Conference.html)。

### 完整走步回测

先运行 `quant_project.py` 生成未纳入 Git 的 `output/monthly_panel.csv.gz`，然后：

```powershell
python -m pip install -r requirements.txt
python advanced_quant.py --start 2018-01-01
python advanced_robustness.py
python -m unittest -q test_advanced_quant.py
```

默认配置针对普通 CPU 做过完整样本验证；如需快速检查，可加 `--quick`。主要输出为 `advanced_backtest.csv`、`advanced_metrics.csv`、`advanced_model_log.csv`、`advanced_config.json` 及三份稳健性检验 CSV。

### 同口径结果（信号期 2018-01 至 2025-11，收益期 2018-02 至 2025-12，20bp 成本）

| 策略 | 年化收益 | 年化波动 | 夏普（rf=0） | 最大回撤 | 总收益 | 平均月换手 |
|---|---:|---:|---:|---:|---:|---:|
| TabM 排序检索集成 | 11.51% | 19.62% | 0.587 | -24.68% | 136.91% | 41.28% |
| 上一版 MLP + 岭回归 | 10.83% | 19.03% | 0.569 | -24.97% | 125.73% | 47.20% |
| 沪深300 | 1.01% | 18.39% | 0.055 | -39.92% | 8.28% | — |

新版相对上一版年化收益提高约 0.68 个百分点、总收益提高约 11.19 个百分点，同时月换手下降约 5.92 个百分点。结果仍是历史回测，不等于未来收益，也没有完全模拟涨跌停排队、冲击成本和资金容量。
